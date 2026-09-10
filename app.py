# =========================================================
# A+B 통합 7버전
# 기준: 통합 6버전 유지 + 실시간 웹캠 포인트 고정 표시 영역 추가
# =========================================================
import os
import tempfile
import hashlib
import json
import random
from datetime import datetime, timedelta
import cv2
import streamlit as st
import torch
import pandas as pd
import altair as alt

# 🟢 [추가] YOLO-World 객체 탐지를 위해 사용
import numpy as np
import time

from PIL import Image, ImageOps
from transformers import AutoImageProcessor, SiglipForImageClassification

# 🟠 [수정] 직접 학습한 YOLO 모델 사용
from ultralytics import YOLO


# ---------------------------------------------------------
# [오프라인 설정]
# ---------------------------------------------------------

# 🔴 [삭제/주석처리]
# 최초 YOLO-World 모델 다운로드가 필요하므로
# 테스트 단계에서는 기존 오프라인 설정을 사용하지 않습니다.
#
# os.environ["HF_HUB_OFFLINE"] = "1"


# =========================================================
# 1. 페이지 설정
# =========================================================
st.set_page_config(
    page_title="재활용품 AI 분류",
    page_icon="♻️",
    layout="wide"
)

# 🟠 [수정]
# 기존:
# st.title("♻️ 재활용품 AI 분류 시스템")

st.title("♻️ 재활용품 AI 분류 시스템")

# 🟢 [추가]
st.caption(
    "YOLO로 여러 재활용품을 탐지하고 SigLIP2로 종류를 보조 판정합니다. "
    "판정이 불확실한 객체는 자동 재분석 후 필요 시 재투입 대상으로 처리합니다."
)


# =========================================================
# 🟢 [추가] AI 판정 기준
# =========================================================

# SigLIP2 최종 분류 신뢰도
CONFIDENCE_THRESHOLD = 0.50

# YOLO-World 객체 탐지 신뢰도
YOLO_CONFIDENCE = 0.25

# 🟢 [추가]
# YOLO가 이 신뢰도 이상이면
# SigLIP2로 다시 분류하지 않고 YOLO 결과를 우선 사용
YOLO_CLASSIFY_THRESHOLD = 0.80

# 🟢 [추가]
# YOLO와 SigLIP2가 서로 다를 때
# SigLIP2가 이 신뢰도 이상이면 SigLIP2 결과를 채택
SIGLIP_STRONG_THRESHOLD = 0.80

# Bounding Box 주변 여백
BOX_PADDING = 8

# 🟠 [수정]
# 기존 실시간 캠은 3프레임마다 추론
# → YOLO + SigLIP2 두 모델을 사용하므로 15프레임마다 분석
WEBCAM_FRAME_INTERVAL = 15

# 실시간 캠 기록 간격
WEBCAM_LOG_INTERVAL = 1.0

# 긴 동영상 메모리 과사용 방지
MAX_VIDEO_FRAMES = 100

# 🟢 [통합 3버전 추가]
# AI가 애매하게 판단한 객체는 사용자 수동 수정 대신 자동 재분석 1회 수행
REANALYSIS_ATTEMPTS = 1

# 🟢 [통합 4버전 추가]
# 동영상/실시간 웹캠에서 같은 물체가 여러 프레임에 반복 검출되어도
# 포인트는 객체 Tracking ID 기준으로 한 번만 적립합니다.
TRACKER_CONFIG = "botsort.yaml"

# 🟢 [통합 6버전 추가]
# 같은 물체를 YOLO가 겹치는 Bounding Box 여러 개로 잡았을 때
# 같은 최종 분류 라벨 + IoU 기준으로 하나만 남깁니다.
DUPLICATE_BOX_IOU_THRESHOLD = 0.65


# =========================================================
# 🟢 [통합 추가] app_jiwon.py 포인트 / 탄소중립 기능 설정
# =========================================================
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data_ref")
os.makedirs(DATA_DIR, exist_ok=True)

# app.py의 기존 최종 분류 라벨을 기준으로 포인트 정책을 연결합니다.
# 플라스틱/캔은 app_jiwon.py의 단가를 그대로 사용합니다.
# 종이는 app.py가 '종이팩'을 별도 구분하지 못하므로 자동 적립 대상에서 제외합니다.
APP_REWARD_INFO = {
    "플라스틱": {"rate": 20, "carbon_factor": 2.0, "avg_kg": 0.05},
    "캔/금속": {"rate": 50, "carbon_factor": 9.0, "avg_kg": 0.015},
}

DONATION_CAUSES = [
    "해양 정화 활동 지원",
    "도시 숲 조성 캠페인",
    "저소득층 냉방비 지원",
    "유기동물 보호소 후원",
]

CO2_TO_CAR_KM = 6.0
CO2_PER_TREE_YEAR = 22.0
CO2_PER_LED_HOUR = 0.01
MOCK_OTHER_USERS = 339
STATE_FILE = os.path.join(DATA_DIR, "user_state.json")

# =========================================================
# 🟢 [B팀원 기능 복원] 포인트 적립 가능 4종 표시용 정보
#    A팀원 AI 분류 로직은 변경하지 않습니다.
# =========================================================
B_CLASS_INFO = {
    "clear_pet":      {"label": "투명 페트병", "rate": 20},
    "plastic_bottle": {"label": "플라스틱병",   "rate": 20},
    "can":            {"label": "캔류",         "rate": 50},
    "carton":         {"label": "종이팩",       "rate": 30},
}

# B팀원 원본의 카드형 UI를 복원하되 A팀원 화면 구조는 유지합니다.
st.markdown("""
<style>
.eco-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 14px 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.eco-card .icon { font-size: 26px; }
.eco-card .label { font-size: 12px; margin-bottom: 2px; }
.eco-card .value { font-size: 15px; font-weight: 700; }
.success-box {
    background: #f0fff4;
    border: 1px solid #86efac;
    border-radius: 12px;
    padding: 16px 18px;
    color: #14532d;
    font-weight: 500;
}
.briefing-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 12px;
    padding: 18px 20px;
    color: #1e3a8a;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 2. 기존 SigLIP2 AI 모델 로드
# =========================================================
@st.cache_resource
def load_model():
    model_name = "prithivMLmods/Augmented-Waste-Classifier-SigLIP2"

    local_model_dir = os.path.join(
        os.path.dirname(__file__),
        "model"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    processor = AutoImageProcessor.from_pretrained(
        model_name,
        cache_dir=local_model_dir
    )

    model = SiglipForImageClassification.from_pretrained(
        model_name,
        cache_dir=local_model_dir
    )

    model.to(device)
    model.eval()

    return processor, model, device


processor, model, device = load_model()


# =========================================================
# 🟠 [수정] 2-1. Colab에서 직접 학습한 YOLO 모델 로드
# =========================================================
@st.cache_resource
def load_yolo_model():

    # 🟠 [수정]
    # Colab에서 직접 학습한 best.pt 사용
    model_path = os.path.join(
        os.path.dirname(__file__),
        "best.pt"
    )

    yolo_model = YOLO(model_path)

    return yolo_model


yolo_model = load_yolo_model()

# =========================================================
# 3. 이미지 분석 및 매핑 설정
# =========================================================
LABEL_MAP = {
    "Plastic": "플라스틱",
    "Paper": "종이",
    "Metal": "캔/금속",
    "Glass": "유리병",
    "Cardboard": "종이상자",
    "Clothes": "의류",
    "Battery": "건전지",
    "Biological": "생분해성/음식물",
    "Shoes": "신발",
    "Trash": "일반쓰레기"
}

# =========================================================
# 🟢 [추가] YOLO 42개 클래스 → 기존 10개 분류 기준 매핑
# =========================================================
YOLO_TO_10_MAP = {

    # 플라스틱
    "Combined plastic": "플라스틱",
    "Plastic bag": "플라스틱",
    "Plastic bottle": "플라스틱",
    "Plastic can": "플라스틱",
    "Plastic canister": "플라스틱",
    "Plastic caps": "플라스틱",
    "Plastic cup": "플라스틱",
    "Plastic shaker": "플라스틱",
    "Plastic shavings": "플라스틱",
    "Plastic toys": "플라스틱",
    "Stretch film": "플라스틱",
    "Unknown plastic": "플라스틱",
    "Zip plastic bag": "플라스틱",

    # 종이
    "Cellulose": "종이",
    "Paper bag": "종이",
    "Paper cups": "종이",
    "Paper shavings": "종이",
    "Paper": "종이",
    "Papier mache": "종이",
    "Tetra pack": "종이",

    # 종이상자
    "Cardboard": "종이상자",
    "Postal packaging": "종이상자",

    # 캔 / 금속
    "Aerosols": "캔/금속",
    "Aluminum can": "캔/금속",
    "Aluminum caps": "캔/금속",
    "Foil": "캔/금속",
    "Iron utensils": "캔/금속",
    "Metal shavings": "캔/금속",
    "Scrap metal": "캔/금속",
    "Tin": "캔/금속",

    # 유리
    "Glass bottle": "유리병",

    # 의류
    "Textile": "의류",

    # 생분해성 / 음식물
    "Organic": "생분해성/음식물"
}

RECYCLE_GUIDE = {
    "플라스틱":
        "내용물을 비우고 물로 헹군 뒤, "
        "라벨 스티커를 제거하여 투입하세요.",

    "종이":
        "물기에 젖지 않게 펼쳐서 배출하며, "
        "테이프/스프링은 제거하세요.",

    "캔/금속":
        "내용물을 비우고 가급적 압착하여 배출하세요. "
        "가스용기는 구멍을 뚫어야 합니다.",

    "유리병":
        "담배꽁초 등 이물질을 넣지 말고, "
        "병뚜껑을 분리하여 배출하세요.",

    "종이상자":
        "운송장 스티커와 테이프를 완전히 제거한 후 "
        "납작하게 접어 배출하세요.",

    "의류":
        "의류수거함에 배출하거나 오염이 심할 경우 "
        "일반쓰레기로 분류하세요.",

    "건전지":
        "폐건전지 전용 수거함에 별도로 배출하세요.",

    "생분해성/음식물":
        "물기를 최대한 제거한 후 "
        "음식물 쓰레기 전용 용기에 배출하세요.",

    "신발":
        "짝을 맞춰 신발 수거함에 배출하거나, "
        "훼손이 심하면 일반쓰레기로 배출하세요.",

    "일반쓰레기":
        "종량제 봉투에 담아 배출하세요."
}


# =========================================================
# 3-1. 기존 SigLIP2 이미지 분류
# =========================================================
def predict_trash(image):

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    results = {}

    for index, probability in enumerate(probabilities):

        raw_label = model.config.id2label[index]

        korean_label = LABEL_MAP.get(
            raw_label,
            raw_label
        )

        results[korean_label] = probability.item()

    return results


# =========================================================
# 🟠 [수정] 3-2. 직접 학습 YOLO + SigLIP2 보조 분석
# =========================================================
def detect_and_classify(image, use_tracking=False):

    """
    1. 직접 학습한 YOLO가 여러 물체의 위치 + 클래스 탐지
    2. YOLO 42개 클래스를 우리 10개 분류 기준으로 매핑
    3. YOLO 신뢰도 80% 이상이면 YOLO 결과 우선 사용
    4. 80% 미만이면 SigLIP2 보조 판정
    5. YOLO와 SigLIP2 결과가 다르면 수동 확인
    """

    image = image.convert("RGB")
    np_image = np.array(image)

    # =========================================================
    # YOLO 객체 탐지
    # =========================================================
    if use_tracking:
        # 🟢 [통합 4버전 추가]
        # 연속 프레임에서는 Tracking ID를 유지하여 동일 객체를 구분합니다.
        yolo_results = yolo_model.track(
            source=np_image,
            conf=YOLO_CONFIDENCE,
            iou=0.50,
            agnostic_nms=False,
            persist=True,
            tracker=TRACKER_CONFIG,
            verbose=False
        )
    else:
        yolo_results = yolo_model.predict(
            source=np_image,
            conf=YOLO_CONFIDENCE,
            iou=0.50,
            agnostic_nms=False,
            verbose=False
        )

    detections = []

    if not yolo_results:
        return detections

    result = yolo_results[0]

    if result.boxes is None:
        return detections

    image_width, image_height = image.size

    # =========================================================
    # 객체별 처리
    # =========================================================
    for object_index, box in enumerate(result.boxes):

        coordinates = (
            box.xyxy[0]
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        )

        x1, y1, x2, y2 = coordinates

        # 기존 Bounding Box 여백 유지
        x1 = max(0, x1 - BOX_PADDING)
        y1 = max(0, y1 - BOX_PADDING)
        x2 = min(image_width, x2 + BOX_PADDING)
        y2 = min(image_height, y2 + BOX_PADDING)

        if x2 <= x1 or y2 <= y1:
            continue

        # =====================================================
        # YOLO 탐지 결과
        # =====================================================
        yolo_class_id = int(
            box.cls[0].item()
        )

        yolo_score = float(
            box.conf[0].item()
        )

        # 🟢 [통합 4버전 추가]
        # YOLO track()을 사용한 경우 동일 물체에 같은 Tracking ID가 부여됩니다.
        track_id = None
        if getattr(box, "id", None) is not None:
            try:
                track_id = int(box.id[0].item())
            except Exception:
                track_id = None

        yolo_label = yolo_model.names[
            yolo_class_id
        ]

        # 탐지된 객체 Crop
        cropped_image = image.crop(
            (x1, y1, x2, y2)
        )

        # =====================================================
        # 🟢 [추가]
        # YOLO 42개 클래스 → 기존 10개 분류 기준 매핑
        # =====================================================
        mapped_yolo_label = YOLO_TO_10_MAP.get(
            yolo_label
        )

        # 기본 상태
        needs_review = False
        decision_source = ""

        # =====================================================
        # 1. YOLO가 80% 이상이고 매핑 가능
        # → YOLO 결과 우선 사용
        # =====================================================
        if (
            mapped_yolo_label is not None
            and yolo_score
            >= YOLO_CLASSIFY_THRESHOLD
        ):

            top_label = mapped_yolo_label
            top_score = yolo_score

            decision_source = (
                "YOLO 우선 판정"
            )

            needs_review = False

            # 기존 화면 구조 유지
            sorted_results = [
                (
                    top_label,
                    top_score
                )
            ]

        # =====================================================
        # 2. YOLO 80% 미만 또는 매핑 불가
        # → SigLIP2 보조 판정
        # =====================================================
        else:

            classification_results = (
                predict_trash(
                    cropped_image
                )
            )

            sorted_results = sorted(
                classification_results.items(),
                key=lambda x: x[1],
                reverse=True
            )

            siglip_label = (
                sorted_results[0][0]
            )

            siglip_score = (
                sorted_results[0][1]
            )

            # =================================================
            # 2-1. YOLO 클래스가 10개 분류에 매핑 안 됨
            # → SigLIP2 사용
            # =================================================
            if mapped_yolo_label is None:

                top_label = siglip_label
                top_score = siglip_score

                decision_source = (
                    "SigLIP2 보조 판정"
                )

                needs_review = (
                    siglip_score
                    < CONFIDENCE_THRESHOLD
                )

            # =================================================
            # 2-2. YOLO + SigLIP2 같은 재질
            # =================================================
            elif (
                mapped_yolo_label
                == siglip_label
            ):

                top_label = siglip_label
                top_score = siglip_score

                decision_source = (
                    "YOLO + SigLIP2 일치"
                )

                needs_review = (
                    siglip_score
                    < CONFIDENCE_THRESHOLD
                )

            # =================================================
            # 🟠 [수정]
            # 2-3. YOLO + SigLIP2 결과가 서로 다름
            # =================================================
            else:

                # -------------------------------------------------
                # 🟢 [추가]
                # SigLIP2가 80% 이상으로 강하게 판단하면
                # SigLIP2 결과를 최종 판정으로 사용
                # -------------------------------------------------
                if siglip_score >= SIGLIP_STRONG_THRESHOLD:

                    top_label = siglip_label
                    top_score = siglip_score

                    decision_source = (
                        "SigLIP2 강한 보조 판정"
                    )

                    needs_review = False

                # -------------------------------------------------
                # 🟠 [수정]
                # 두 모델이 다르고
                # SigLIP2 신뢰도도 80% 미만이면
                # 수동 확인
                # -------------------------------------------------
                else:

                    top_label = siglip_label
                    top_score = siglip_score

                    decision_source = (
                        "YOLO + SigLIP2 불일치"
                    )

                    needs_review = True

        # =====================================================
        # 객체 결과 저장
        # =====================================================
        detections.append({

            "object_id":
                object_index + 1,

            "track_id":
                track_id,

            "bbox":
                (x1, y1, x2, y2),

            "crop":
                cropped_image,

            # YOLO 결과
            "yolo_label":
                yolo_label,

            "yolo_score":
                yolo_score,

            # 최종 결과
            "label":
                top_label,

            "score":
                top_score,

            "all_results":
                sorted_results,

            # 🟢 [추가]
            "decision_source":
                decision_source,

            # 🟢 [추가]
            "needs_review":
                needs_review
        })

    return detections


# =========================================================
# 🟢 [통합 6버전 추가] 3-3. 동일 물체 중복 Bounding Box 제거
# =========================================================
def calculate_iou(box_a, box_b):
    """두 Bounding Box가 얼마나 겹치는지 IoU(0~1)로 계산합니다."""
    if box_a is None or box_b is None:
        return 0.0

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def deduplicate_detections(detections):
    """
    같은 최종 라벨의 Bounding Box가 많이 겹치면 같은 물체로 보고
    신뢰도가 가장 높은 박스 하나만 남깁니다.

    - 이미지/카메라 촬영
    - 동영상
    - 실시간 웹캠
    모두 동일하게 적용됩니다.
    """
    if len(detections) <= 1:
        return detections

    # 높은 신뢰도부터 검사해야 더 좋은 박스를 남길 수 있습니다.
    sorted_detections = sorted(
        detections,
        key=lambda d: float(d.get("score", 0.0)),
        reverse=True,
    )

    kept = []

    for candidate in sorted_detections:
        candidate_box = candidate.get("bbox")
        candidate_label = candidate.get("label")

        is_duplicate = False

        for existing in kept:
            if candidate_label != existing.get("label"):
                continue

            iou = calculate_iou(
                candidate_box,
                existing.get("bbox"),
            )

            if iou >= DUPLICATE_BOX_IOU_THRESHOLD:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(candidate)

    # 화면의 OBJ 번호가 1,2,3... 순서로 자연스럽게 보이도록 다시 번호 부여
    for index, detection in enumerate(kept, start=1):
        detection["object_id"] = index

    return kept


# =========================================================
# 🟢 [추가] 3-4. Bounding Box 그리기
# =========================================================
def draw_detections(image, detections):

    annotated = np.array(
        image.convert("RGB")
    )

    annotated = cv2.cvtColor(
        annotated,
        cv2.COLOR_RGB2BGR
    )

    for detection in detections:

        x1, y1, x2, y2 = detection["bbox"]

        object_id = detection["object_id"]
        track_id = detection.get("track_id")
        score = detection["score"]

        # Bounding Box
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # OpenCV 기본 폰트는 한글 표시 문제가 있어서
        # 객체 번호 + 신뢰도만 영상에 표시
        label_prefix = (
            f"T{track_id}"
            if track_id is not None
            else f"OBJ {object_id}"
        )
        label_text = (
            f"{label_prefix} "
            f"{score:.0%}"
        )

        cv2.putText(
            annotated,
            label_text,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    annotated = cv2.cvtColor(
        annotated,
        cv2.COLOR_BGR2RGB
    )

    return annotated


# =========================================================
# 🟠 [수정] 3-5. YOLO 탐지 실패 시 기존 SigLIP2 사용
# =========================================================
def fallback_classification(image):

    results = predict_trash(image)

    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        "object_id": 1,
        "track_id": None,
        "bbox": None,
        "crop": image,

        "yolo_label":
            "YOLO 탐지 없음",

        "yolo_score":
            0.0,

        "label":
            sorted_results[0][0],

        "score":
            sorted_results[0][1],

        "all_results":
            sorted_results,

        # 🟢 [추가]
        "decision_source":
            "SigLIP2 fallback",

        # 🟢 [추가]
        "needs_review":
            (
                sorted_results[0][1]
                < CONFIDENCE_THRESHOLD
            )
    }


# =========================================================
# 🟢 [통합 3버전 추가] 자동 재분석 + 재투입 판정
# =========================================================
def reanalyze_uncertain_detection(detection):
    """
    사용자 수동 수정 없이 애매한 객체를 자동으로 한 번 더 분석합니다.
    재분석에도 기준을 통과하지 못하면 포인트를 적립하지 않고
    '재투입 필요' 상태로 보냅니다.
    """

    detection = dict(detection)
    detection["reanalysis_count"] = 0
    detection["reanalysis_status"] = "재분석 불필요"
    detection["requires_reentry"] = False

    if not detection.get("needs_review", False):
        return detection

    crop = detection.get("crop")
    if crop is None:
        detection["reanalysis_status"] = "재분석 실패"
        detection["requires_reentry"] = True
        return detection

    # 같은 이미지를 그대로 반복하는 대신 대비를 자동 보정하여
    # 촬영 각도/조명 문제에 대한 2차 판정 효과를 줍니다.
    retry_image = ImageOps.autocontrast(crop.convert("RGB"))

    retry_results = predict_trash(retry_image)
    sorted_retry = sorted(
        retry_results.items(),
        key=lambda x: x[1],
        reverse=True
    )

    retry_label, retry_score = sorted_retry[0]
    mapped_yolo_label = YOLO_TO_10_MAP.get(
        detection.get("yolo_label")
    )

    # 재분석 성공 조건
    # 1) YOLO와 SigLIP2가 같은 재질로 일치하고 50% 이상
    # 2) SigLIP2가 80% 이상으로 강하게 판단
    # 3) YOLO 매핑이 없는 경우 SigLIP2가 50% 이상
    retry_success = (
        (
            mapped_yolo_label is not None
            and mapped_yolo_label == retry_label
            and retry_score >= CONFIDENCE_THRESHOLD
        )
        or retry_score >= SIGLIP_STRONG_THRESHOLD
        or (
            mapped_yolo_label is None
            and retry_score >= CONFIDENCE_THRESHOLD
        )
    )

    detection["reanalysis_count"] = REANALYSIS_ATTEMPTS
    detection["label"] = retry_label
    detection["score"] = retry_score
    detection["all_results"] = sorted_retry

    if retry_success:
        detection["needs_review"] = False
        detection["requires_reentry"] = False
        detection["reanalysis_status"] = "자동 재분석 성공"
        detection["decision_source"] = (
            f"{detection.get('decision_source', '')} → 자동 재분석 성공"
        ).strip(" →")
    else:
        detection["needs_review"] = True
        detection["requires_reentry"] = True
        detection["reanalysis_status"] = "자동 재분석 실패 → 재투입 필요"
        detection["decision_source"] = (
            f"{detection.get('decision_source', '')} → 자동 재분석 실패"
        ).strip(" →")

    return detection


def reanalyze_detections(detections):
    """탐지 결과 전체에 자동 재분석 정책을 적용합니다."""
    return [
        reanalyze_uncertain_detection(detection)
        for detection in detections
    ]


# =========================================================
# 🟢 [통합 추가] app_jiwon.py 포인트 / 탄소중립 데이터 처리
# =========================================================
@st.cache_data
def load_companies():
    path = os.path.join(DATA_DIR, "carbon_neutral_companies.csv")
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame(columns=["제조사/유통사", "제품수", "대표인증구분"])


companies_df = load_companies()
COMPANY_LIST = (
    sorted(companies_df["제조사/유통사"].dropna().unique().tolist())
    if not companies_df.empty
    else []
)


def _dt_to_str(d):
    d = dict(d)
    if isinstance(d.get("시각"), datetime):
        d["시각"] = d["시각"].isoformat()
    return d


def _str_to_dt(d):
    d = dict(d)
    if isinstance(d.get("시각"), str):
        try:
            d["시각"] = datetime.fromisoformat(d["시각"])
        except ValueError:
            pass
    return d


def save_reward_state():
    try:
        payload = {
            "credited_items": {
                k: _dt_to_str(v)
                for k, v in st.session_state.credited_items.items()
            },
            "points_balance": st.session_state.points_balance,
            "donations": [
                _dt_to_str(d)
                for d in st.session_state.donations
            ],
            "demo_days_ago": st.session_state.demo_days_ago,
        }
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_reward_state():
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            payload = json.load(f)
        payload["credited_items"] = {
            k: _str_to_dt(v)
            for k, v in payload.get("credited_items", {}).items()
        }
        payload["donations"] = [
            _str_to_dt(d)
            for d in payload.get("donations", [])
        ]
        return payload
    except Exception:
        return None


def credit_detection_once(object_key, label, brand):
    """app.py 정상 분류 결과를 기준으로 포인트를 한 번만 적립합니다."""
    if label not in APP_REWARD_INFO:
        return None

    if object_key in st.session_state.credited_items:
        return st.session_state.credited_items[object_key]

    info = APP_REWARD_INFO[label]
    is_partner = brand in COMPANY_LIST and brand != "해당없음"
    multiplier = 2 if is_partner else 1
    points = int(info["rate"] * multiplier)
    carbon = float(info["avg_kg"] * info["carbon_factor"])

    ts = datetime.now() - timedelta(days=st.session_state.demo_days_ago)
    st.session_state.demo_days_ago += 1

    record = {
        "시각": ts,
        "품목": label,
        "포인트": points,
        "탄소절감(kg)": carbon,
        "2배적용": is_partner,
        "브랜드": brand,
        "수량": 1,
    }

    st.session_state.credited_items[object_key] = record
    st.session_state.points_balance += points
    save_reward_state()
    return record


def eco_card(col, icon, label, value):
    with col:
        st.markdown(f"""
        <div class="eco-card">
            <div class="icon">{icon}</div>
            <div>
                <div class="label">{label}</div>
                <div class="value">{value}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def show_credit_success(record):
    """B팀원 원본의 포인트 적립 성공 안내 UI. A팀원 AI 판정 결과를 받아 표시합니다."""
    if not record:
        return
    partner_text = ""
    if record.get("2배적용"):
        partner_text = (
            f"<br>🌱 <b>{record.get('브랜드', '')}</b>는 참여기업입니다 — "
            "포인트가 <b>2배</b> 적립되었습니다!"
        )
    st.markdown(
        f"""<div class="success-box">
        ✅ <b>{record.get('품목', '')}</b> {record.get('수량', 1)}개 적립 완료.{partner_text}<br>
        이번 적립: <b>+{int(record.get('포인트', 0))}P</b> / 
        탄소 절감 <b>+{float(record.get('탄소절감(kg)', 0)):.3f}kg CO2</b>
        </div>""",
        unsafe_allow_html=True,
    )


def render_ai_briefing():
    """B팀원 원본 AI 배출 브리핑 기능 복원."""
    hist_df = pd.DataFrame(list(st.session_state.credited_items.values()))
    if hist_df.empty:
        st.info("분류 이력이 쌓이면 이곳에 AI가 요약 브리핑을 만들어줍니다.")
        return

    top_item = hist_df["품목"].value_counts().idxmax()
    total_points = int(hist_df["포인트"].sum())
    total_carbon = float(hist_df["탄소절감(kg)"].sum())
    doubled = int(hist_df["2배적용"].sum())

    api_key = os.environ.get("OPENAI_API_KEY", "")
    used_openai = False
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            prompt = (
                f"사용자가 가장 많이 배출한 품목은 '{top_item}'이고, 총 {total_points}포인트를 적립했으며 "
                f"약 {total_carbon:.2f}kg의 CO2를 절감했다. 이 중 {doubled}건은 참여기업 제품이라 2배 적립되었다. "
                "이 내용을 친근한 톤의 3문장짜리 한국어 브리핑으로 요약해줘."
            )
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
            )
            st.markdown(
                f'<div class="briefing-box">{resp.choices[0].message.content}</div>',
                unsafe_allow_html=True,
            )
            used_openai = True
        except Exception as e:
            st.error(f"OpenAI 호출 오류로 규칙 기반 요약으로 대체합니다: {e}")

    if not used_openai:
        st.markdown(f"""<div class="briefing-box">
        가장 많이 배출한 품목은 <b>{top_item}</b>이에요. 지금까지 <b>{total_points}P</b>를 적립했고
        약 <b>{total_carbon:.2f}kg의 CO2</b>를 절감했어요. 이 중 <b>{doubled}건</b>은 참여기업 제품이라
        2배 적립됐어요. 참여기업 제품을 더 활용하면 포인트를 훨씬 빠르게 모을 수 있어요!
        </div>""", unsafe_allow_html=True)
        st.caption("※ OPENAI_API_KEY 환경변수를 설정하면 실제 생성형 AI 브리핑으로 전환됩니다.")


def render_reward_dashboard(key_prefix="reward"):
    """B팀원 원본의 개인 배출 통계를 A팀원 분류 결과 기반 적립 기록으로 표시합니다."""
    hist_df = pd.DataFrame(list(st.session_state.credited_items.values()))
    if hist_df.empty:
        st.info("아직 적립 이력이 없습니다.")
        return

    st.markdown("### 📊 나의 배출 리포트")

    hist_df["시각"] = pd.to_datetime(hist_df["시각"])
    total_points = st.session_state.points_balance
    total_count = int(hist_df["수량"].sum()) if "수량" in hist_df else len(hist_df)
    total_carbon = float(hist_df["탄소절감(kg)"].sum())

    if "my_rank" not in st.session_state:
        st.session_state.my_rank = random.randint(1, MOCK_OTHER_USERS)
    rank = st.session_state.my_rank
    total_participants = MOCK_OTHER_USERS + 1

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("나의 포인트", f"{total_points:,} P")
    c2.metric("이번 달 배출", f"{total_count} 개")
    c3.metric("절감한 탄소", f"{total_carbon:.2f} kg CO2")
    c4.metric("전체 중 나의 순위", f"{rank}위 / {total_participants}명")

    car_km = total_carbon * CO2_TO_CAR_KM
    tree_year = total_carbon / CO2_PER_TREE_YEAR
    led_hour = total_carbon / CO2_PER_LED_HOUR if CO2_PER_LED_HOUR else 0

    e1, e2, e3 = st.columns(3)
    eco_card(e1, "🚗", "자동차 대신", f"{car_km:.1f}km 안 탄 효과")
    eco_card(e2, "🌳", "나무가 흡수하는 양", f"{tree_year:.2f}그루 · 1년분")
    eco_card(e3, "💡", "전구 사용량", f"LED {led_hour:.0f}시간분")

    st.markdown("")
    st.subheader("배출 추이")
    unit = st.radio(
        "보기 단위", ["일별", "월별"], horizontal=True,
        key=f"{key_prefix}_trend_unit",
    )
    trend = hist_df.copy()
    trend["기간"] = trend["시각"].dt.strftime("%Y-%m-%d" if unit == "일별" else "%Y-%m")
    grouped = trend.groupby("기간")[["포인트", "탄소절감(kg)"]].sum().reset_index()
    zero_scale = alt.Scale(domainMin=0, nice=False)
    bar = alt.Chart(grouped).mark_bar().encode(
        x=alt.X("기간:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("포인트:Q", scale=zero_scale),
        color=alt.Color("기간:N", scale=alt.Scale(scheme="tableau20"), legend=None),
    )
    line = alt.Chart(grouped).mark_line(point=True).encode(
        x=alt.X("기간:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("탄소절감(kg):Q", scale=zero_scale),
    )
    st.altair_chart(alt.layer(bar, line).resolve_scale(y="independent"), width="stretch")
    st.caption("막대 = 날짜별 포인트(색상 구분) · 선 = 탄소 절감 추이")

    st.subheader("품목별 이번 달 배출 비중")
    if "수량" in hist_df:
        item_count = hist_df.groupby("품목")["수량"].sum().rename("개수").reset_index()
    else:
        item_count = hist_df["품목"].value_counts().rename_axis("품목").reset_index(name="개수")
    chart = alt.Chart(item_count).mark_bar().encode(
        x=alt.X("개수:Q", scale=zero_scale, axis=alt.Axis(format="d", tickMinStep=1)),
        y=alt.Y("품목:N", sort="-x"),
        color=alt.Color("품목:N", legend=None),
    )
    st.altair_chart(chart, width="stretch")

    doubled_count = int(hist_df["2배적용"].sum()) if "2배적용" in hist_df else 0
    if doubled_count:
        st.caption(f"🌱 이 중 {doubled_count}건은 탄소중립실천포인트 참여기업 제품으로 2배 적립되었습니다.")
    st.caption("※ 탄소 절감량·비유 수치는 품목별 평균 무게 추정치 x LCA 기반 계수를 곱한 참고값입니다.")


# =========================================================
# 4. 분석 기록 저장
# =========================================================
if "history" not in st.session_state:
    st.session_state.history = []


# 🟢 [추가] 중복 분석 기록 방지
if "processed_keys" not in st.session_state:
    st.session_state.processed_keys = set()


# 🟢 [추가] 실시간 웹캠 기록 시간
if "last_webcam_log" not in st.session_state:
    st.session_state.last_webcam_log = 0.0

if "webcam_session_token" not in st.session_state:
    st.session_state.webcam_session_token = None

if "webcam_processed_track_keys" not in st.session_state:
    st.session_state.webcam_processed_track_keys = set()

# 🟢 [통합 5버전 추가]
# 실시간 웹캠에서 마지막으로 실제 적립된 결과를 화면에 표시
if "last_webcam_reward" not in st.session_state:
    st.session_state.last_webcam_reward = None


# 🟢 [통합 추가] 포인트 / 기부 상태 초기화
if "credited_items" not in st.session_state:
    _saved_reward = load_reward_state()
    if _saved_reward:
        st.session_state.credited_items = _saved_reward["credited_items"]
        st.session_state.points_balance = _saved_reward["points_balance"]
        st.session_state.donations = _saved_reward["donations"]
        st.session_state.demo_days_ago = _saved_reward.get(
            "demo_days_ago",
            len(_saved_reward["credited_items"]),
        )
    else:
        st.session_state.credited_items = {}
        st.session_state.points_balance = 0
        st.session_state.donations = []
        st.session_state.demo_days_ago = 0


# =========================================================
# 🟢 [B팀원 기능 복원] 사이드바
#    A팀원의 10종 AI 분류는 그대로 유지합니다.
#    아래 4종은 B팀원의 리워드 정책 표시입니다.
# =========================================================
with st.sidebar:
    st.header("♻️ AI 재활용품 플랫폼")
    st.caption("이물질 투입으로 인한 무인 회수기 오작동을 선제 차단하고, 사용자 리워드를 자동화합니다.")
    st.divider()
    st.subheader("분류 가능 클래스")
    for info in B_CLASS_INFO.values():
        st.markdown(f"- {info['label']} ({info['rate']}원/개)")
    st.divider()
    st.markdown(
        "**B팀원 리워드 정책**: 위 4개 클래스가 포인트 적립 대상입니다. "
        "A팀원 AI의 10종 탐지·분류 기능은 그대로 유지됩니다."
    )

# =========================================================
# 5. 입력 방식 선택
# =========================================================
st.subheader("📷 미디어 입력 선택")

# 🟢 [통합 추가] 참여기업 제품이면 포인트 2배
selected_brand = st.selectbox(
    "제조사/브랜드 (참여기업이면 포인트 2배)",
    ["해당없음"] + COMPANY_LIST,
)

input_type = st.radio(
    "분석할 미디어 형식을 선택하세요.",
    [
        "이미지 업로드",
        "카메라 촬영",
        "동영상 업로드",
        "🎥 실시간 웹캠 감지"
    ],
    horizontal=True
)

images = []
video_source_key = None


# ---------------------------------------------------------
# 1) 이미지 업로드
# ---------------------------------------------------------
if input_type == "이미지 업로드":

    uploaded_files = st.file_uploader(
        "이미지를 업로드하세요. (여러 개 선택 가능)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:

        for file in uploaded_files:

            img = Image.open(file)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")

            images.append(
                (file.name, img)
            )


# ---------------------------------------------------------
# 2) 카메라 촬영
# ---------------------------------------------------------
elif input_type == "카메라 촬영":

    camera_file = st.camera_input(
        "카메라로 재활용품을 촬영하세요."
    )

    if camera_file is not None:

        img = Image.open(camera_file)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")

        images.append(
            ("카메라 촬영 이미지", img)
        )


# ---------------------------------------------------------
# 3) 동영상 업로드
# ---------------------------------------------------------
elif input_type == "동영상 업로드":

    video_file = st.file_uploader(
        "동영상 파일을 업로드하세요 (MP4, AVI, MOV)",
        type=["mp4", "avi", "mov"]
    )

    if video_file is not None:

        video_source_key = hashlib.md5(video_file.getvalue()).hexdigest()[:16]

        st.video(video_file)

        sample_interval_sec = st.slider(
            "프레임 추출 간격 (초 단위)",
            min_value=0.5,
            max_value=5.0,
            value=1.0,
            step=0.5
        )

        if st.button(
            "🎬 동영상 프레임 분석 시작"
        ):

            # 🟠 [수정]
            # 원본 확장자를 임시파일에 사용
            suffix = os.path.splitext(
                video_file.name
            )[1]

            if not suffix:
                suffix = ".mp4"

            tfile = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            )

            # 🟠 [수정]
            # video_file.read() 대신 getvalue()
            tfile.write(video_file.getvalue())
            tfile.close()

            cap = cv2.VideoCapture(
                tfile.name
            )

            fps = cap.get(
                cv2.CAP_PROP_FPS
            )

            if fps == 0 or not fps:
                fps = 30.0

            # 🟠 [수정]
            # 최소 1프레임 보장
            frame_interval = max(
                1,
                int(
                    fps * sample_interval_sec
                )
            )

            frame_count = 0
            extracted_count = 0

            with st.spinner(
                "동영상에서 주요 프레임을 추출하는 중..."
            ):

                while cap.isOpened():

                    ret, frame = cap.read()

                    if not ret:
                        break

                    if frame_count % frame_interval == 0:

                        # 🟢 [추가]
                        # 최대 100프레임까지만 분석
                        if extracted_count >= MAX_VIDEO_FRAMES:
                            break

                        frame_rgb = cv2.cvtColor(
                            frame,
                            cv2.COLOR_BGR2RGB
                        )

                        pil_img = Image.fromarray(
                            frame_rgb
                        )

                        timestamp_sec = (
                            frame_count / fps
                        )

                        extracted_count += 1

                        images.append(
                            (
                                f"동영상 "
                                f"{timestamp_sec:.1f}초 시점 "
                                f"(프레임 #{extracted_count})",
                                pil_img
                            )
                        )

                    frame_count += 1

                cap.release()

                try:
                    os.remove(tfile.name)

                except OSError:
                    pass

            st.success(
                f"총 {extracted_count}개의 "
                f"프레임을 추출하여 분석합니다."
            )


# ---------------------------------------------------------
# 4) 실시간 웹캠
# ---------------------------------------------------------
elif input_type == "🎥 실시간 웹캠 감지":

    # 🟠 [수정]
    st.info(
        "웹캠을 통해 실시간으로 여러 재활용품을 "
        "탐지하고 분류합니다."
    )

    col_cam, col_info = st.columns(
        [2, 1]
    )

    with col_cam:

        run_cam = st.checkbox(
            "🎥 웹캠 실시간 가동 시작"
        )

        frame_placeholder = st.empty()

    with col_info:

        st.subheader(
            "실시간 판정 정보"
        )

        status_placeholder = st.empty()
        guide_placeholder = st.empty()

        # 🟢 [통합 7버전 추가]
        # 실시간 분석으로 화면이 계속 갱신되어도
        # 마지막 포인트 적립 결과와 현재 잔액은 이 영역에 고정 표시합니다.
        reward_placeholder = st.empty()

    if run_cam:

        if st.session_state.webcam_session_token is None:
            st.session_state.webcam_session_token = (
                f"webcam_{int(time.time() * 1000)}"
            )
            st.session_state.webcam_processed_track_keys = set()
            st.session_state.last_webcam_reward = None

        # 🟢 [통합 7버전 추가]
        with reward_placeholder.container():
            st.info(
                "💰 포인트 대상 재활용품을 인식하면 "
                "이곳에 적립 결과가 표시됩니다."
            )
            st.metric(
                "💳 현재 보유 포인트",
                f"{st.session_state.points_balance:,} P",
            )

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():

            st.error(
                "웹캠을 열 수 없습니다. "
                "카메라 연결 상태를 확인해 주세요."
            )

        else:

            frame_skip = 0

            # 🟢 [추가]
            latest_detections = []

            while run_cam:

                ret, frame = cap.read()

                if not ret:

                    st.error(
                        "웹캠 스트림을 가져오는 데 실패했습니다."
                    )

                    break

                # =================================================
                # 🔴 [삭제]
                #
                # 기존:
                # if frame_skip % 3 == 0:
                #
                # 🟠 [수정]
                # YOLO + SigLIP2를 사용하므로
                # 15프레임마다 AI 분석
                # =================================================
                if (
                    frame_skip
                    % WEBCAM_FRAME_INTERVAL
                    == 0
                ):

                    frame_rgb = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    pil_img = Image.fromarray(
                        frame_rgb
                    )

                    # 🟢 [추가]
                    # YOLO-World → 여러 물체 탐지
                    # → SigLIP2 개별 분류
                    detections = detect_and_classify(
                        pil_img,
                        use_tracking=True
                    )

                    # 🟢 [통합 3버전 추가]
                    # 애매한 객체는 사용자 수정 대신 자동 재분석 1회
                    detections = reanalyze_detections(detections)

                    # 🟢 [통합 6버전 추가]
                    # 같은 물체가 겹치는 박스 2개 이상으로 잡히면 1개만 유지
                    detections = deduplicate_detections(detections)

                    latest_detections = detections

                    if detections:

                        with status_placeholder.container():

                            st.success(
                                f"✅ {len(detections)}개 "
                                f"물체 탐지"
                            )

                            for detection in detections:

                                object_id = (
                                    detection["object_id"]
                                )

                                top_label = (
                                    detection["label"]
                                )

                                top_score = (
                                    detection["score"]
                                )

                                # 🟠 [수정]
                                if not detection["needs_review"]:

                                    st.write(
                                        f"**OBJ {object_id}** "
                                        f"→ {top_label} "
                                        f"({top_score:.1%})"
                                    )

                                else:

                                    st.warning(
                                        f"OBJ {object_id} "
                                        f"→ {top_label} "
                                        f"({top_score:.1%}) "
                                        f"→ 자동 재분석 실패 · 재투입 필요"
                                    )

                        with guide_placeholder.container():

                            st.markdown(
                                "#### ♻️ 분리배출 안내"
                            )

                            for detection in detections:

                                top_label = (
                                    detection["label"]
                                )

                                top_score = (
                                    detection["score"]
                                )

                                # 🟠 [수정]
                                if not detection["needs_review"]:

                                    guide_text = RECYCLE_GUIDE.get(
                                        top_label,
                                        "정보 없음"
                                    )

                                    st.info(
                                        f"**{top_label}**\n\n"
                                        f"{guide_text}"
                                    )

                                else:

                                    st.warning(
                                        "⚠️ 신뢰도 부족 → "
                                        "재투입 라인"
                                    )

                        # 🟢 [통합 4버전 수정]
                        # 실시간 웹캠은 Tracking ID 기준으로 같은 객체를 한 번만 기록/적립
                        current_time = time.time()

                        if (
                            current_time
                            - st.session_state.last_webcam_log
                            >= WEBCAM_LOG_INTERVAL
                        ):

                            for detection in detections:
                                top_label = detection["label"]
                                top_score = detection["score"]
                                track_id = detection.get("track_id")

                                if track_id is not None:
                                    webcam_track_key = (
                                        f"{st.session_state.webcam_session_token}_"
                                        f"track_{track_id}"
                                    )
                                else:
                                    webcam_track_key = (
                                        f"{st.session_state.webcam_session_token}_"
                                        f"fallback_{int(current_time)}_"
                                        f"{detection['object_id']}_{top_label}"
                                    )

                                if (
                                    webcam_track_key
                                    in st.session_state.webcam_processed_track_keys
                                ):
                                    continue

                                st.session_state.history.append(
                                    {
                                        "파일명": "실시간 웹캠",
                                        "객체 번호": detection["object_id"],
                                        "분류 결과": top_label,
                                        "신뢰도": top_score,
                                        "입력 방식": "실시간 웹캠",
                                        "판정 상태": (
                                            "정상 분류"
                                            if not detection["needs_review"]
                                            else "재투입 필요"
                                        ),
                                    }
                                )

                                st.session_state.webcam_processed_track_keys.add(
                                    webcam_track_key
                                )

                                if not detection["needs_review"]:
                                    reward_record = credit_detection_once(
                                        webcam_track_key,
                                        top_label,
                                        selected_brand,
                                    )

                                    # 포인트 대상 품목일 때만 적립 결과 표시
                                    if reward_record is not None:
                                        st.session_state.last_webcam_reward = reward_record

                            st.session_state.last_webcam_log = current_time

                        # 🟢 [통합 7버전 수정]
                        # 실시간 프레임이 계속 바뀌어도 포인트 안내가 사라지지 않도록
                        # 전용 placeholder 안에서 마지막 적립 결과 + 현재 잔액을 갱신합니다.
                        with reward_placeholder.container():
                            if st.session_state.last_webcam_reward is not None:
                                reward = st.session_state.last_webcam_reward
                                partner_text = (
                                    " · 참여기업 2배 적용"
                                    if reward.get("2배적용")
                                    else ""
                                )
                                st.success(
                                    f"💰 이번 적립: "
                                    f"{reward['품목']} +{reward['포인트']}P"
                                    f"{partner_text}"
                                )
                                st.metric(
                                    "💳 현재 보유 포인트",
                                    f"{st.session_state.points_balance:,} P",
                                )
                            else:
                                st.info(
                                    "💰 포인트 대상 재활용품을 인식하면 "
                                    "이곳에 적립 결과가 표시됩니다."
                                )

                    else:

                        with status_placeholder.container():

                            st.warning(
                                "탐지된 재활용품이 없습니다."
                            )

                # 🟢 [추가]
                # 최근 탐지 결과의 Bounding Box 표시
                if latest_detections:

                    display_rgb = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    display_pil = Image.fromarray(
                        display_rgb
                    )

                    annotated = draw_detections(
                        display_pil,
                        latest_detections
                    )

                    # 🟠 [수정]
                    # use_container_width=True
                    # → width="stretch"
                    frame_placeholder.image(
                        annotated,
                        channels="RGB",
                        width="stretch"
                    )

                else:

                    # 🟠 [수정]
                    frame_placeholder.image(
                        frame,
                        channels="BGR",
                        width="stretch"
                    )

                frame_skip += 1

            cap.release()


# =========================================================
# 6 & 7. 이미지 / 카메라 / 동영상 분석 결과
# =========================================================
if images:

    # 🟠 [수정]
    st.subheader(
        f"🔍 AI 객체 탐지 및 분류 결과 "
        f"(총 {len(images)}개 이미지/프레임)"
    )

    for file_name, img in images:

        # 🟢 [추가]
        # 같은 이미지가 Streamlit rerun 때
        # history에 계속 중복 저장되는 것을 방지
        image_key = (
            f"{input_type}_"
            f"{file_name}_"
            f"{hash(img.tobytes())}"
        )

        with st.spinner(
            f"'{file_name}' "
            f"YOLO 객체 탐지 + SigLIP2 분석 중..."
        ):

            # =====================================================
            # 🔴 [삭제]
            #
            # 기존:
            # results = predict_trash(img)
            #
            # 🟢 [추가]
            # YOLO-World가 여러 객체를 먼저 탐지
            # =====================================================
            detections = detect_and_classify(
                img,
                use_tracking=(input_type == "동영상 업로드")
            )

        # 🟢 [추가]
        # YOLO가 아무 물체도 찾지 못하면
        # 기존 SigLIP2 전체 이미지 분류 사용
        if not detections:

            st.warning(
                "⚠️ YOLO-World가 개별 물체를 "
                "찾지 못했습니다. "
                "기존 SigLIP2로 전체 이미지를 분석합니다."
            )

            detections = [
                fallback_classification(img)
            ]

        # 🟢 [통합 3버전 추가]
        # 애매한 객체는 사용자 수정 없이 자동 재분석 1회
        detections = reanalyze_detections(detections)

        # 🟢 [통합 6버전 추가]
        # 이미지/카메라/동영상에서도 같은 물체의 겹치는 박스는 하나만 유지
        detections = deduplicate_detections(detections)

        # Bounding Box가 있는 객체만 추림
        bbox_detections = [
            detection
            for detection in detections
            if detection["bbox"] is not None
        ]

        if bbox_detections:

            annotated = draw_detections(
                img,
                bbox_detections
            )

        else:

            annotated = np.array(img)

        # =====================================================
        # 전체 이미지 + 요약
        # =====================================================
        col_img, col_summary = st.columns(
            [1.3, 1]
        )

        with col_img:

            # 🟠 [수정]
            # use_container_width=True
            # → width="stretch"
            st.image(
                annotated,
                caption=file_name,
                width="stretch"
            )

        with col_summary:

            st.metric(
                "탐지된 객체",
                f"{len(detections)}개"
            )

            # 🟠 [수정]
            normal_count = sum(
                1
                for detection in detections
                if not detection["needs_review"]
            )

            reject_count = (
                len(detections)
                - normal_count
            )

            col_a, col_b = st.columns(2)

            with col_a:

                st.metric(
                    "정상 분류",
                    f"{normal_count}개"
                )

            with col_b:

                st.metric(
                    "재투입 필요",
                    f"{reject_count}개"
                )

        # =====================================================
        # 🟢 [추가] 객체별 결과
        # =====================================================
        st.markdown(
            "### 📦 객체별 분류 결과"
        )

        for detection in detections:

            object_id = detection["object_id"]
            top_label = detection["label"]
            top_score = detection["score"]

            yolo_label = detection["yolo_label"]
            yolo_score = detection["yolo_score"]

            col_crop, col_res = st.columns(
                [1, 2]
            )

            with col_crop:

                st.image(
                    detection["crop"],
                    caption=f"OBJ {object_id}",
                    width="stretch"
                )

            with col_res:

                st.markdown(
                    f"#### OBJ {object_id} → {top_label}"
                )

                # YOLO 탐지 정보
                if detection["bbox"] is not None:

                    st.caption(
                        f"YOLO-World 탐지: "
                        f"{yolo_label} "
                        f"({yolo_score:.1%})"
                    )

                else:

                    st.caption(
                        "YOLO 탐지 없음 → "
                        "SigLIP2 전체 이미지 분석"
                    )

                # 🟢 [통합 3버전 추가] 자동 재분석 결과 표시
                if detection.get("reanalysis_count", 0) > 0:
                    if detection.get("requires_reentry", False):
                        st.caption("🔄 자동 재분석 1회 수행 → 재투입 필요")
                    else:
                        st.caption("🔄 자동 재분석 1회 수행 → 판정 성공")

                # 🟠 [수정]
                # 기존 0.5 직접 사용
                # → CONFIDENCE_THRESHOLD
                # 🟠 [수정]
                if detection["needs_review"]:

                    st.error(
                        "⚠️ **[경고] 자동 재분석 후에도 판정이 불확실합니다.**"
                    )

                    st.warning(
                        "물품 판정이 불확실하여 "
                        "자동 재분석 후에도 판정이 어려워 "
                        "**'재투입 라인'**으로 이동합니다."
                    )

                else:

                    guide_text = RECYCLE_GUIDE.get(
                        top_label,
                        "지침 정보 없음"
                    )

                    st.success(
                        f"💡 **분리배출 지침:** "
                        f"{guide_text}"
                    )

                # 기존 TOP 3 결과 유지
                for label, score in (
                    detection["all_results"][:3]
                ):

                    st.write(
                        f"**{label}**: "
                        f"{score:.2%}"
                    )

                    st.progress(
                        float(score)
                    )

            # 🟠 [수정]
            # 기존에는 이미지 1장당 기록
            # → YOLO 객체 1개당 기록
            track_id = detection.get("track_id")
            if (
                input_type == "동영상 업로드"
                and track_id is not None
                and video_source_key is not None
            ):
                object_key = (
                    f"video_{video_source_key}_"
                    f"track_{track_id}"
                )
            else:
                object_key = (
                    f"{image_key}_"
                    f"{object_id}"
                )

            if (
                object_key
                not in st.session_state.processed_keys
            ):

                st.session_state.history.append(
                    {
                        "파일명":
                            file_name,

                        "객체 번호":
                            object_id,

                        "분류 결과":
                            top_label,

                        "신뢰도":
                            top_score,

                        "입력 방식":
                            input_type,

                        # 🟠 [수정]
                        "판정 상태":
                        (
                            "정상 분류"
                            if not detection["needs_review"]
                            else
                            "재투입 필요"
                        )
                    }
                )

                st.session_state.processed_keys.add(
                    object_key
                )

                # 🟢 [통합 추가]
                # app.py의 정상 분류 결과 중 포인트 정책 대상만 자동 적립
                if not detection["needs_review"]:
                    reward_record = credit_detection_once(
                        object_key,
                        top_label,
                        selected_brand,
                    )
                    if reward_record is not None:
                        bonus_text = (
                            " · 참여기업 2배"
                            if reward_record["2배적용"]
                            else ""
                        )
                        st.toast(
                            f"{top_label} +{reward_record['포인트']}P 적립"
                            f"{bonus_text}",
                            icon="✅",
                        )
                        show_credit_success(reward_record)

            st.divider()


elif input_type != "🎥 실시간 웹캠 감지":

    st.info(
        "💡 분석할 이미지를 업로드하거나 "
        "카메라 촬영 또는 동영상을 선택해 주세요."
    )


# =========================================================
# 8. 대시보드
# =========================================================
st.divider()

st.header(
    "📊 재활용품 분석 대시보드"
)


if len(st.session_state.history) > 0:

    df = pd.DataFrame(
        st.session_state.history
    )

    total_count = len(df)

    most_common = (
        df["분류 결과"]
        .value_counts()
        .idxmax()
    )

    average_score = (
        df["신뢰도"].mean()
    )

    # 🟢 [추가]
    normal_total = (
        df["판정 상태"]
        == "정상 분류"
    ).sum()

    reject_total = (
        df["판정 상태"].isin([
        "리젝트(수동확인)",
        "재투입 필요"
    ])
    ).sum()


    # 🟠 [수정]
    # 기존 3개 → 5개 지표
    col1, col2, col3, col4, col5 = (
        st.columns(5)
    )

    with col1:

        st.metric(
            "총 분석 객체",
            f"{total_count}개"
        )

    with col2:

        st.metric(
            "가장 많이 분류된 항목",
            most_common
        )

    with col3:

        st.metric(
            "평균 신뢰도",
            f"{average_score:.1%}"
        )

    with col4:

        st.metric(
            "정상 분류",
            f"{normal_total}개"
        )

    with col5:

        st.metric(
            "재투입 필요",
            f"{reject_total}개"
        )


    # ---------------------------------------------------------
    # 재활용품 분류 현황
    # ---------------------------------------------------------
    st.subheader(
        "📈 재활용품 분류 현황"
    )

    count_df = (
        df["분류 결과"]
        .value_counts()
        .reset_index()
    )

    count_df.columns = [
        "재활용품 종류",
        "분류 횟수"
    ]

    chart = alt.Chart(
        count_df
    ).mark_bar().encode(

        x=alt.X(
            "재활용품 종류:N",
            title="재활용품 종류",
            axis=alt.Axis(
                labelAngle=0
            )
        ),

        y=alt.Y(
            "분류 횟수:Q",
            title="분류 횟수"
        )
    )

    # 🟠 [수정]
    # use_container_width=True
    # → width="stretch"
    st.altair_chart(
        chart,
        width="stretch"
    )


    # ---------------------------------------------------------
    # 미디어 입력 방식
    # ---------------------------------------------------------
    st.subheader(
        "📷 미디어 입력 방식"
    )

    input_count = (
        df["입력 방식"]
        .value_counts()
        .reset_index()
    )

    input_count.columns = [
        "입력 방식",
        "횟수"
    ]

    input_chart = alt.Chart(
        input_count
    ).mark_bar().encode(

        x=alt.X(
            "입력 방식:N",
            title="입력 방식",
            axis=alt.Axis(
                labelAngle=0
            )
        ),

        y=alt.Y(
            "횟수:Q",
            title="횟수"
        )
    )

    # 🟠 [수정]
    st.altair_chart(
        input_chart,
        width="stretch"
    )


    # ---------------------------------------------------------
    # 🟢 [추가] AI 판정 상태
    # ---------------------------------------------------------
    st.subheader(
        "🚦 AI 판정 상태"
    )

    status_count = (
        df["판정 상태"]
        .value_counts()
        .reset_index()
    )

    status_count.columns = [
        "판정 상태",
        "횟수"
    ]

    status_chart = alt.Chart(
        status_count
    ).mark_bar().encode(

        x=alt.X(
            "판정 상태:N",
            title="판정 상태",
            axis=alt.Axis(
                labelAngle=0
            )
        ),

        y=alt.Y(
            "횟수:Q",
            title="횟수"
        )
    )

    st.altair_chart(
        status_chart,
        width="stretch"
    )


    # ---------------------------------------------------------
    # 최근 분석 기록
    # ---------------------------------------------------------
    st.subheader(
        "📋 최근 분석 기록"
    )

    history_df = df.copy()

    history_df["신뢰도"] = (
        history_df["신뢰도"]
        * 100
    ).round(2).astype(str) + "%"

    # 🟠 [수정]
    st.dataframe(
        history_df,
        width="stretch",
        hide_index=True
    )


    # ---------------------------------------------------------
    # CSV 다운로드
    # ---------------------------------------------------------
    csv_data = (
        history_df
        .to_csv(index=False)
        .encode("utf-8-sig")
    )

    st.download_button(
        label="📥 전체 분석 이력 CSV 다운로드",
        data=csv_data,
        file_name="recycle_classification_log.csv",
        mime="text/csv"
    )


    # ---------------------------------------------------------
    # 분석 기록 초기화
    # ---------------------------------------------------------
    st.divider()

    if st.button(
        "분석 기록 초기화"
    ):

        st.session_state.history = []

        # 🟢 [추가]
        st.session_state.processed_keys = set()

        # 🟢 [추가]
        st.session_state.last_webcam_log = 0.0
        st.session_state.webcam_session_token = None
        st.session_state.webcam_processed_track_keys = set()
        st.session_state.last_webcam_reward = None

        st.rerun()


else:

    st.info(
        "아직 분석 기록이 없습니다. "
        "미디어를 분석하면 "
        "대시보드에 결과가 표시됩니다."
    )

# =========================================================
# 9. 🟢 [통합 추가] 포인트 / 탄소중립 / 기부 확장 기능
#    기존 app.py 대시보드는 그대로 두고 아래에 추가합니다.
# =========================================================
st.divider()
st.header("🌱 리워드 · 탄소중립 확장 기능")

reward_tabs = st.tabs([
    "💰 포인트 적립 현황",
    "🏢 탄소중립실천포인트 참여기업",
    "🌱 제품별 탄소발자국",
    "🎁 포인트로 기부하기",
    "⚖️ 도입 전후 비교",
])

with reward_tabs[0]:
    st.caption(
        "AI 판정은 A팀원 app.py의 YOLO + SigLIP2 결과를 그대로 사용합니다. "
        "B팀원 리워드·탄소중립 기능만 결합했습니다."
    )
    render_reward_dashboard(key_prefix="reward_tab")

    if st.session_state.credited_items:
        hist_df_reward = pd.DataFrame(list(st.session_state.credited_items.values()))
        hist_df_reward["시각"] = pd.to_datetime(hist_df_reward["시각"])
        zero_scale_reward = alt.Scale(domainMin=0, nice=False)

        st.divider()
        st.markdown("### 💰 포인트 적립 현황")
        p1, p2, p3 = st.columns(3)
        p1.metric("현재 보유 포인트", f"{st.session_state.points_balance:,} P")
        p2.metric("누적 배출 개수", f"{int(hist_df_reward['수량'].sum())} 개")
        p3.metric("누적 탄소 절감", f"{float(hist_df_reward['탄소절감(kg)'].sum()):.2f} kg CO2")

        st.markdown("#### 📅 날짜별 포인트 적립 현황")
        daily = hist_df_reward.copy()
        daily["날짜"] = daily["시각"].dt.strftime("%Y-%m-%d")
        daily_grouped = (
            daily.groupby("날짜")
            .agg(포인트=("포인트", "sum"), **{"배출 개수": ("수량", "sum")})
            .reset_index()
            .sort_values("날짜")
        )
        st.altair_chart(
            alt.Chart(daily_grouped).mark_bar().encode(
                x=alt.X("날짜:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("포인트:Q", scale=zero_scale_reward),
                color=alt.Color("날짜:N", scale=alt.Scale(scheme="tableau20"), legend=None),
            ),
            width="stretch",
        )
        st.dataframe(
            daily_grouped.sort_values("날짜", ascending=False),
            width="stretch",
            hide_index=True,
        )

        st.divider()
        st.subheader("🤖 AI 배출 브리핑")
        render_ai_briefing()

with reward_tabs[1]:
    st.caption(
        "data_ref/carbon_neutral_companies.csv 기반 참여기업 목록입니다. "
        "분류 전에 참여기업을 선택하면 적립 포인트가 2배 적용됩니다."
    )
    if companies_df.empty:
        st.warning("data_ref/carbon_neutral_companies.csv 파일이 없습니다.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("참여기업 수", f"{len(companies_df)} 곳")
        c2.metric(
            "등록 제품 수(합계)",
            f"{int(companies_df['제품수'].sum()):,} 개",
        )

        company_query = st.text_input(
            "기업명 검색",
            placeholder="예: 이마트",
            key="company_search",
        )
        filtered_companies = companies_df.copy()
        if company_query:
            filtered_companies = filtered_companies[
                filtered_companies["제조사/유통사"]
                .astype(str)
                .str.contains(company_query, case=False, na=False)
            ]
        st.dataframe(
            filtered_companies,
            width="stretch",
            height=300,
            hide_index=True,
        )

        st.subheader("제품 등록 수 상위 15개 기업")
        top15 = (
            companies_df.sort_values("제품수", ascending=False)
            .head(15)
        )
        company_chart = alt.Chart(top15).mark_bar().encode(
            x=alt.X(
                "제품수:Q",
                scale=alt.Scale(domainMin=0, nice=False),
            ),
            y=alt.Y("제조사/유통사:N", sort="-x"),
        )
        st.altair_chart(company_chart, width="stretch")

with reward_tabs[2]:
    carbon_path = os.path.join(
        DATA_DIR,
        "carbon_footprint_products.csv",
    )
    if not os.path.exists(carbon_path):
        st.warning("data_ref/carbon_footprint_products.csv 파일이 없습니다.")
    else:
        st.caption("환경부 · 탄소발자국 인증 제품 참고 데이터")
        carbon_df = pd.read_csv(carbon_path)
        carbon_query = st.text_input(
            "제품/업체명 검색",
            placeholder="예: 생수",
            key="carbon_search",
        )
        filtered_carbon = carbon_df.copy()
        if carbon_query:
            mask = (
                carbon_df["제품명"].astype(str)
                .str.contains(carbon_query, case=False, na=False)
                | carbon_df["업체명"].astype(str)
                .str.contains(carbon_query, case=False, na=False)
            )
            filtered_carbon = carbon_df[mask]
        st.dataframe(
            filtered_carbon,
            width="stretch",
            hide_index=True,
        )

        if not filtered_carbon.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("탄소배출량 상위 10개 제품")
                top10 = (
                    filtered_carbon
                    .sort_values("탄소배출량", ascending=False)
                    .head(10)
                )
                chart = alt.Chart(top10).mark_bar().encode(
                    x=alt.X(
                        "탄소배출량:Q",
                        scale=alt.Scale(domainMin=0, nice=False),
                    ),
                    y=alt.Y("제품명:N", sort="-x"),
                )
                st.altair_chart(chart, width="stretch")

            with c2:
                st.subheader("제품군별 평균 탄소배출량")
                avg_group = (
                    filtered_carbon
                    .groupby("제품군")["탄소배출량"]
                    .mean()
                    .reset_index()
                    .sort_values("탄소배출량", ascending=False)
                )
                chart = alt.Chart(avg_group).mark_bar().encode(
                    x=alt.X(
                        "탄소배출량:Q",
                        scale=alt.Scale(domainMin=0, nice=False),
                    ),
                    y=alt.Y("제품군:N", sort="-x"),
                )
                st.altair_chart(chart, width="stretch")

with reward_tabs[3]:
    st.subheader("🎁 포인트로 기부하기")
    st.metric(
        "사용 가능한 포인트",
        f"{st.session_state.points_balance:,} P",
    )

    if st.session_state.points_balance <= 0:
        st.info("적립된 포인트가 없습니다.")
    else:
        cause = st.selectbox(
            "기부처를 선택하세요",
            DONATION_CAUSES,
            key="donate_cause",
        )
        max_points = int(st.session_state.points_balance)
        step = 10 if max_points >= 10 else 1
        amount = st.slider(
            "기부할 포인트",
            min_value=0,
            max_value=max_points,
            value=0,
            step=step,
            key="donate_amount",
        )
        if st.button(
            "💚 기부하기",
            type="primary",
            key="donate_submit",
        ):
            if amount <= 0:
                st.warning("기부할 포인트를 선택해주세요.")
            else:
                st.session_state.points_balance -= amount
                st.session_state.donations.append({
                    "시각": datetime.now(),
                    "기부처": cause,
                    "포인트": amount,
                })
                save_reward_state()
                st.success(
                    f"{cause}에 {amount}P를 기부했습니다."
                )
                st.rerun()

    if st.session_state.donations:
        st.divider()
        st.subheader("나의 기부 이력")
        donation_df = pd.DataFrame(st.session_state.donations)
        st.dataframe(
            donation_df,
            width="stretch",
            hide_index=True,
        )
        st.metric(
            "누적 기부 포인트",
            f"{int(donation_df['포인트'].sum()):,} P",
        )

with reward_tabs[4]:
    compare_df = pd.DataFrame({
        "구분": ["도입 전(전품목 수거)", "도입 후(AI 검사)"],
        "기계 고장 빈도(월/건)": [8, 2],
        "이물질 혼입률(%)": [23, 4],
    })

    st.subheader("AI 분류 도입 전후 비교")
    c1, c2 = st.columns(2)
    with c1:
        chart = alt.Chart(compare_df).mark_bar().encode(
            x="구분:N",
            y=alt.Y(
                "기계 고장 빈도(월/건):Q",
                scale=alt.Scale(domainMin=0, nice=False),
            ),
        )
        st.altair_chart(chart, width="stretch")
    with c2:
        chart = alt.Chart(compare_df).mark_bar().encode(
            x="구분:N",
            y=alt.Y(
                "이물질 혼입률(%):Q",
                scale=alt.Scale(domainMin=0, nice=False),
            ),
        )
        st.altair_chart(chart, width="stretch")

    st.caption(
        "※ 비교 수치는 app_jiwon.py의 데모용 참고값을 유지했습니다. "
        "실제 운영 데이터가 생기면 교체해야 합니다."
    )

    st.divider()
    st.subheader("🚨 회수기 오류 현황 — 데모용 합성 데이터")
    random.seed(42)
    hours = list(range(24))
    error_by_hour = pd.DataFrame({
        "시간대": hours,
        "반려 건수": [
            random.randint(0, 3)
            if 7 <= h <= 22
            else random.randint(3, 12)
            for h in hours
        ],
    })
    error_chart = alt.Chart(error_by_hour).mark_bar().encode(
        x=alt.X("시간대:O", axis=alt.Axis(labelAngle=0)),
        y=alt.Y(
            "반려 건수:Q",
            scale=alt.Scale(domainMin=0, nice=False),
        ),
    )
    st.altair_chart(error_chart, width="stretch")
