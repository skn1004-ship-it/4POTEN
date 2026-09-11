# =========================================================
# A+B 통합 7버전 (사이드바 버튼형 UI 개선 적용)
# 기준: 통합 6버전 유지 + 실시간 웹캠 포인트 고정 표시 영역 추가
# + 사이드바 메뉴를 라디오(동그라미) → 순수 텍스트 버튼형으로 교체
# =========================================================
import os
import json
import base64
import random
import sys
import asyncio
from datetime import datetime, timedelta
import cv2
AUTO_STOP_IDLE_SECONDS = 2  # 마지막 탐지 이후 이 시간(초)동안 새 탐지가 없으면 자동 정지

# =========================================================
# 🟢 [추가] Windows + uvicorn 연결 종료 경고(WinError 10054) 억제
# ---------------------------------------------------------
if sys.platform == "win32":
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport

        def _silence_connection_reset(func):
            def wrapper(self, *args, **kwargs):
                try:
                    return func(self, *args, **kwargs)
                except ConnectionResetError:
                    pass

            return wrapper

        _ProactorBasePipeTransport._call_connection_lost = _silence_connection_reset(
            _ProactorBasePipeTransport._call_connection_lost
        )
    except Exception:
        pass

import streamlit as st
import torch
import pandas as pd
from streamlit_echarts import st_echarts, JsCode

import numpy as np
import time

from PIL import Image, ImageOps
from transformers import AutoImageProcessor, SiglipForImageClassification
from ultralytics import YOLO

# =========================================================
# 1. 페이지 설정
# =========================================================
st.set_page_config(
    page_title="재활용품 AI 분류",
    page_icon="♻️",
    layout="wide"
)

if "page" not in st.session_state:
    st.session_state.page = "platform"

# =========================================================
# AI 판정 기준
# =========================================================
CONFIDENCE_THRESHOLD = 0.80
YOLO_CONFIDENCE = 0.25
SIGLIP_STRONG_THRESHOLD = 0.80
BOX_PADDING = 8
WEBCAM_FRAME_INTERVAL = 15
WEBCAM_LOG_INTERVAL = 1.0
WEBCAM_AUTO_STOP_CONFIDENCE = 0.80  # 실시간 웹캠: 신뢰도가 이 값 이상이면 자동 정지
REANALYSIS_ATTEMPTS = 1
TRACKER_CONFIG = "botsort.yaml"
DUPLICATE_BOX_IOU_THRESHOLD = 0.65

# =========================================================
# 포인트 / 탄소중립 기능 설정
# =========================================================
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data_ref")
os.makedirs(DATA_DIR, exist_ok=True)

# 🟢 사이드바 상단 로고 (assets/logo.png 등에 파일을 넣어두면 자동 표시)
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


def _find_logo_path():
    for _name in ("logo.png", "logo.jpg", "logo.jpeg", "logo.webp", "4poten.png", "4poten_logo.png"):
        _p = os.path.join(ASSETS_DIR, _name)
        if os.path.exists(_p):
            return _p
    return None


def render_sidebar_logo():
    _p = _find_logo_path()
    if not _p:
        return
    _ext = os.path.splitext(_p)[1].lower().lstrip(".")
    _mime = "jpeg" if _ext in ("jpg", "jpeg") else _ext
    with open(_p, "rb") as _f:
        _b64 = base64.b64encode(_f.read()).decode()
    st.markdown(
        f'''
        <div style="text-align:center; padding:0; margin:-0.5rem 0 56px;">
            <img src="data:image/{_mime};base64,{_b64}"
                 style="width:230px; max-width:100%; height:auto; display:block; margin:0 auto;" />
        </div>
        ''',
        unsafe_allow_html=True,
    )

APP_REWARD_INFO = {
    "플라스틱": {"rate": 20, "carbon_factor": 2.0, "avg_kg": 0.05},
    "캔": {"rate": 50, "carbon_factor": 9.0, "avg_kg": 0.015},
    "종이팩": {"rate": 30, "carbon_factor": 1.0, "avg_kg": 0.02},  # TODO: carbon_factor/avg_kg는 추정치, 검토 필요
}

# 🟢 재활용품 수거 대상(플라스틱/캔/종이팩) 외 물체에 표시할 통일된 경고 문구.
#    실제로 어떤 품목인지는 노출하지 않는다.
OUT_OF_SCOPE_LABEL = "재활용 대상 아님"
OUT_OF_SCOPE_WARNING = "플라스틱, 종이, 캔 제외한 다른 품목들은 분류가 인식되더라도 🚫재활용수거대상이 아닙니다. 다시 확인후 투입해주십시오."

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

B_CLASS_INFO = {
    "plastic":      {"label": "플라스틱", "rate": 20},
    "can":            {"label": "캔",           "rate": 50},
    "carton":         {"label": "종이팩",       "rate": 30},
}

# =========================================================
# UI 스타일
# 🟢 [변경] 사이드바 라디오 숨김/스타일 CSS를 제거하고
#           st.button 기반 텍스트 메뉴용 스타일로 교체
# =========================================================
st.markdown("""
<style>
/* 🟢 상단 고정 헤더(툴바)는 투명 유지 (높이는 건드리지 않음) */
[data-testid="stHeader"] {
    background: transparent !important;
}

/* 🟢 메인 콘텐츠 컨테이너: 상단 여백을 사이드바 첫 메뉴 높이에 맞춤 + 왼쪽 여백 축소 */
/*    (Streamlit 최신 버전은 .main 클래스가 없으므로 stMainBlockContainer / block-container 직접 타겟) */
/*    ⬇️ 헤더 높이가 어긋나면 padding-top 값만 조절하세요 (사이드바 첫 메뉴와 눈으로 맞추기) */
[data-testid="stMainBlockContainer"],
[data-testid="stMain"] .block-container,
section.main > .block-container,
.block-container {
    padding-top: 3.5rem !important;
    padding-left: 1.2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}

/* 🟢 메인 최상단 제목의 위쪽 마진 제거 → 사이드바 첫 메뉴와 같은 높이에서 시작 */
[data-testid="stMainBlockContainer"] > div:first-child,
[data-testid="stMain"] .block-container [data-testid="stVerticalBlock"] > div:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
}
[data-testid="stMain"] [data-testid="stHeading"]:first-child h1,
[data-testid="stMain"] [data-testid="stHeading"]:first-child h2,
[data-testid="stMain"] [data-testid="stHeading"]:first-child h3,
[data-testid="stMain"] .block-container h1:first-child,
[data-testid="stMain"] .block-container h2:first-child,
[data-testid="stMain"] .block-container h3:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* 🟢 사이드바 상단 헤더(접기 버튼 영역) 여백 축소 → 로고 위 공백 제거 */
[data-testid="stSidebarHeader"] {
    padding-top: 0.15rem !important;
    padding-bottom: 0 !important;
    min-height: 0 !important;
    height: auto !important;
}

/* 🟢 사이드바 안쪽 여백 축소 + 첫 메뉴를 메인 제목과 같은 상단 높이에 맞춤 */
[data-testid="stSidebarUserContent"],
[data-testid="stSidebarContent"] {
    padding-top: 0 !important;
    padding-left: 0.6rem !important;
    padding-right: 0.6rem !important;
}

/* CSS 주입용 빈 마크다운 컨테이너가 상단 여백을 밀어내지 않도록 제거 */
[data-testid="stMain"] [data-testid="stMarkdownContainer"]:empty {
    display: none !important;
}

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
.points-card {
    background: #eafaf0;
    border-radius: 16px;
    padding: 16px 22px;
    display: flex;
    align-items: center;
    gap: 14px;
}
.points-card .icon { font-size: 30px; line-height: 1; }
.points-card .label { font-size: 14px; color: #1e293b; font-weight: 600; margin-bottom: 4px; }
.points-card .value { font-size: 30px; font-weight: 800; color: #10b981; line-height: 1.1; }
.impact-banner {
    background: #eafaf0;
    border-radius: 16px;
    padding: 18px 24px;
    display: flex;
    align-items: center;
    gap: 24px;
    flex-wrap: wrap;
}
.impact-banner .impact-main {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-right: 24px;
    border-right: 1px solid rgba(16, 185, 129, 0.25);
}
.impact-banner .impact-main .icon { font-size: 30px; line-height: 1; }
.impact-banner .impact-main .label { font-size: 13px; color: #1e293b; font-weight: 600; margin-bottom: 4px; }
.impact-banner .impact-main .value { font-size: 26px; font-weight: 800; color: #10b981; line-height: 1.1; }
.impact-banner .impact-message { flex: 1 1 220px; }
.impact-banner .impact-message .msg-title { font-size: 14px; font-weight: 700; color: #0f5132; margin-bottom: 2px; }
.impact-banner .impact-message .msg-sub { font-size: 12px; color: #64748b; }
.stat-card {
    background: #eafaf0;
    border-radius: 14px;
    padding: 16px 18px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.stat-card .icon-badge {
    width: 40px;
    height: 40px;
    min-width: 40px;
    border-radius: 50%;
    background: #cdeeda;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 19px;
}
.stat-card .label { font-size: 13px; color: #475569; font-weight: 500; margin-bottom: 4px; }
.stat-card .value { font-size: 22px; font-weight: 800; color: #10b981; line-height: 1.1; white-space: nowrap; }
.success-box {
    background: #f0fff4;
    border: 1px solid #86efac;
    border-radius: 12px;
    padding: 16px 18px;
    color: #14532d;
    font-weight: 500;
}
.briefing-box {
    background: #eafaf0;
    border: 1px solid #a7e8c1;
    border-radius: 12px;
    padding: 18px 20px;
    color: #0f5132;
    line-height: 1.6;
}

/* 🟢 사이드바 메뉴 버튼 스타일 (텍스트만, 동그라미 없음, 완전 왼쪽 정렬) */
[data-testid="stSidebar"] .stButton > button {
    display: flex !important;
    justify-content: flex-start !important;
    align-items: center !important;
    text-align: left !important;
    width: 100% !important;
    padding: 12px 8px !important;
    margin-bottom: 2px !important;
    border-radius: 8px !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    border: none !important;
    background-color: transparent !important;
    color: #1e293b !important;
    box-shadow: none !important;
    line-height: 1.2 !important;
    transition: all 0.2s ease !important;
}

/* 🟢 버튼 내부 컨테이너(div)도 왼쪽 정렬 강제 */
[data-testid="stSidebar"] .stButton > button > div {
    display: flex !important;
    justify-content: flex-start !important;
    width: 100% !important;
}

/* 🟢 버튼 내부 텍스트(p)도 왼쪽 정렬 강제 */
[data-testid="stSidebar"] .stButton > button p {
    text-align: left !important;
    width: 100% !important;
    margin: 0 !important;
    font-size: 20px !important;
    font-weight: 700 !important;
}

/* 🟢 버튼 사이 세로 여백 자체(스트림릿 element 컨테이너) 줄이기 */
[data-testid="stSidebar"] .stButton {
    margin-bottom: 0px !important;
}

/* 🟢 마우스 호버 시 배경색 (연두) */
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #e3f6e9 !important;
    color: #1e293b !important;
    border: none !important;
}

/* 🟢 선택된(활성) 메뉴 강조 스타일 — type="primary" 버튼에 적용 (연두) */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background-color: #cdeeda !important;
    color: #1e293b !important;
    border-left: 4px solid #10b981 !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background-color: #cdeeda !important;
}

/* =========================================================
   🟢 연보라(라벤더 #f0f2f6) → 연두색(#eaf7ee) 일괄 통일
   * 기본 배경은 .streamlit/config.toml 의 secondaryBackgroundColor 로 처리되고,
     아래는 캔버스/특수 위젯까지 확실히 덮기 위한 보강 규칙 (색상 값 동일하게 유지)
   ========================================================= */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"],
[data-baseweb="select"] > div,
[data-baseweb="input"],
[data-baseweb="base-input"],
.stTextInput div[data-baseweb="input"],
.stNumberInput div[data-baseweb="input"],
[data-testid="stTextInput"] input,
[data-testid="stNumberInputContainer"],
[data-testid="stDateInput"] div[data-baseweb="input"],
.stMultiSelect div[data-baseweb="select"] > div,
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="menu"],
[data-testid="stFileUploaderDropzone"],
[data-testid="stMain"] code,
[data-testid="stSidebar"] code,
pre, .stCodeBlock,
[data-baseweb="tab-list"],
[data-testid="stExpander"] summary {
    background-color: #eaf7ee !important;
}

/* 🟢 정보(st.info) 알림 박스: 기본 파란색 대신 연두색으로 통일 */
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]),
.stAlert:has([data-testid="stAlertContentInfo"]) [data-baseweb="notification"],
[data-testid="stAlertContentInfo"] {
    background-color: #eaf7ee !important;
    color: #0f5132 !important;
}
[data-testid="stAlertContentInfo"] svg {
    fill: #10b981 !important;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 2. ECharts 공통 헬퍼 (친환경 테마 통일 · streamlit-echarts)
#    - 강조색 #10b981(에메랄드) / 보조색 #3b82f6 / 포인트색 #f59e0b
#    - tooltip 인터랙션 + grid.containLabel True 반응형
#    - 데이터가 비어 있으면 st.info 로 안내하고 렌더링을 건너뜀
# =========================================================
ECO_PRIMARY = "#10b981"
ECO_SECONDARY = "#3b82f6"
ECO_ACCENT = "#f59e0b"
ECO_PALETTE = [
    "#10b981", "#3b82f6", "#f59e0b", "#6366f1",
    "#14b8a6", "#ec4899", "#84cc16", "#f97316",
]
_ECO_FONT = "Pretendard, -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif"
_ECO_TEXT_STYLE = {"fontFamily": _ECO_FONT, "color": "#1e293b"}
_ECO_GRID = {"left": "3%", "right": "5%", "bottom": "4%", "top": "16%", "containLabel": True}
_ECO_TOOLTIP_BOX = {
    "backgroundColor": "rgba(255, 255, 255, 0.96)",
    "borderColor": "#e2e8f0",
    "borderWidth": 1,
    "textStyle": {"color": "#1e293b", "fontFamily": _ECO_FONT},
}


def _eco_num(v):
    """NaN 방어 + 정수는 정수로 표기."""
    try:
        if v is None or pd.isna(v):
            return None
    except TypeError:
        pass
    f = float(v)
    return int(f) if f.is_integer() else round(f, 3)


def _eco_axis_tooltip(unit=""):
    return {
        "trigger": "axis",
        "axisPointer": {"type": "shadow"},
        **_ECO_TOOLTIP_BOX,
        "formatter": JsCode(
            "function (ps) {"
            "  if (!ps || !ps.length) { return ''; }"
            "  var s = ps[0].axisValueLabel;"
            "  for (var i = 0; i < ps.length; i++) {"
            "    var v = ps[i].value;"
            "    v = (v === null || v === undefined) ? '-' : Number(v).toLocaleString();"
            "    s += '<br/>' + ps[i].marker + ' ' + ps[i].seriesName + ': <b>' + v"
            "      + '" + unit + "</b>';"
            "  }"
            "  return s;"
            "}"
        ).js_code,
    }


def _eco_item_tooltip(unit=""):
    return {
        "trigger": "item",
        **_ECO_TOOLTIP_BOX,
        "formatter": JsCode(
            "function (p) {"
            "  var v = (p.value === null || p.value === undefined) ? '-' : Number(p.value).toLocaleString();"
            "  var pct = (p.percent !== undefined) ? '  (' + p.percent + '%)' : '';"
            "  return p.name + '<br/>' + p.marker + ' <b>' + v + '" + unit + "</b>' + pct;"
            "}"
        ).js_code,
    }


#  x축 라벨이 겹치기 시작하는(=스크롤/솎아내기가 필요한) 카테고리 개수 기준
_ECO_MANY = 14


def _eco_cat_axis(categories, *, horizontal):
    """카테고리 축. 항목이 많으면 라벨을 자동으로 솎아내고 기울여 가독성을 확보한다."""
    n = len(categories)
    # 라벨은 항상 수평 유지. 항목이 많으면 기울이지 않고 자동으로 솎아낸다.
    interval = 0 if (horizontal or n <= _ECO_MANY) else "auto"
    rotate = 0
    return {
        "type": "category",
        "data": list(categories),
        "axisTick": {"show": False},
        "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
        "axisLabel": {"color": "#475569", "interval": interval, "rotate": rotate,
                      "hideOverlap": True, "fontSize": 11},
    }


def _eco_datazoom(n, *, window=45):
    """카테고리가 많을 때만 확대/스크롤(dataZoom)을 붙인다. 처음에는 최근 구간만 보여준다."""
    if n <= _ECO_MANY:
        return None
    start = max(0, 100 - window / n * 100)
    return [
        {"type": "inside", "start": start, "end": 100, "zoomOnMouseWheel": False,
         "moveOnMouseWheel": True, "moveOnMouseMove": True},
        {"type": "slider", "start": start, "end": 100, "height": 18, "bottom": 8,
         "borderColor": "#e2e8f0", "fillerColor": "rgba(16,185,129,0.15)",
         "handleStyle": {"color": "#10b981"}, "textStyle": {"color": "#94a3b8", "fontSize": 10}},
    ]


def _eco_grid_for(n):
    """dataZoom 슬라이더/기운 라벨이 들어갈 아래 여백을 확보한 grid."""
    if n > _ECO_MANY:
        return {"left": "3%", "right": "5%", "bottom": 56, "top": "16%", "containLabel": True}
    return _ECO_GRID


def _eco_value_axis(name=""):
    return {
        "type": "value",
        "name": name,
        "nameTextStyle": {"color": "#64748b"},
        "axisLabel": {"color": "#475569"},
        "axisLine": {"show": False},
        "splitLine": {"lineStyle": {"color": "#eef2f6"}},
    }


def eco_bar_chart(categories, values, *, key, unit="", value_name="",
                  horizontal=False, rounded=True, color=ECO_PRIMARY,
                  highlight_top=0, bar_colors=None, height="360px"):
    """막대 차트. horizontal=True 이면 가로 막대(위에서부터 큰 값이 되도록 오름차순 정렬해서 전달).

    bar_colors: 막대별 색을 직접 지정할 때 사용(길이는 categories 와 동일).
    """
    categories = list(categories)
    values = [_eco_num(v) for v in values]
    if not categories:
        st.info("표시할 데이터가 없습니다.")
        return

    n = len(values)
    bar_data = []
    for i, v in enumerate(values):
        if bar_colors is not None:
            bar_data.append({"value": v, "itemStyle": {"color": bar_colors[i]}})
            continue
        top = highlight_top and (i >= n - highlight_top)
        bar_data.append({"value": v, "itemStyle": {"color": ECO_ACCENT if top else color}})

    if rounded:
        radius = [0, 6, 6, 0] if horizontal else [6, 6, 0, 0]
    else:
        radius = 0

    cat_axis = _eco_cat_axis(categories, horizontal=horizontal)
    val_axis = _eco_value_axis(value_name)
    zoom = None if horizontal else _eco_datazoom(n)
    option = {
        "color": ECO_PALETTE,
        "textStyle": _ECO_TEXT_STYLE,
        "grid": _ECO_GRID if horizontal else _eco_grid_for(n),
        "tooltip": _eco_axis_tooltip(unit),
        "xAxis": val_axis if horizontal else cat_axis,
        "yAxis": cat_axis if horizontal else val_axis,
        "series": [{
            "name": value_name or "값",
            "type": "bar",
            "data": bar_data,
            "barMaxWidth": 46,
            "itemStyle": {"borderRadius": radius},
            "emphasis": {"focus": "series"},
        }],
    }
    if zoom:
        option["dataZoom"] = zoom
    st_echarts(options=option, height=height, key=key)


def eco_bar_line_chart(categories, bar_values, line_values, *, key,
                       bar_name="포인트", line_name="탄소절감(kg)",
                       bar_unit=" P", line_unit=" kg", height="380px"):
    """막대(왼쪽 축) + 곡선(오른쪽 축) 이중축 차트."""
    categories = list(categories)
    if not categories:
        st.info("표시할 데이터가 없습니다.")
        return

    n = len(categories)
    cat_axis = _eco_cat_axis(categories, horizontal=False)
    cat_axis["boundaryGap"] = True
    zoom = _eco_datazoom(n)
    option = {
        "color": [ECO_PRIMARY, ECO_ACCENT],
        "textStyle": _ECO_TEXT_STYLE,
        "grid": _eco_grid_for(n),
        "legend": {"data": [bar_name, line_name], "top": 4, "textStyle": {"color": "#475569"}},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}, **_ECO_TOOLTIP_BOX},
        "xAxis": cat_axis,
        "yAxis": [
            {"type": "value", "name": bar_name, "position": "left",
             "nameTextStyle": {"color": "#64748b"},
             "axisLabel": {"color": "#475569", "formatter": "{value}" + bar_unit},
             "splitLine": {"lineStyle": {"color": "#eef2f6"}}},
            {"type": "value", "name": line_name, "position": "right",
             "nameTextStyle": {"color": "#64748b"},
             "axisLabel": {"color": "#475569", "formatter": "{value}" + line_unit},
             "splitLine": {"show": False}},
        ],
        "series": [
            {"name": bar_name, "type": "bar",
             "data": [_eco_num(v) for v in bar_values],
             "barMaxWidth": 40,
             "itemStyle": {"borderRadius": [6, 6, 0, 0], "color": ECO_PRIMARY},
             "emphasis": {"focus": "series"}},
            {"name": line_name, "type": "line", "yAxisIndex": 1, "smooth": True,
             "data": [_eco_num(v) for v in line_values],
             "symbol": "circle", "symbolSize": 7,
             "lineStyle": {"width": 3, "color": ECO_ACCENT},
             "itemStyle": {"color": ECO_ACCENT},
             "areaStyle": {"opacity": 0.08, "color": ECO_ACCENT}},
        ],
    }
    if zoom:
        option["dataZoom"] = zoom
    st_echarts(options=option, height=height, key=key)


def eco_donut_chart(labels, values, *, key, unit="회", height="360px"):
    """도넛형 파이 차트."""
    labels = list(labels)
    if not labels:
        st.info("표시할 데이터가 없습니다.")
        return

    data = [{"name": str(name), "value": _eco_num(v)} for name, v in zip(labels, values)]
    option = {
        "color": ECO_PALETTE,
        "textStyle": _ECO_TEXT_STYLE,
        "tooltip": _eco_item_tooltip(unit),
        "legend": {"orient": "vertical", "left": "left", "top": "center",
                   "textStyle": {"color": "#475569"}},
        "series": [{
            "name": "분류 현황",
            "type": "pie",
            "radius": ["45%", "72%"],
            "center": ["62%", "54%"],
            "avoidLabelOverlap": True,
            "itemStyle": {"borderColor": "#ffffff", "borderWidth": 2, "borderRadius": 6},
            "label": {"show": True, "formatter": "{b}\n{d}%", "color": "#475569"},
            "labelLine": {"length": 12, "length2": 10},
            "emphasis": {"label": {"show": True, "fontSize": 14, "fontWeight": "bold"},
                         "itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0,0,0,0.15)"}},
            "data": data,
        }],
    }
    st_echarts(options=option, height=height, key=key)


def eco_area_chart(categories, values, *, key, unit="건", value_name="",
                   color=ECO_SECONDARY, height="340px"):
    """곡선(smooth) 영역 차트."""
    categories = [str(c) for c in categories]
    if not categories:
        st.info("표시할 데이터가 없습니다.")
        return

    r, g, b = tuple(int(color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    option = {
        "color": [color],
        "textStyle": _ECO_TEXT_STYLE,
        "grid": _ECO_GRID,
        "tooltip": _eco_axis_tooltip(unit),
        "xAxis": {
            "type": "category",
            "data": categories,
            "boundaryGap": False,
            "axisLabel": {"color": "#475569"},
            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
        },
        "yAxis": _eco_value_axis(value_name),
        "series": [{
            "name": value_name or "값",
            "type": "line",
            "smooth": True,
            "data": [_eco_num(v) for v in values],
            "symbol": "circle",
            "symbolSize": 6,
            "lineStyle": {"width": 3, "color": color},
            "itemStyle": {"color": color},
            "areaStyle": {
                "opacity": 0.25,
                "color": {
                    "type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                    "colorStops": [
                        {"offset": 0, "color": f"rgba({r},{g},{b},0.45)"},
                        {"offset": 1, "color": f"rgba({r},{g},{b},0.02)"},
                    ],
                },
            },
        }],
    }
    st_echarts(options=option, height=height, key=key)


# 순위가 높을수록(값이 클수록) 진한 에메랄드가 되도록 만든 에메랄드 그린 계열 팔레트
_ECO_GREENS = ["#065f46", "#047857", "#059669", "#10b981", "#34d399", "#6ee7b7", "#a7f3d0"]


def eco_bubble_chart(labels, values, *, key, unit="", value_name="값", height="480px"):
    """진환경 버블 클러스터(Packed Bubble). force 레이아웃으로 원들이 서로 밀치며 뭉친다.

    각 원의 넓이가 값에 비례하도록 반지름은 sqrt 스케일로 계산한다.
    """
    pairs = [(str(n), _eco_num(v)) for n, v in zip(labels, values)]
    pairs = [p for p in pairs if p[1] is not None and p[1] > 0]
    if not pairs:
        st.info("표시할 데이터가 없습니다.")
        return

    pairs.sort(key=lambda t: t[1], reverse=True)
    max_v = pairs[0][1]
    min_px, max_px = 24, 132

    data = []
    for i, (n, v) in enumerate(pairs):
        size = min_px + (max_px - min_px) * (v / max_v) ** 0.5
        data.append({
            "name": n,
            "value": v,
            "symbolSize": round(size, 1),
            "itemStyle": {
                "color": _ECO_GREENS[min(i, len(_ECO_GREENS) - 1)],
                "shadowBlur": 8,
                "shadowColor": "rgba(15,23,42,0.12)",
                "borderColor": "#ffffff",
                "borderWidth": 1.5,
            },
            "label": {"show": size >= 44},
        })

    option = {
        "textStyle": _ECO_TEXT_STYLE,
        "tooltip": {
            "trigger": "item",
            **_ECO_TOOLTIP_BOX,
            "formatter": JsCode(
                "function (p) {"
                "  var v = (p.value === null || p.value === undefined) ? '-' : Number(p.value).toLocaleString();"
                "  return '<b>' + p.name + '</b><br/>' + (p.marker || '') + ' " + value_name
                + ": <b>' + v + '" + unit + "</b>';"
                "}"
            ).js_code,
        },
        "series": [{
            "type": "graph",
            "layout": "force",
            "roam": True,
            "draggable": True,
            "force": {
                "repulsion": 92,
                "gravity": 0.22,
                "friction": 0.16,
                "edgeLength": 4,
                "layoutAnimation": True,
            },
            "label": {
                "show": True,
                "position": "inside",
                "color": "#ffffff",
                "fontFamily": _ECO_FONT,
                "fontSize": 11,
                "lineHeight": 14,
                "formatter": JsCode(
                    "function (p) {"
                    "  return p.name + '\\n' + Number(p.value).toLocaleString() + '" + unit + "';"
                    "}"
                ).js_code,
            },
            "emphasis": {
                "scale": 1.06,
                "label": {"fontWeight": "bold"},
                "itemStyle": {"shadowBlur": 16, "shadowColor": "rgba(15,23,42,0.25)"},
            },
            "data": data,
            "links": [],
        }],
    }
    st_echarts(options=option, height=height, key=key)


_ECO_RANK_GRID_CSS = """
<style>
.eco-rank-grid{display:grid;grid-auto-flow:column;gap:8px 22px;margin:2px 0 6px;}
.erg-row{display:flex;align-items:center;gap:11px;padding:9px 13px;
  border:1px solid #e6eaf0;border-radius:11px;background:#ffffff;}
.erg-row.erg-hi{border-color:#bbf7e0;background:#f0fdf7;}
.erg-badge{flex:none;width:23px;height:23px;border-radius:7px;display:flex;
  align-items:center;justify-content:center;font-size:12px;font-weight:700;
  color:#ffffff;background:#9aa7b8;font-family:%(font)s;}
.erg-row.erg-hi .erg-badge{background:#10b981;}
.erg-name{flex:1;min-width:0;font-size:13px;color:#1e293b;font-weight:600;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-family:%(font)s;}
.erg-val{flex:none;font-size:12px;color:#64748b;font-weight:600;font-family:%(font)s;}
</style>
""" % {"font": _ECO_FONT}


def eco_rank_grid(labels, values, *, unit="", top_n=15, cols=2, highlight_top=3):
    """랭킹 그리드 카드(표 + 뱃지). 상위 highlight_top 은 에메랄드로 강조하고
    나머지는 회색 뱃지로 표시한다. 열은 위→아래 순서(column-major)로 채운다."""
    rows = [(str(n), _eco_num(v)) for n, v in zip(labels, values)]
    rows = [r for r in rows if r[1] is not None]
    if not rows:
        st.info("표시할 데이터가 없습니다.")
        return

    rows.sort(key=lambda t: t[1], reverse=True)
    rows = rows[:top_n]
    per_col = -(-len(rows) // cols)  # ceil

    items = []
    for i, (name, val) in enumerate(rows, start=1):
        hi = " erg-hi" if i <= highlight_top else ""
        num = f"{int(val):,}" if float(val).is_integer() else f"{val:,}"
        items.append(
            f'<div class="erg-row{hi}">'
            f'<span class="erg-badge">{i}</span>'
            f'<span class="erg-name" title="{name}">{name}</span>'
            f'<span class="erg-val">{num}{unit}</span>'
            f'</div>'
        )

    grid = (
        f'<div class="eco-rank-grid" '
        f'style="grid-template-rows:repeat({per_col},auto);'
        f'grid-template-columns:repeat({cols},1fr);">'
        + "".join(items) + "</div>"
    )
    st.markdown(_ECO_RANK_GRID_CSS + grid, unsafe_allow_html=True)


def eco_lollipop_chart(categories, values, *, key, unit="", value_name="값", height="420px"):
    """롤리팝 막대 차트. 얇은 막대 + 연회색 트랙 배경 + 끝단 원형 뱃지 + 우측 텍스트 라벨.

    극단적인 값 격차가 있어도 하위 항목이 뭉개지지 않도록 트랙 배경으로 위치를 보여준다.
    (categories/values 는 위에서부터 큰 값이 되도록 오름차순 정렬해서 전달)
    """
    cats = [str(c) for c in categories]
    vals = [_eco_num(v) for v in values]
    if not cats:
        st.info("표시할 데이터가 없습니다.")
        return

    nums = [v for v in vals if v is not None]
    max_v = max(nums) if nums else 0
    scatter_data = [[v, cats[i]] for i, v in enumerate(vals) if v is not None]
    option = {
        "color": [ECO_PRIMARY],
        "textStyle": _ECO_TEXT_STYLE,
        "grid": {"left": "3%", "right": "16%", "bottom": "4%", "top": "12%", "containLabel": True},
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {"type": "shadow", "shadowStyle": {"color": "rgba(16,185,129,0.06)"}},
            **_ECO_TOOLTIP_BOX,
            "formatter": JsCode(
                "function (ps) {"
                "  if (!ps || !ps.length) { return ''; }"
                "  var p = ps[0];"
                "  var v = Array.isArray(p.value) ? p.value[0] : p.value;"
                "  v = (v === null || v === undefined) ? '-' : Number(v).toLocaleString();"
                "  return p.axisValueLabel + '<br/>' + p.marker + ' " + value_name
                + ": <b>' + v + '" + unit + "</b>';"
                "}"
            ).js_code,
        },
        "xAxis": {
            "type": "value",
            "max": round(max_v * 1.15, 2) if max_v else None,
            "axisLabel": {"color": "#475569"},
            "axisLine": {"show": False},
            "splitLine": {"lineStyle": {"color": "#eef2f6"}},
        },
        "yAxis": {
            "type": "category",
            "data": cats,
            "axisTick": {"show": False},
            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
            "axisLabel": {"color": "#475569", "fontSize": 11},
        },
        "series": [
            {
                "name": value_name,
                "type": "bar",
                "data": vals,
                "barWidth": 6,
                "showBackground": True,
                "backgroundStyle": {"color": "#eef2f6", "borderRadius": 4},
                "itemStyle": {"color": ECO_PRIMARY, "borderRadius": 4},
                "z": 2,
            },
            {
                "name": value_name,
                "type": "scatter",
                "data": scatter_data,
                "symbolSize": 15,
                "itemStyle": {
                    "color": ECO_PRIMARY,
                    "borderColor": "#ffffff",
                    "borderWidth": 2,
                    "shadowBlur": 4,
                    "shadowColor": "rgba(16,185,129,0.35)",
                },
                "label": {
                    "show": True,
                    "position": "right",
                    "distance": 8,
                    "color": "#334155",
                    "fontWeight": "bold",
                    "fontSize": 11,
                    "formatter": JsCode(
                        "function (p) {"
                        "  return Number(p.value[0]).toLocaleString() + '" + unit + "';"
                        "}"
                    ).js_code,
                },
                "z": 3,
            },
        ],
    }
    st_echarts(options=option, height=height, key=key)


# 도넛 + 값 리스트 범례용 10색 팔레트(사진 배색: 그린·블루·오렌지·퍼플·핑크·시안·틸·라임·옐로·그레이)
_ECO_DONUT_PALETTE = [
    "#10b981", "#3b82f6", "#f59e0b", "#8b5cf6", "#ec4899",
    "#06b6d4", "#14b8a6", "#84cc16", "#eab308", "#64748b",
]


def eco_donut_legend_chart(labels, values, *, key, unit="", value_name="값",
                           center_title="제품군별\n평균 배출량", height="360px"):
    """도넛 차트 + 우측 값 리스트 범례. 중앙에 제목, 범례에 항목별 수치를 오른쪽 정렬로 표시한다."""
    rows = [(str(n), _eco_num(v)) for n, v in zip(labels, values)]
    rows = [r for r in rows if r[1] is not None]
    if not rows:
        st.info("표시할 데이터가 없습니다.")
        return

    rows.sort(key=lambda t: t[1], reverse=True)
    names = [n for n, _ in rows]
    data = [{"name": n, "value": v} for n, v in rows]
    label_map = {
        n: f"{v:,.2f}".rstrip("0").rstrip(".") + unit
        for n, v in rows
    }
    _t = center_title.split("\n")
    option = {
        "color": _ECO_DONUT_PALETTE,
        "textStyle": _ECO_TEXT_STYLE,
        "tooltip": _eco_item_tooltip(unit),
        "toolbox": {
            "right": 4, "top": -4,
            "feature": {"saveAsImage": {
                "title": "이미지 저장", "name": "제품군별_평균_탄소배출량", "pixelRatio": 2,
            }},
            "iconStyle": {"borderColor": "#94a3b8"},
            "emphasis": {"iconStyle": {"borderColor": ECO_PRIMARY}},
        },
        "title": {
            "text": _t[0],
            "subtext": _t[1] if len(_t) > 1 else "",
            "left": "27%",
            "top": "40%",
            "textAlign": "center",
            "textStyle": {"color": "#94a3b8", "fontSize": 12, "fontWeight": "normal",
                          "fontFamily": _ECO_FONT},
            "subtextStyle": {"color": "#334155", "fontSize": 15, "fontWeight": "bold",
                             "fontFamily": _ECO_FONT},
        },
        "legend": {
            "type": "scroll",
            "orient": "vertical",
            "right": 8,
            "top": "middle",
            "data": names,
            "icon": "circle",
            "itemWidth": 10,
            "itemHeight": 10,
            "itemGap": 13,
            "textStyle": {
                "color": "#334155",
                "rich": {
                    "name": {"width": 74, "color": "#334155", "fontSize": 12,
                             "fontFamily": _ECO_FONT},
                    "val": {"width": 96, "align": "right", "color": "#64748b",
                            "fontSize": 12, "fontFamily": _ECO_FONT},
                },
            },
            "formatter": JsCode(
                "function (name) {"
                "  var m = " + json.dumps(label_map, ensure_ascii=False) + ";"
                "  return '{name|' + name + '}{val|' + (m[name] || '') + '}';"
                "}"
            ).js_code,
        },
        "series": [{
            "name": value_name,
            "type": "pie",
            "radius": ["45%", "68%"],
            "center": ["27%", "52%"],
            "avoidLabelOverlap": True,
            "label": {"show": False},
            "labelLine": {"show": False},
            "itemStyle": {"borderColor": "#ffffff", "borderWidth": 2},
            "emphasis": {
                "scale": True,
                "scaleSize": 6,
                "itemStyle": {"shadowBlur": 12, "shadowColor": "rgba(15,23,42,0.18)"},
            },
            "data": data,
        }],
    }
    st_echarts(options=option, height=height, key=key)


# =========================================================
# 2-1. SigLIP2 AI 모델 로드
# =========================================================
@st.cache_resource
def load_model():
    model_name = "prithivMLmods/Augmented-Waste-Classifier-SigLIP2"
    local_model_dir = os.path.join(os.path.dirname(__file__), "model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# 2-1. YOLO 모델 로드
# 🟢 best.pt 외에 추가된 aihub_sum_best.pt / aihub_glass_best.pt / roboflow_best.pt도
#    모두 함께 로드하여 탐지에 사용한다 (여러 모델의 탐지 결과를 합쳐서 분류).
# =========================================================
@st.cache_resource
def load_yolo_model(filename):
    model_path = os.path.join(os.path.dirname(__file__), filename)
    return YOLO(model_path)

yolo_model = load_yolo_model("best.pt")
yolo_model_aihub_sum = load_yolo_model("aihub_sum_best.pt")
yolo_model_aihub_glass = load_yolo_model("aihub_glass_best.pt")
yolo_model_roboflow = load_yolo_model("roboflow_best.pt")

# =========================================================
# 3. 이미지 분석 및 매핑 설정
# =========================================================
LABEL_MAP = {
    "Plastic": "플라스틱",
    "Paper": "종이",
    "Metal": "캔",
    "Glass": "유리병",
    "Cardboard": "종이상자",
    "Clothes": "의류",
    "Battery": "건전지",
    "Biological": "생분해성/음식물",
    "Shoes": "신발",
    "Trash": "일반쓰레기"
}

# 🟢 재활용품 수거 대상은 플라스틱 / 캔 / 종이팩 3종뿐 (유리병은 대상 아님).
#    best.pt의 나머지 YOLO 클래스는 매핑하지 않으며(None), 해당 박스는 수거 대상이 아니므로
#    detect_and_classify()에서 결과 목록에서 제외한다.
YOLO_TO_10_MAP = {
    "Aluminum can": "캔",
    "Aluminum caps": "캔",
    "Iron utensils": "캔",
    "Metal shavings": "캔",
    "Scrap metal": "캔",
    "Tin": "캔",
    "Foil": "캔",
    "Aerosols": "캔",
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
    "Milk bottle": "종이팩",
    "Tetra pack": "종이팩"
}

# 🟢 aihub_sum_best.pt / aihub_glass_best.pt 공용 클래스 매핑 (CAN/PAPER/PLASTIC/GLASS)
#    재활용품 수거 대상(플라스틱/캔/종이팩) 외 클래스는 매핑하지 않아 결과에서 제외한다.
AIHUB_LABEL_MAP = {
    "CAN": "캔",
    "PLASTIC": "플라스틱",
}

# 🟢 roboflow_best.pt 클래스 매핑 (can/paper/plastic/glass/organic/tetra_pack/textile)
ROBOFLOW_LABEL_MAP = {
    "can": "캔",
    "plastic": "플라스틱",
    "tetra_pack": "종이팩",
}

# 🟢 모든 YOLO 모델을 함께 실행하여 탐지 결과를 합친다: (모델, 클래스매핑, 이름)
YOLO_MODEL_SOURCES = [
    (yolo_model, YOLO_TO_10_MAP, "best"),
    (yolo_model_aihub_sum, AIHUB_LABEL_MAP, "aihub_sum"),
    (yolo_model_aihub_glass, AIHUB_LABEL_MAP, "aihub_glass"),
    (yolo_model_roboflow, ROBOFLOW_LABEL_MAP, "roboflow"),
]

RECYCLE_GUIDE = {
    "플라스틱": "내용물을 비우고 물로 헹군 뒤, 라벨 스티커를 제거하여 투입하세요.",
    "종이": "물기에 젖지 않게 펼쳐서 배출하며, 테이프/스프링은 제거하세요.",
    "캔": "내용물을 비우고 가급적 압착하여 배출하세요. 가스용기는 구멍을 뚫어야 합니다.",
    "유리병": "담배꽁초 등 이물질을 넣지 말고, 병뚜껑을 분리하여 배출하세요.",
    "종이팩": "내용물을 비우고 물로 헹군 뒤, 펼쳐서 말린 후 종이팩 전용 수거함에 배출하세요.",
    "종이상자": "운송장 스티커와 테이프를 완전히 제거한 후 납작하게 접어 배출하세요.",
    "의류": "의류수거함에 배출하거나 오염이 심할 경우 일반쓰레기로 분류하세요.",
    "건전지": "폐건전지 전용 수거함에 별도로 배출하세요.",
    "생분해성/음식물": "물기를 최대한 제거한 후 음식물 쓰레기 전용 용기에 배출하세요.",
    "신발": "짝을 맞춰 신발 수거함에 배출하거나, 훼손이 심하면 일반쓰레기로 배출하세요.",
    "일반쓰레기": "종량제 봉투에 담아 배출하세요."
}

def predict_trash(image):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)[0]
    results = {}

    for index, probability in enumerate(probabilities):
        raw_label = model.config.id2label[index]
        korean_label = LABEL_MAP.get(raw_label, raw_label)
        results[korean_label] = probability.item()

    return results

def _run_single_model_detections(yolo_source, label_map, model_name, np_image, use_tracking):
    if use_tracking:
        yolo_results = yolo_source.track(
            source=np_image,
            conf=YOLO_CONFIDENCE,
            iou=0.50,
            agnostic_nms=False,
            persist=True,
            tracker=TRACKER_CONFIG,
            verbose=False
        )
    else:
        yolo_results = yolo_source.predict(
            source=np_image,
            conf=YOLO_CONFIDENCE,
            iou=0.50,
            agnostic_nms=False,
            verbose=False
        )

    raw_detections = []
    if not yolo_results:
        return raw_detections

    result = yolo_results[0]
    if result.boxes is None:
        return raw_detections

    for box in result.boxes:
        coordinates = box.xyxy[0].detach().cpu().numpy().astype(int)
        x1, y1, x2, y2 = coordinates

        yolo_class_id = int(box.cls[0].item())
        yolo_score = float(box.conf[0].item())

        track_id = None
        if getattr(box, "id", None) is not None:
            try:
                track_id = int(box.id[0].item())
            except Exception:
                track_id = None

        yolo_label = yolo_source.names[yolo_class_id]
        # 🟢 플라스틱/캔/종이팩 3종 외 클래스는 mapped_label=None → "비대상 품목"으로 표시(제외하지 않음)
        mapped_label = label_map.get(yolo_label)

        raw_detections.append({
            "bbox_raw": (x1, y1, x2, y2),
            "yolo_label": yolo_label,
            "yolo_score": yolo_score,
            "mapped_label": mapped_label,
            "track_id": track_id,
            "model_name": model_name,
        })

    return raw_detections

def detect_and_classify(image, use_tracking=False):
    image = image.convert("RGB")
    np_image = np.array(image)
    image_width, image_height = image.size

    # 🟢 추가된 aihub_sum / aihub_glass / roboflow 모델을 best.pt와 함께 모두 실행하여
    #    탐지 결과를 하나의 목록으로 합친다.
    raw_detections = []
    for yolo_source, label_map, model_name in YOLO_MODEL_SOURCES:
        raw_detections.extend(
            _run_single_model_detections(yolo_source, label_map, model_name, np_image, use_tracking)
        )

    detections = []
    for object_index, raw in enumerate(raw_detections):
        x1, y1, x2, y2 = raw["bbox_raw"]

        x1 = max(0, x1 - BOX_PADDING)
        y1 = max(0, y1 - BOX_PADDING)
        x2 = min(image_width, x2 + BOX_PADDING)
        y2 = min(image_height, y2 + BOX_PADDING)

        if x2 <= x1 or y2 <= y1:
            continue

        cropped_image = image.crop((x1, y1, x2, y2))
        out_of_scope = raw["mapped_label"] is None
        top_label = raw["mapped_label"] if not out_of_scope else OUT_OF_SCOPE_LABEL
        top_score = raw["yolo_score"]
        decision_source = f"YOLO 판정 ({raw['model_name']})" if not out_of_scope else f"YOLO 판정 ({raw['model_name']}) - 비대상"
        # 🟢 비대상 품목은 재분석/재투입 대상이 아니므로 needs_review를 강제로 False 처리한다.
        needs_review = (top_score < CONFIDENCE_THRESHOLD) if not out_of_scope else False
        sorted_results = [(top_label, top_score)]

        detections.append({
            "object_id": object_index + 1,
            "track_id": raw["track_id"],
            "model_name": raw["model_name"],
            "bbox": (x1, y1, x2, y2),
            "crop": cropped_image,
            "yolo_label": raw["yolo_label"],
            "yolo_score": top_score,
            "label": top_label,
            "score": top_score,
            "all_results": sorted_results,
            "decision_source": decision_source,
            "needs_review": needs_review,
            "out_of_scope": out_of_scope,
        })

    return detections

def calculate_iou(box_a, box_b):
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
    if len(detections) <= 1:
        return detections

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

            iou = calculate_iou(candidate_box, existing.get("bbox"))
            if iou >= DUPLICATE_BOX_IOU_THRESHOLD:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(candidate)

    for index, detection in enumerate(kept, start=1):
        detection["object_id"] = index

    return kept

def draw_detections(image, detections):
    annotated = np.array(image.convert("RGB"))
    annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        object_id = detection["object_id"]
        track_id = detection.get("track_id")
        score = detection["score"]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_prefix = f"T{track_id}" if track_id is not None else f"OBJ {object_id}"
        label_text = f"{label_prefix} {score:.0%}"

        cv2.putText(
            annotated,
            label_text,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated

def fallback_classification(image):
    results = predict_trash(image)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    top_label, top_score = sorted_results[0]

    # 🟢 SigLIP2 fallback도 플라스틱/캔/종이팩 3종 외 결과는 비대상 품목으로 표시한다.
    out_of_scope = top_label not in APP_REWARD_INFO
    display_label = top_label if not out_of_scope else OUT_OF_SCOPE_LABEL

    return {
        "object_id": 1,
        "track_id": None,
        "bbox": None,
        "crop": image,
        "yolo_label": "YOLO 탐지 없음",
        "yolo_score": 0.0,
        "label": display_label,
        "score": top_score,
        "all_results": sorted_results,
        "decision_source": "SigLIP2 fallback",
        "needs_review": (top_score < CONFIDENCE_THRESHOLD) if not out_of_scope else False,
        "out_of_scope": out_of_scope,
    }

def reanalyze_uncertain_detection(detection):
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

    retry_image = ImageOps.autocontrast(crop.convert("RGB"))
    retry_results = predict_trash(retry_image)
    sorted_retry = sorted(retry_results.items(), key=lambda x: x[1], reverse=True)

    retry_label, retry_score = sorted_retry[0]
    # 🟢 여러 YOLO 모델을 함께 쓰므로, 원본 판정 라벨(이미 각 모델의 매핑을 거친 값)을 그대로 비교 기준으로 사용한다.
    mapped_yolo_label = detection.get("label")

    retry_success = (
        (mapped_yolo_label is not None and mapped_yolo_label == retry_label and retry_score >= CONFIDENCE_THRESHOLD)
        or retry_score >= SIGLIP_STRONG_THRESHOLD
        or (mapped_yolo_label is None and retry_score >= CONFIDENCE_THRESHOLD)
    )

    detection["reanalysis_count"] = REANALYSIS_ATTEMPTS
    detection["label"] = retry_label
    detection["score"] = retry_score
    detection["all_results"] = sorted_retry

    if retry_success:
        detection["needs_review"] = False
        detection["requires_reentry"] = False
        detection["reanalysis_status"] = "자동 재분석 성공"
        detection["decision_source"] = f"{detection.get('decision_source', '')} → 자동 재분석 성공".strip(" →")
    else:
        detection["needs_review"] = True
        detection["requires_reentry"] = True
        detection["reanalysis_status"] = "자동 재분석 실패 → 재투입 필요"
        detection["decision_source"] = f"{detection.get('decision_source', '')} → 자동 재분석 실패".strip(" →")

    return detection

def reanalyze_detections(detections):
    return [reanalyze_uncertain_detection(d) for d in detections]

# =========================================================
# 포인트 / 탄소중립 데이터 처리
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
COMPANY_LIST = sorted(companies_df["제조사/유통사"].dropna().unique().tolist()) if not companies_df.empty else []

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
            "credited_items": {k: _dt_to_str(v) for k, v in st.session_state.credited_items.items()},
            "points_balance": st.session_state.points_balance,
            "donations": [_dt_to_str(d) for d in st.session_state.donations],
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
        payload["credited_items"] = {k: _str_to_dt(v) for k, v in payload.get("credited_items", {}).items()}
        payload["donations"] = [_str_to_dt(d) for d in payload.get("donations", [])]
        return payload
    except Exception:
        return None

def credit_detection_once(object_key, label, brand):
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

def points_balance_card(label, value, icon="🪙"):
    st.markdown(f"""
    <div class="points-card">
        <div class="icon">{icon}</div>
        <div>
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def stat_card(col, icon, label, value):
    with col:
        st.markdown(f"""
        <div class="stat-card">
            <div class="icon-badge">{icon}</div>
            <div>
                <div class="label">{label}</div>
                <div class="value">{value}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_donation_impact(total_points):
    """누적 기부 포인트를 응원 메시지와 함께 배너로 보여준다."""
    total_points = int(total_points)
    st.markdown(f"""
    <div class="impact-banner">
        <div class="impact-main">
            <div class="icon">🌱</div>
            <div>
                <div class="label">누적 기부 포인트 <span title="지금까지 기부한 포인트의 합계입니다">ⓘ</span></div>
                <div class="value">{total_points:,} P</div>
            </div>
        </div>
        <div class="impact-message">
            <div class="msg-title">당신의 {total_points:,}P가 더 나은 세상을 만들고 있어요.</div>
            <div class="msg-sub">함께하는 마음이 큰 변화를 만듭니다. 앞으로도 따뜻한 나눔을 이어가주세요.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_credit_success(record):
    if not record:
        return
    partner_text = ""
    if record.get("2배적용"):
        partner_text = f"<br>🌱 <b>{record.get('브랜드', '')}</b>는 참여기업입니다 — 포인트가 <b>2배</b> 적립되었습니다!"
    st.markdown(
        f"""<div class="success-box">
        ✅ <b>{record.get('품목', '')}</b> {record.get('수량', 1)}개 적립 완료.{partner_text}<br>
        이번 적립: <b>+{int(record.get('포인트', 0))}P</b> / 
        탄소 절감 <b>+{float(record.get('탄소절감(kg)', 0)):.3f}kg CO2</b>
        </div>""",
        unsafe_allow_html=True,
    )

def render_ai_briefing():
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
                "이 내용을 핵심만 담아 간결한 한국어 한 문장으로 요약해줘. 군더더기 설명 없이 숫자 중심으로."
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
        doubled_part = f" · 참여기업 2배 적립 <b>{doubled}건</b>" if doubled else ""
        st.markdown(f"""<div class="briefing-box">
        🌱 최다 배출 <b>{top_item}</b> · 누적 <b>{total_points}P</b> · 탄소 절감 <b>{total_carbon:.2f}kg</b>{doubled_part}
        </div>""", unsafe_allow_html=True)
        st.caption("※ OPENAI_API_KEY 환경변수를 설정하면 실제 생성형 AI 브리핑으로 전환됩니다.")

def render_reward_dashboard(key_prefix="reward"):
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
    grouped = grouped.sort_values("기간")
    eco_bar_line_chart(
        grouped["기간"], grouped["포인트"], grouped["탄소절감(kg)"],
        key=f"{key_prefix}_trend_chart",
        bar_name="포인트", line_name="탄소절감(kg)",
        bar_unit=" P", line_unit=" kg",
    )
    st.caption("막대 = 기간별 포인트 · 곡선 = 탄소 절감 추이")

    st.subheader("품목별 이번 달 배출 비중")
    if "수량" in hist_df:
        item_count = hist_df.groupby("품목")["수량"].sum().rename("개수").reset_index()
    else:
        item_count = hist_df["품목"].value_counts().rename_axis("품목").reset_index(name="개수")
    item_count = item_count.sort_values("개수")  # 가로 막대: 위로 갈수록 큰 값
    eco_bar_chart(
        item_count["품목"], item_count["개수"],
        key=f"{key_prefix}_item_chart",
        unit="개", value_name="개수", horizontal=True,
    )

    doubled_count = int(hist_df["2배적용"].sum()) if "2배적용" in hist_df else 0
    if doubled_count:
        st.caption(f"🌱 이 중 {doubled_count}건은 탄소중립실천포인트 참여기업 제품으로 2배 적립되었습니다.")
    st.caption("※ 탄소 절감량·비유 수치는 품목별 평균 무게 추정치 x LCA 기반 계수를 곱한 참고값입니다.")

# =========================================================
# 4. 분석 기록 및 세션 상태 초기화
# =========================================================
if "history" not in st.session_state:
    st.session_state.history = []

if "last_batch_results" not in st.session_state:
    st.session_state.last_batch_results = []

if "webcam_snapshot" not in st.session_state:
    st.session_state.webcam_snapshot = None

if "processed_keys" not in st.session_state:
    st.session_state.processed_keys = set()

if "last_webcam_log" not in st.session_state:
    st.session_state.last_webcam_log = 0.0

if "webcam_session_token" not in st.session_state:
    st.session_state.webcam_session_token = None

if "webcam_processed_track_keys" not in st.session_state:
    st.session_state.webcam_processed_track_keys = set()

if "last_webcam_reward" not in st.session_state:
    st.session_state.last_webcam_reward = None

if "webcam_capture" not in st.session_state:
    st.session_state.webcam_capture = None

if "webcam_force_stop" not in st.session_state:
    st.session_state.webcam_force_stop = False

if "pending_point_popups" not in st.session_state:
    st.session_state.pending_point_popups = []

def start_point_popup_batch():
    """새 탐지 1회(이미지 업로드 1회 / 웹캠 자동정지 1회)를 시작할 때 이전 팝업 큐를 비운다.
    → 팝업이 이전 탐지 결과와 누적되지 않고, 이번 탐지 결과만 깔끔하게 뜨도록 한다."""
    st.session_state.pending_point_popups = []

def queue_point_popup(record):
    """이번 탐지 배치에서 적립된 품목을 팝업 큐에 추가한다."""
    if record is None:
        return
    st.session_state.pending_point_popups.append(record)

@st.dialog("🎉 포인트 적립 완료!")
def render_point_popup():
    records = st.session_state.get("pending_point_popups") or []
    if not records:
        return
    total_points = sum(int(r.get("포인트", 0)) for r in records)
    total_carbon = sum(float(r.get("탄소절감(kg)", 0)) for r in records)

    st.markdown(f"### ✅ {len(records)}개 품목 인식 완료")
    st.markdown(f"**총 적립 포인트: +{total_points:,}P** · 탄소 절감 +{total_carbon:.3f}kg CO2")

    for r in records:
        bonus_text = " · 참여기업 2배" if r.get("2배적용") else ""
        st.write(f"- {r.get('품목', '')} +{int(r.get('포인트', 0))}P{bonus_text}")

    st.divider()
    st.metric("💳 현재 보유 포인트", f"{st.session_state.points_balance:,} P")

    if st.button("확인", width="stretch", key="point_popup_confirm_btn"):
        st.session_state.pending_point_popups = []
        st.rerun()

def release_webcam():
    cap = st.session_state.get("webcam_capture")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    st.session_state.webcam_capture = None

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
# 사이드바 (텍스트 버튼형 메뉴)
# 🟢 [변경] st.radio → st.button + session_state.page 방식으로 교체
#           동그라미 없이 순수 텍스트를 누르면 페이지가 전환됨
# =========================================================
_NAV = {
    "♻️ AI 재활용품 플랫폼": "platform",
    "🪙 포인트 적립 현황": "points",
    "🌱 저탄소 제품 & 참여기업": "eco_catalog",
    "💚 포인트로 기부하기": "donate",
}

with st.sidebar:
    render_sidebar_logo()
    for _label, _key in _NAV.items():
        _is_active = (st.session_state.page == _key)
        if st.button(
            _label,
            key=f"nav_{_key}",
            use_container_width=True,
            type="primary" if _is_active else "secondary",
        ):
            st.session_state.page = _key
            st.rerun()

page = st.session_state.page

# =========================================================
# 메뉴 분기
# 🟢 [변경] 모든 페이지 최상단 대표 제목을 st.header()로 통일하고
#           페이지 시작부의 최상단 st.divider() 제거 → 상단 시작 위치/여백 일치
# =========================================================
if page == "platform":
    st.header("♻️ AI 재활용품 플랫폼")


elif page == "points":
    # 🟢 [이동] 사이드바에 있던 "분류 가능 클래스"를 이 페이지 우측 상단으로 이전
    _title_col, _class_col = st.columns([4, 1])
    with _title_col:
        st.header("🪙 포인트 적립 현황")
    with _class_col:
        with st.popover("🪙 품목별 적립 기준", use_container_width=True):
            for info in B_CLASS_INFO.values():
                st.markdown(f"- {info['label']} ({info['rate']}원/개)")

    # 🟢 [이동] AI 재활용품 플랫폼 페이지에 있던 "제조사/브랜드" 선택을 이곳으로 이전.
    #    여기서 고른 브랜드가 다음 분석(스캔)의 2배 적립 대상이 되며, 동시에 아래 이력 필터로도 쓰인다.
    #    이력이 없어도 항상 노출되어야 하므로 credited_items 유무와 무관하게 렌더링.
    brand_options = ["전체", "해당없음"] + COMPANY_LIST
    selected_points_brand = st.selectbox(
        "제조사/브랜드 (참여기업이면 포인트 2배)",
        brand_options,
        key="points_brand_filter",
    )

    if st.session_state.credited_items:
        hist_df_reward = pd.DataFrame(list(st.session_state.credited_items.values()))
        hist_df_reward["시각"] = pd.to_datetime(hist_df_reward["시각"])

        if selected_points_brand != "전체":
            hist_df_reward = hist_df_reward[hist_df_reward["브랜드"] == selected_points_brand]

        p1, p2, p3 = st.columns(3)
        stat_card(p1, "🪙", "현재 보유 포인트", f"{st.session_state.points_balance:,} P")
        stat_card(p2, "🗑️", "누적 배출 개수", f"{int(hist_df_reward['수량'].sum())} 개")
        stat_card(p3, "🍃", "누적 탄소 절감", f"{float(hist_df_reward['탄소절감(kg)'].sum()):.2f} kg CO2")

        st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:1.25rem; font-weight:600; margin:0.5rem 0;'>📅 날짜별 포인트 적립 현황</div>",
            unsafe_allow_html=True,
        )
        _period = st.radio(
            "집계 단위", ["일별", "주별", "월별"], horizontal=True, index=2,
            key="points_daily_period", label_visibility="collapsed",
        )
        daily = hist_df_reward.copy()
        if _period == "일별":
            daily["구간"] = daily["시각"].dt.strftime("%Y-%m-%d")
        elif _period == "주별":
            daily["구간"] = (
                daily["시각"].dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d") + " 주"
            )
        else:
            daily["구간"] = daily["시각"].dt.strftime("%Y-%m")
        daily_grouped = (
            daily.groupby("구간")
            .agg(포인트=("포인트", "sum"), **{"배출 개수": ("수량", "sum")})
            .reset_index()
            .sort_values("구간")
        )
        eco_bar_chart(
            daily_grouped["구간"], daily_grouped["포인트"],
            key="points_daily_chart",
            unit=" P", value_name="포인트", rounded=True, height="380px",
        )
        if len(daily_grouped) > _ECO_MANY:
            st.caption("막대가 많을 경우 아래 슬라이더로 기간을 좁혀 볼 수 있습니다.")
        st.dataframe(
            daily_grouped.sort_values("구간", ascending=False).rename(columns={"구간": _period}),
            width="stretch",
            hide_index=True,
        )

        st.divider()
        st.subheader("🤖 AI 배출 브리핑")
        render_ai_briefing()

elif page == "eco_catalog":
    st.header("🌱 저탄소 제품 & 참여기업")

    # NOTE: streamlit-echarts 차트는 st.tabs 의 비활성 패널(폭 0)에서 마운트되면
    #       탭 전환 시 리사이즈되지 않아 빈 화면으로 남는다. 선택된 섹션만 렌더링하는
    #       segmented_control 로 탭 UX 를 유지하면서 차트가 항상 보이는 컨테이너에 그려지게 한다.
    _ECO_TABS = ["🏢 탄소중립 참여기업", "🏷️ 제품별 탄소발자국"]
    eco_tab = st.segmented_control(
        "보기 선택", _ECO_TABS, default=_ECO_TABS[0],
        key="eco_catalog_tab", label_visibility="collapsed",
    ) or _ECO_TABS[0]

    if eco_tab == _ECO_TABS[0]:
        if companies_df.empty:
            st.warning("data_ref/carbon_neutral_companies.csv 파일이 없습니다.")
        else:
            c1, c2 = st.columns(2)
            stat_card(c1, "🏢", "참여기업 수", f"{len(companies_df)} 곳")
            stat_card(c2, "📦", "등록 제품 수(합계)", f"{int(companies_df['제품수'].sum()):,} 개")

            st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
            company_query = st.text_input("기업명 검색", placeholder="예: 이마트", key="company_search")
            filtered_companies = companies_df.copy()
            if company_query:
                filtered_companies = filtered_companies[
                    filtered_companies["제조사/유통사"].astype(str).str.contains(company_query, case=False, na=False)
                ]
            st.dataframe(filtered_companies, width="stretch", height=300, hide_index=True)

            st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)
            st.subheader("제품 등록수 상위 기업")
            top_companies = companies_df.sort_values("제품수", ascending=False).head(15)
            eco_rank_grid(
                top_companies["제조사/유통사"], top_companies["제품수"],
                unit="개", top_n=15, cols=2, highlight_top=3,
            )

    elif eco_tab == _ECO_TABS[1]:
        carbon_path = os.path.join(DATA_DIR, "carbon_footprint_products.csv")
        if not os.path.exists(carbon_path):
            st.warning("data_ref/carbon_footprint_products.csv 파일이 없습니다.")
        else:
            carbon_df = pd.read_csv(carbon_path)
            carbon_query = st.text_input("제품/업체명 검색", placeholder="예: 생수", key="carbon_search")
            filtered_carbon = carbon_df.copy()
            if carbon_query:
                mask = (
                    carbon_df["제품명"].astype(str).str.contains(carbon_query, case=False, na=False)
                    | carbon_df["업체명"].astype(str).str.contains(carbon_query, case=False, na=False)
                )
                filtered_carbon = carbon_df[mask]
            st.dataframe(filtered_carbon, width="stretch", hide_index=True)

            if not filtered_carbon.empty:
                st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("탄소배출량 상위 10개 제품")
                    top10 = (
                        filtered_carbon.sort_values("탄소배출량", ascending=False)
                        .head(10)
                        .sort_values("탄소배출량")  # 가로 막대: 위로 갈수록 큰 값
                    )
                    eco_lollipop_chart(
                        top10["제품명"], top10["탄소배출량"],
                        key="footprint_top10_chart",
                        unit=" kg", value_name="탄소배출량", height="420px",
                    )

                with c2:
                    st.subheader("제품군별 평균 탄소배출량")
                    st.caption("마우스를 올리면 항목별 상세 수치가 표시됩니다.")
                    avg_group = (
                        filtered_carbon.groupby("제품군")["탄소배출량"].mean()
                        .reset_index()
                        .sort_values("탄소배출량", ascending=False)
                    )
                    eco_donut_legend_chart(
                        avg_group["제품군"], avg_group["탄소배출량"],
                        key="footprint_avg_group_chart",
                        unit=" kg CO2", value_name="평균 탄소배출량",
                        center_title="제품군별\n평균 배출량", height="380px",
                    )

            st.markdown(
                "<div style='text-align:right; color:#94a3b8; font-size:0.8rem;'>"
                "환경부 · 탄소발자국 인증 제품 참고 데이터</div>",
                unsafe_allow_html=True,
            )

elif page == "donate":
    _title_col, _points_col = st.columns([4, 1])
    with _title_col:
        st.header("💚 포인트로 기부하기")
    with _points_col:
        points_balance_card("사용 가능한 포인트", f"{st.session_state.points_balance:,} P")

    if st.session_state.points_balance <= 0:
        st.info("적립된 포인트가 없습니다.")
    else:
        cause = st.selectbox("기부처를 선택하세요", DONATION_CAUSES, key="donate_cause")
        max_points = int(st.session_state.points_balance)
        step = 10 if max_points >= 10 else 1
        amount = st.slider("기부할 포인트", min_value=0, max_value=max_points, value=0, step=step, key="donate_amount")
        if st.button("💚 기부하기", type="primary", key="donate_submit"):
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
                st.success(f"{cause}에 {amount}P를 기부했습니다.")
                st.rerun()

    if st.session_state.donations:
        st.divider()
        st.subheader("나의 기부 이력")
        donation_df = pd.DataFrame(st.session_state.donations)
        st.dataframe(donation_df, width="stretch", hide_index=True)
        st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)
        render_donation_impact(donation_df["포인트"].sum())

if page != "platform":
    release_webcam()
    st.stop()

# =========================================================
# 5. 미디어 입력 방식 선택
# =========================================================

# 🟢 [이동] 제조사/브랜드 선택 UI는 "🪙 포인트 적립 현황" 페이지로 옮겼습니다.
#    이곳에서는 그 페이지에서 고른 값을 세션 상태로 받아와 그대로 사용합니다.
selected_brand = st.session_state.get("points_brand_filter", "해당없음")
if selected_brand == "전체":
    selected_brand = "해당없음"
if selected_brand != "해당없음":
    st.caption(f"🏷️ 선택된 브랜드: **{selected_brand}** (참여기업 2배 적립 적용 · 변경은 '💰 포인트 적립 현황' 메뉴에서)")

input_type = st.radio(
    "분석할 미디어 형식을 선택하세요.",
    [
        "이미지 업로드",
        "카메라 촬영",
        "🎥 실시간 웹캠 감지"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

if input_type != "🎥 실시간 웹캠 감지":
    release_webcam()

images = []

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
            images.append((file.name, img))

elif input_type == "카메라 촬영":
    camera_file = st.camera_input("카메라로 재활용품을 촬영하세요.")
    if camera_file is not None:
        img = Image.open(camera_file)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        images.append(("카메라 촬영 이미지", img))

elif input_type == "🎥 실시간 웹캠 감지":
    release_webcam()

    st.info(
        "웹캠을 통해 실시간으로 재활용품을 탐지합니다. "
        f"신뢰도 {WEBCAM_AUTO_STOP_CONFIDENCE:.0%} 이상인 항목이 인식되면 포인트가 1회 적립되고 웹캠이 자동으로 정지됩니다. "
        "다시 탐지하려면 아래 체크박스를 다시 켜주세요."
    )
    col_cam, col_info = st.columns([2, 1])

    with col_cam:
        if "webcam_run_cam" not in st.session_state:
            st.session_state.webcam_run_cam = False
        if st.session_state.webcam_force_stop:
            st.session_state.webcam_run_cam = False
            st.session_state.webcam_force_stop = False
        run_cam = st.checkbox("🎥 웹캠 실시간 가동 시작", key="webcam_run_cam")
        frame_placeholder = st.empty()

    with col_info:
        st.subheader("실시간 판정 정보")
        status_placeholder = st.empty()
        guide_placeholder = st.empty()
        reward_placeholder = st.empty()

    if run_cam:
        if st.session_state.webcam_session_token is None:
            st.session_state.webcam_session_token = f"webcam_{int(time.time() * 1000)}"
            st.session_state.webcam_processed_track_keys = set()
            st.session_state.last_webcam_reward = None

        with reward_placeholder.container():
            st.info(f"💰 신뢰도 {WEBCAM_AUTO_STOP_CONFIDENCE:.0%} 이상인 재활용품을 인식하면 포인트가 적립되고 웹캠이 자동으로 정지됩니다.")
            st.metric("💳 현재 보유 포인트", f"{st.session_state.points_balance:,} P")

        cap = cv2.VideoCapture(0)
        st.session_state.webcam_capture = cap

        if not cap.isOpened():
            st.error("웹캠을 열 수 없습니다. 카메라 연결 상태를 확인해 주세요.")
        else:
            frame_skip = 0
            latest_detections = []

            try:
                while run_cam:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("웹캠 스트림을 가져오는 데 실패했습니다.")
                        break

                    if frame_skip % WEBCAM_FRAME_INTERVAL == 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(frame_rgb)

                        detections = detect_and_classify(pil_img, use_tracking=True)
                        detections = reanalyze_detections(detections)
                        detections = deduplicate_detections(detections)
                        latest_detections = detections

                        if detections:
                            with status_placeholder.container():
                                st.success(f"✅ {len(detections)}개 물체 탐지")
                                for detection in detections:
                                    object_id = detection["object_id"]
                                    top_label = detection["label"]
                                    top_score = detection["score"]

                                    if detection.get("out_of_scope"):
                                        st.markdown(
                                            f"**OBJ {object_id}** → "
                                            f"<span style='color:#dc2626; font-weight:700;'>{OUT_OF_SCOPE_WARNING}</span>",
                                            unsafe_allow_html=True,
                                        )
                                    elif not detection["needs_review"]:
                                        st.write(f"**OBJ {object_id}** → {top_label} ({top_score:.1%})")
                                    else:
                                        st.warning(f"OBJ {object_id} → {top_label} ({top_score:.1%}) → 자동 재분석 실패 · 재투입 필요")

                            with guide_placeholder.container():
                                st.markdown("#### ♻️ 분리배출 안내")
                                for detection in detections:
                                    top_label = detection["label"]
                                    if detection.get("out_of_scope"):
                                        st.markdown(
                                            f"<span style='color:#dc2626; font-weight:700;'>{OUT_OF_SCOPE_WARNING}</span>",
                                            unsafe_allow_html=True,
                                        )
                                    elif not detection["needs_review"]:
                                        guide_text = RECYCLE_GUIDE.get(top_label, "정보 없음")
                                        st.info(f"**{top_label}**\n\n{guide_text}")
                                    else:
                                        st.warning("⚠️ 신뢰도 부족 → 재투입 라인")

                            current_time = time.time()

                            # 🟢 신뢰도가 WEBCAM_AUTO_STOP_CONFIDENCE 이상인 탐지가 하나라도 나오면 웹캠을 자동 정지하고,
                            #    이때 화면에 잡힌 물체 중 "정상 분류"(재투입 필요 아님)된 것은 모두 포인트 적립한다.
                            high_conf_detection = next(
                                (d for d in detections
                                 if not d["needs_review"] and not d.get("out_of_scope") and d["score"] >= WEBCAM_AUTO_STOP_CONFIDENCE),
                                None,
                            )

                            def _webcam_key(d):
                                # 🟢 크레딧 시점과 목록 조회 시점에 항상 같은 키가 나오도록 하나로 통일한다.
                                #    (fallback 키를 조회 쪽에서 다시 만들지 않아 0P로 보이던 버그 수정)
                                d_track_id = d.get("track_id")
                                d_model_name = d.get("model_name", "m")
                                if d_track_id is not None:
                                    return f"{st.session_state.webcam_session_token}_track_{d_model_name}_{d_track_id}"
                                return f"{st.session_state.webcam_session_token}_fallback_{int(current_time)}_{d['object_id']}_{d_model_name}_{d['label']}"

                            if high_conf_detection is not None:
                                top_score = high_conf_detection["score"]
                                credited_pairs = []  # (detection, reward_record) — 사진과 함께 보여주기 위해 detection도 보관
                                # 🟢 이번 자동정지 1건만의 팝업이 되도록, 크레딧 시작 전에 큐를 비운다(이전 탐지와 누적 방지).
                                start_point_popup_batch()

                                for d in detections:
                                    if d["needs_review"] or d.get("out_of_scope"):
                                        continue

                                    d_label = d["label"]
                                    d_track_key = _webcam_key(d)

                                    if d_track_key not in st.session_state.webcam_processed_track_keys:
                                        st.session_state.history.append({
                                            "파일명": "실시간 웹캠",
                                            "객체 번호": d["object_id"],
                                            "분류 결과": d_label,
                                            "신뢰도": d["score"],
                                            "입력 방식": "실시간 웹캠",
                                            "판정 상태": "정상 분류",
                                        })
                                        st.session_state.webcam_processed_track_keys.add(d_track_key)

                                        reward_record = credit_detection_once(
                                            d_track_key,
                                            d_label,
                                            selected_brand,
                                        )
                                        if reward_record is not None:
                                            credited_pairs.append((d, reward_record))
                                            bonus_text = " · 참여기업 2배" if reward_record["2배적용"] else ""
                                            st.toast(
                                                f"{d_label} +{reward_record['포인트']}P 적립{bonus_text}",
                                                icon="✅",
                                            )
                                            # 🟢 포인트 적립이 될 때 무조건 모달 팝업이 뜨도록 큐에 등록
                                            # (reward_placeholder는 곧바로 st.rerun()으로 초기화되어 사라지므로 신뢰할 수 없음).
                                            queue_point_popup(reward_record)

                                if credited_pairs:
                                    st.session_state.last_webcam_reward = credited_pairs[-1][1]

                                with reward_placeholder.container():
                                    st.success(f"🛑 신뢰도 {top_score:.0%} 도달 → 웹캠 자동 정지")
                                    if not credited_pairs:
                                        st.info("💰 포인트 대상 품목이 아닙니다.")
                                    st.metric("💳 현재 보유 포인트", f"{st.session_state.points_balance:,} P")

                                stop_batch = []
                                for d in detections:
                                    if d["needs_review"] or d.get("out_of_scope"):
                                        continue
                                    d_credited = st.session_state.credited_items.get(_webcam_key(d))
                                    stop_batch.append({
                                        "품목": d["label"],
                                        "신뢰도": d["score"],
                                        "포인트": d_credited["포인트"] if d_credited else 0,
                                        "상태": "정상 분류",
                                    })
                                st.session_state.last_batch_results = stop_batch

                                snap_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                snap_pil = Image.fromarray(snap_rgb)
                                st.session_state.webcam_snapshot = draw_detections(snap_pil, detections)

                                st.session_state.last_webcam_log = current_time
                                st.session_state.webcam_force_stop = True
                                release_webcam()
                                st.rerun()

                            st.session_state.last_webcam_log = current_time

                        else:
                            with status_placeholder.container():
                                st.warning("탐지된 재활용품이 없습니다.")

                    if latest_detections:
                        display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        display_pil = Image.fromarray(display_rgb)
                        annotated = draw_detections(display_pil, latest_detections)
                        frame_placeholder.image(annotated, channels="RGB", width="stretch")
                    else:
                        frame_placeholder.image(frame, channels="BGR", width="stretch")

                    frame_skip += 1

            finally:
                release_webcam()

# =========================================================
# 6 & 7. 이미지 / 카메라 분석 결과
# =========================================================
if images:
    st.subheader(f"🔍 AI 객체 탐지 및 분류 결과 (총 {len(images)}개 이미지/프레임)")

    st.session_state.webcam_snapshot = None
    current_batch_results = []
    # 🟢 이번 업로드/촬영 1건만의 팝업이 되도록, 처리 시작 전에 큐를 비운다(이전 탐지와 누적 방지).
    start_point_popup_batch()

    for file_name, img in images:
        image_key = f"{input_type}_{file_name}_{hash(img.tobytes())}"

        with st.spinner(f"'{file_name}' YOLO 객체 탐지 + SigLIP2 분석 중..."):
            detections = detect_and_classify(img, use_tracking=False)

        if not detections:
            st.warning("⚠️ YOLO-World가 개별 물체를 찾지 못했습니다. 기존 SigLIP2로 전체 이미지를 분석합니다.")
            detections = [fallback_classification(img)]

        detections = reanalyze_detections(detections)
        detections = deduplicate_detections(detections)

        bbox_detections = [d for d in detections if d["bbox"] is not None]
        if bbox_detections:
            annotated = draw_detections(img, bbox_detections)
        else:
            annotated = np.array(img)

        col_img, col_summary = st.columns([1.3, 1])

        with col_img:
            st.image(annotated, caption=file_name, width="stretch")

        with col_summary:
            st.metric("탐지된 객체", f"{len(detections)}개")
            out_of_scope_count = sum(1 for d in detections if d.get("out_of_scope"))
            normal_count = sum(1 for d in detections if not d["needs_review"] and not d.get("out_of_scope"))
            reject_count = len(detections) - normal_count - out_of_scope_count

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("정상 분류", f"{normal_count}개")
            with col_b:
                st.metric("재투입 필요", f"{reject_count}개")
            with col_c:
                st.metric("비대상 품목", f"{out_of_scope_count}개")

        st.markdown("### 📦 객체별 분류 결과")

        for detection in detections:
            object_id = detection["object_id"]
            top_label = detection["label"]
            top_score = detection["score"]
            yolo_label = detection["yolo_label"]
            yolo_score = detection["yolo_score"]

            col_crop, col_res = st.columns([1, 2])

            with col_crop:
                st.image(detection["crop"], caption=f"OBJ {object_id}", width="stretch")

            with col_res:
                if detection.get("out_of_scope"):
                    # 🟢 플라스틱/캔/종이팩 외 품목: 실제 판정 결과는 노출하지 않고 통일된 경고만 표시.
                    st.markdown(f"#### OBJ {object_id}")
                    st.markdown(
                        f"<span style='color:#dc2626; font-weight:700;'>{OUT_OF_SCOPE_WARNING}</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f"#### OBJ {object_id} → {top_label}")

                    if detection["bbox"] is not None:
                        model_name = detection.get("model_name", "best")
                        st.caption(f"YOLO 탐지 [{model_name}]: {yolo_label} ({yolo_score:.1%})")
                    else:
                        st.caption("YOLO 탐지 없음 → SigLIP2 전체 이미지 분석")

                    if detection.get("reanalysis_count", 0) > 0:
                        if detection.get("requires_reentry", False):
                            st.caption("🔄 자동 재분석 1회 수행 → 재투입 필요")
                        else:
                            st.caption("🔄 자동 재분석 1회 수행 → 판정 성공")

                    if detection["needs_review"]:
                        st.error("⚠️ **[경고] 자동 재분석 후에도 판정이 불확실합니다.**")
                        st.warning("물품 판정이 불확실하여 자동 재분석 후에도 판정이 어려워 **'재투입 라인'**으로 이동합니다.")
                    else:
                        guide_text = RECYCLE_GUIDE.get(top_label, "지침 정보 없음")
                        st.success(f"💡 **분리배출 지침:** {guide_text}")

                    for label, score in detection["all_results"][:3]:
                        st.write(f"**{label}**: {score:.2%}")
                        st.progress(float(score))

            object_key = f"{image_key}_{object_id}"
            # 🟢 비대상 품목 제외, "정상 분류"(재투입 필요 아님)된 물체는 모두 포인트 적립/목록 대상.
            qualifies_for_points = (
                not detection.get("out_of_scope")
                and not detection["needs_review"]
            )

            if object_key not in st.session_state.processed_keys:
                if detection.get("out_of_scope"):
                    status_text = "비대상 품목"
                elif detection["needs_review"]:
                    status_text = "재투입 필요"
                else:
                    status_text = "정상 분류"

                st.session_state.history.append({
                    "파일명": file_name,
                    "객체 번호": object_id,
                    "분류 결과": top_label,
                    "신뢰도": top_score,
                    "입력 방식": input_type,
                    "판정 상태": status_text
                })
                st.session_state.processed_keys.add(object_key)

                if qualifies_for_points:
                    reward_record = credit_detection_once(
                        object_key,
                        top_label,
                        selected_brand,
                    )
                    if reward_record is not None:
                        bonus_text = " · 참여기업 2배" if reward_record["2배적용"] else ""
                        st.toast(
                            f"{top_label} +{reward_record['포인트']}P 적립{bonus_text}",
                            icon="✅",
                        )
                        show_credit_success(reward_record)
                        # 🟢 포인트 적립이 될 때 무조건 모달 팝업이 뜨도록 큐에 등록
                        queue_point_popup(reward_record)

            if qualifies_for_points:
                credited = st.session_state.credited_items.get(object_key)
                current_batch_results.append({
                    "품목": top_label,
                    "신뢰도": top_score,
                    "포인트": credited["포인트"] if credited else 0,
                    "상태": "정상 분류",
                })

            st.divider()

    st.session_state.last_batch_results = current_batch_results

elif input_type != "🎥 실시간 웹캠 감지":
    st.info("💡 분석할 이미지를 업로드하거나 카메라로 촬영해 주세요.")

# =========================================================
# 8. 대시보드
# =========================================================
st.divider()
st.markdown(
    "<h3 style='margin:0 0 0.5rem;'>📊 재활용품 분석 대시보드</h3>",
    unsafe_allow_html=True,
)

if len(st.session_state.last_batch_results) > 0:
    batch_df = pd.DataFrame(st.session_state.last_batch_results)

    display_df = batch_df.copy()
    display_df["신뢰도"] = (display_df["신뢰도"] * 100).round(1).astype(str) + "%"
    display_df["포인트"] = display_df["포인트"].map(lambda p: f"{p:,} P")
    display_df = display_df.rename(columns={"품목": "재활용품 품목"})

    st.subheader("📦 이번 투입 목록")

    if st.session_state.webcam_snapshot is not None:
        col_snap, col_list = st.columns([1, 1.3])
        with col_snap:
            st.image(st.session_state.webcam_snapshot, caption="📸 탐지 순간", width="stretch")
        with col_list:
            st.dataframe(
                display_df[["재활용품 품목", "신뢰도", "포인트", "상태"]],
                width="stretch", hide_index=True,
            )
    else:
        st.dataframe(
            display_df[["재활용품 품목", "신뢰도", "포인트", "상태"]],
            width="stretch", hide_index=True,
        )

    total_batch_points = int(batch_df["포인트"].sum())
    st.metric("💰 총 포인트", f"{total_batch_points:,} P")

    st.divider()
    if st.button("분석 기록 초기화"):
        st.session_state.history = []
        st.session_state.last_batch_results = []
        st.session_state.webcam_snapshot = None
        st.session_state.processed_keys = set()
        st.session_state.last_webcam_log = 0.0
        st.session_state.webcam_session_token = None
        st.session_state.webcam_processed_track_keys = set()
        st.session_state.last_webcam_reward = None
        release_webcam()
        st.rerun()

else:
    st.info("아직 분석 기록이 없습니다. 미디어를 분석하거나 웹캠을 실행하면 대시보드에 결과가 표시됩니다.")

# =========================================================
# 9. 포인트 적립 팝업 (무조건 표시)
# 🟢 웹캠 플로우는 크레딧 직후 st.rerun()으로 화면을 새로 그리므로, 그 전에 띄운 내용은
#    사라진다. 그래서 크레딧 시점엔 큐에만 쌓아두고, 매 스크립트 실행의 맨 끝에서
#    큐에 남은 팝업을 모달로 띄운다 (업로드 흐름은 같은 실행 내에서, 웹캠은 재실행 직후에 뜬다).
# =========================================================
if st.session_state.get("pending_point_popups"):
    render_point_popup()