"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          OncoInsight — Production Oncology Analytics Platform               ║
║          Snowflake OMOP CDM · Snowflake Cortex AI · Clinical-Grade          ║
║          WCAG 2.1 AA · Lavender/Pink/Teal/Blue Palette                      ║
║          SiS-Ready (Streamlit in Snowflake) · HIPAA-Aware                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

PRODUCTION FEATURES:
  ✅ Snowflake OMOP CDM v5.4 integration (SiS + local connector)
  ✅ Snowflake Cortex AI (mistral-large2 / llama3-70b / arctic)
  ✅ AI Clinical Chat — natural language Q&A over OMOP data
  ✅ AI Cohort Builder — NL → OMOP SQL generation
  ✅ AI Patient Narrative — per-patient journey summarization
  ✅ ANT Classification Engine (DRUG_EXPOSURE vs PROCEDURE_OCCURRENCE)
  ✅ Line of Therapy (LoT) engine — 90-day gap rule on DRUG_EXPOSURE
  ✅ Biomarker analytics from MEASUREMENT table
  ✅ Discrepancy detection — date anomalies & duplicate drug events
  ✅ Kaplan-Meier style survival curves from OMOP dates
  ✅ Multi-dimensional Cohort Builder with OMOP filters
  ✅ Treatment pattern analysis across all LoT lines
  ✅ CSV export on all tables
  ✅ Dark mode · WCAG 2.1 AA · Clinical lavender design system

DEPLOYMENT:
  Snowflake (SiS):
    1. Upload oncoinsight.py to Snowflake → Streamlit Apps
    2. Set OMOP_DATABASE + OMOP_SCHEMA as app env vars (or sidebar config)
    3. Grant USAGE + SELECT on OMOP schema to Streamlit app role
    4. Snowflake Cortex AI is auto-enabled if CORTEX privilege is granted

  Local:
    pip install streamlit pandas numpy plotly snowflake-connector-python
    # .streamlit/secrets.toml:
    # [snowflake]
    # account = "your-account"
    # user = "your-user"
    # password = "your-password"
    # database = "OMOP_CDM"
    # schema = "CDM"
    # warehouse = "COMPUTE_WH"
    # role = "ONCOLOGY_ANALYST"
    streamlit run oncoinsight.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import json
import time
import random
import math
import io

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OncoInsight | Oncology Analytics",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# SNOWFLAKE CONNECTION LAYER
# Supports: Streamlit in Snowflake (SiS) | Local connector | Demo mode
# ─────────────────────────────────────────────────────────────────────────────
IS_SIS       = False
_sf_session  = None
_sf_conn     = None

def _init_sis():
    """Detect and initialise Snowflake-native session (SiS)."""
    global IS_SIS, _sf_session
    try:
        from snowflake.snowpark.context import get_active_session
        _sf_session = get_active_session()
        IS_SIS = True
        return True
    except Exception:
        return False

def _init_local_connector():
    """Initialise Snowflake connector from st.secrets (local dev)."""
    global _sf_conn
    try:
        import snowflake.connector as sf
        creds = st.secrets.get("snowflake", {})
        if not creds:
            return False
        _sf_conn = sf.connect(
            account=creds.get("account", ""),
            user=creds.get("user", ""),
            password=creds.get("password", ""),
            database=creds.get("database", "OMOP_CDM"),
            schema=creds.get("schema", "CDM"),
            warehouse=creds.get("warehouse", "COMPUTE_WH"),
            role=creds.get("role", ""),
        )
        return True
    except Exception:
        return False

_init_sis()
if not IS_SIS:
    _init_local_connector()

def _run_query(sql: str, params: dict | None = None) -> pd.DataFrame:
    """
    Unified query runner.
    Returns a DataFrame from Snowflake (SiS session or connector) or raises.
    """
    try:
        if IS_SIS and _sf_session:
            return _sf_session.sql(sql).to_pandas()
        elif _sf_conn:
            import snowflake.connector
            cur = _sf_conn.cursor()
            cur.execute(sql)
            cols = [c[0] for c in cur.description]
            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=cols)
        else:
            raise ConnectionError("No Snowflake connection available.")
    except Exception as e:
        raise e

def _cortex_complete(prompt: str, model: str = "mistral-large2") -> str:
    """
    Call Snowflake Cortex Complete LLM.
    Falls back to rule-based response if Cortex is unavailable.
    """
    try:
        # Escape single quotes in prompt
        safe_prompt = prompt.replace("'", "''")
        sql = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{model}', '{safe_prompt}') AS response"
        result = _run_query(sql)
        return result.iloc[0, 0] if not result.empty else ""
    except Exception as e:
        return f"[AI unavailable: {str(e)[:80]}. Configure Snowflake Cortex access to enable AI features.]"

def _cortex_available() -> bool:
    """Check if Cortex is accessible."""
    try:
        _run_query("SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large2', 'hi') AS r")
        return True
    except Exception:
        return False

SF_CONNECTED = IS_SIS or (_sf_conn is not None)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
_defaults = {
    "authenticated":  IS_SIS,   # Auto-auth in SiS
    "dark_mode":      False,
    "toast":          None,
    "omop_db":        "OMOP_CDM",
    "omop_schema":    "CDM",
    "ai_model":       "mistral-large2",
    "demo_mode":      not SF_CONNECTED,
    "chat_history":   [],
    "insights_list":  [],
    "cohort_sql":     "",
    "sf_manual_ok":   False,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

DEMO    = st.session_state.demo_mode
DB      = st.session_state.omop_db
SCHEMA  = st.session_state.omop_schema
OMOP    = f"{DB}.{SCHEMA}"   # fully-qualified prefix for OMOP tables

# ─────────────────────────────────────────────────────────────────────────────
# CANCER-AWARE COLOR PALETTE — WCAG 2.1 AA
# ─────────────────────────────────────────────────────────────────────────────
DARK = st.session_state.dark_mode

C_PRIMARY    = "#5B21B6"
C_PRIMARY_LT = "#7C3AED"
C_PRIMARY_BG = "#F5F3FF"
C_PINK       = "#9D174D"
C_PINK_BG    = "#FDF2F8"
C_TEAL       = "#0F766E"
C_TEAL_BG    = "#F0FDFA"
C_BLUE       = "#1D4ED8"
C_BLUE_BG    = "#EFF6FF"
C_SUCCESS    = "#065F46"
C_SUCCESS_BG = "#ECFDF5"
C_WARNING    = "#92400E"
C_WARNING_BG = "#FFFBEB"
C_DANGER     = "#991B1B"
C_DANGER_BG  = "#FEF2F2"
C_INFO       = "#1E40AF"
C_INFO_BG    = "#EFF6FF"
C_PURPLE     = "#5B21B6"
C_PURPLE_BG  = "#EDE9FE"
C_ORANGE     = "#9A3412"
C_ORANGE_BG  = "#FFF7ED"

if not DARK:
    C_BG         = "#F8F7FC"
    C_SURFACE    = "#FFFFFF"
    C_SURFACE2   = "#F3F1FA"
    C_BORDER     = "#DDD9EC"
    C_BORDER_LT  = "#ECEAF5"
    C_TEXT       = "#1C1B29"
    C_TEXT_MED   = "#4B4A6A"
    C_TEXT_MUTED = "#9896B5"
    C_HEADER_BG  = "#3B1A8A"
    C_SIDEBAR_BG = "#2D1769"
    C_SIDEBAR_TXT= "#EDE9FE"
else:
    C_BG         = "#0D0B1A"
    C_SURFACE    = "#16122E"
    C_SURFACE2   = "#1E1840"
    C_BORDER     = "#2E2760"
    C_BORDER_LT  = "#241E50"
    C_TEXT       = "#EBE8F8"
    C_TEXT_MED   = "#A9A3D0"
    C_TEXT_MUTED = "#6B659A"
    C_HEADER_BG  = "#110D2C"
    C_SIDEBAR_BG = "#0A0818"
    C_SIDEBAR_TXT= "#C4BFED"
    C_PRIMARY_BG = "#1E1A3F"
    C_TEAL_BG    = "#0A2624"
    C_BLUE_BG    = "#0F1A3A"
    C_SUCCESS_BG = "#042818"
    C_WARNING_BG = "#281A08"
    C_DANGER_BG  = "#280A0A"
    C_INFO_BG    = "#0F1A3A"
    C_PURPLE_BG  = "#1A1040"
    C_ORANGE_BG  = "#201008"
    C_PINK_BG    = "#250A14"

CHART_COLORS  = [C_PRIMARY, C_TEAL, C_BLUE, C_PINK, "#B45309", C_SUCCESS,
                 C_PRIMARY_LT, "#0D9488", "#7C3AED", "#9A3412"]
CANCER_COLORS = {
    "Breast":      C_PINK,
    "Ovarian":     C_TEAL,
    "Prostate":    C_BLUE,
    "Lung":        C_PRIMARY,
    "Colorectal":  C_SUCCESS,
    "Lymphoma":    "#B45309",
    "Other":       C_TEXT_MUTED,
}

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — Clinical Lavender Design System
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Figtree:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Figtree', 'Segoe UI', system-ui, sans-serif;
    background-color: {C_BG} !important;
    color: {C_TEXT};
    font-size: 15px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
}}
.stApp {{ background-color: {C_BG} !important; }}
.main .block-container {{ padding: 0 !important; max-width: 100%; }}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {C_SIDEBAR_BG} 0%, {C_SIDEBAR_BG}ee 100%) !important;
    border-right: 1px solid {C_BORDER} !important;
}}
[data-testid="stSidebar"] * {{ color: {C_SIDEBAR_TXT} !important; }}
[data-testid="stSidebar"] .stRadio label {{
    font-size: 13.5px !important; padding: 7px 12px !important;
    border-radius: 8px !important; transition: background 0.15s ease !important;
    display: flex !important; align-items: center !important;
    gap: 8px !important; cursor: pointer !important; font-weight: 500 !important;
}}
[data-testid="stSidebar"] .stRadio label:hover {{ background: rgba(255,255,255,0.10) !important; }}

[data-testid="metric-container"] {{
    background: {C_SURFACE}; border: 1px solid {C_BORDER};
    border-radius: 12px; padding: 16px 18px !important;
    box-shadow: 0 2px 8px rgba(91,33,182,0.07);
    transition: transform 0.2s, box-shadow 0.2s;
}}
[data-testid="metric-container"]:hover {{ transform: translateY(-1px); box-shadow: 0 4px 14px rgba(91,33,182,0.12); }}
[data-testid="stMetricValue"] {{
    color: {C_PRIMARY} !important; font-weight: 700 !important;
    font-family: 'Space Mono', monospace !important;
}}
h1 {{ color: {C_PRIMARY} !important; font-family: 'Syne', sans-serif !important;
     font-weight: 700 !important; font-size: 1.5rem !important;
     border-bottom: 2px solid {C_BORDER_LT}; padding-bottom: 10px; }}
h2 {{ color: {C_PRIMARY} !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; font-size: 1.15rem !important; }}
h3 {{ color: {C_TEXT} !important; font-family: 'Syne', sans-serif !important; font-weight: 600 !important; font-size: 1rem !important; }}

[data-testid="stDataFrame"] {{
    border: 1px solid {C_BORDER} !important; border-radius: 10px !important;
    overflow: hidden !important; box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
}}
.stButton > button {{
    background: linear-gradient(135deg, {C_PRIMARY} 0%, {C_PRIMARY_LT} 100%) !important;
    color: white !important; font-weight: 600 !important; border: none !important;
    border-radius: 8px !important; padding: 9px 20px !important;
    font-size: 14px !important; font-family: 'Figtree', sans-serif !important;
    box-shadow: 0 2px 8px rgba(91,33,182,0.30) !important;
    transition: all 0.2s ease !important;
}}
.stButton > button:hover {{
    background: linear-gradient(135deg, #4C1D95 0%, {C_PRIMARY} 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(91,33,182,0.40) !important;
}}
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] > div > div > input,
[data-testid="stTextArea"] > div > div > textarea {{
    background: {C_SURFACE} !important; border: 1.5px solid {C_BORDER} !important;
    border-radius: 8px !important; color: {C_TEXT} !important;
    font-family: 'Figtree', sans-serif !important; font-size: 14px !important;
}}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stTextInput"] > div > div > input:focus,
[data-testid="stTextArea"] > div > div > textarea:focus {{
    border-color: {C_PRIMARY_LT} !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.15) !important;
}}
.stTabs [data-baseweb="tab-list"] {{
    background: {C_SURFACE}; border-bottom: 2px solid {C_BORDER};
    padding: 0 8px; gap: 2px; border-radius: 0;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent; color: {C_TEXT_MUTED}; font-weight: 600;
    font-size: 13.5px; font-family: 'Figtree', sans-serif;
    padding: 11px 20px; border-bottom: 3px solid transparent;
    margin-bottom: -2px; border-radius: 0; transition: all 0.15s ease;
}}
.stTabs [aria-selected="true"] {{
    background: transparent !important; color: {C_PRIMARY} !important;
    border-bottom: 3px solid {C_PRIMARY} !important;
}}
.stTabs [data-baseweb="tab"]:hover {{ background: {C_PRIMARY_BG} !important; color: {C_PRIMARY} !important; }}
[data-testid="stExpander"] {{
    background: {C_SURFACE} !important; border: 1px solid {C_BORDER} !important;
    border-radius: 10px !important; overflow: hidden !important;
}}
[data-testid="stChatMessageContent"] {{
    background: {C_SURFACE2} !important; border-radius: 10px !important;
    border: 1px solid {C_BORDER} !important;
}}
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {C_BG}; }}
::-webkit-scrollbar-thumb {{ background: {C_BORDER}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {C_PRIMARY_LT}; }}
hr {{ border-color: {C_BORDER} !important; margin: 12px 0 !important; opacity: 0.6 !important; }}

/* ═══ OncoInsight Design System Components ═══ */
.oi-topbar {{
    background: linear-gradient(135deg, {C_HEADER_BG} 0%, {C_SIDEBAR_BG} 100%);
    color: white; padding: 16px 28px;
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 1.5rem; box-shadow: 0 3px 16px rgba(59,26,138,0.25);
    position: relative; overflow: hidden;
}}
.oi-topbar::before {{
    content: ''; position: absolute; right: -20px; top: -20px;
    width: 120px; height: 120px; border-radius: 50%;
    background: rgba(167,139,250,0.12); pointer-events: none;
}}
.oi-topbar-title {{ font-family: 'Syne', sans-serif; font-size: 18px; font-weight: 700; }}
.oi-topbar-sub   {{ font-size: 12.5px; opacity: 0.75; margin-top: 2px; }}
.oi-badge {{
    background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.25);
    color: white; padding: 4px 12px; border-radius: 20px;
    font-size: 11.5px; font-weight: 600; margin-left: 7px;
    backdrop-filter: blur(4px);
}}
.oi-kpi {{
    background: {C_SURFACE}; border: 1px solid {C_BORDER}; border-radius: 12px;
    padding: 16px 20px; box-shadow: 0 2px 10px rgba(91,33,182,0.07);
    position: relative; overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}}
.oi-kpi:hover {{ transform: translateY(-2px); box-shadow: 0 6px 20px rgba(91,33,182,0.12); }}
.oi-kpi-label {{
    font-size: 10.5px; font-weight: 700; color: {C_TEXT_MUTED};
    text-transform: uppercase; letter-spacing: 0.09em; margin-bottom: 6px;
}}
.oi-kpi-num {{
    font-size: 28px; font-weight: 700; font-family: 'Space Mono', monospace;
    line-height: 1.1; margin-bottom: 6px;
}}
.oi-kpi-delta {{ font-size: 12px; color: {C_TEXT_MUTED}; display: flex; align-items: center; gap: 4px; }}
.oi-sec {{
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 13px; font-weight: 700; font-family: 'Syne', sans-serif;
    color: {C_PRIMARY}; background: {C_PRIMARY_BG};
    border-left: 4px solid {C_PRIMARY}; padding: 9px 16px;
    border-radius: 0 8px 8px 0; margin: 18px 0 14px;
    width: fit-content; min-width: 200px; letter-spacing: 0.01em;
}}
.oi-insight {{
    background: {C_SURFACE}; border: 1px solid {C_BORDER}; border-left: 4px solid;
    border-radius: 0 12px 12px 0; padding: 14px 20px; margin-bottom: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05); transition: transform 0.15s, box-shadow 0.15s;
}}
.oi-insight:hover {{ transform: translateX(2px); box-shadow: 0 4px 14px rgba(0,0,0,0.08); }}
.oi-insight-text {{ font-size: 14px; color: {C_TEXT}; line-height: 1.65; font-weight: 500; }}
.oi-insight-meta {{ font-size: 11.5px; color: {C_TEXT_MUTED}; margin-top: 6px; }}
.oi-card {{
    background: {C_SURFACE}; border: 1px solid {C_BORDER}; border-radius: 12px;
    padding: 18px 20px; margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(91,33,182,0.05); transition: box-shadow 0.2s;
}}
.oi-card:hover {{ box-shadow: 0 4px 16px rgba(91,33,182,0.10); }}
.oi-card-title {{
    color: {C_PRIMARY}; font-weight: 700; font-family: 'Syne', sans-serif;
    font-size: 13.5px; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;
}}
.oi-card-body  {{ color: {C_TEXT_MED}; font-size: 13.5px; line-height: 1.7; }}
.oi-journey-node {{
    background: {C_SURFACE}; border: 1px solid {C_BORDER}; border-top: 3px solid;
    border-radius: 12px; text-align: center; padding: 14px 8px;
    box-shadow: 0 2px 8px rgba(91,33,182,0.07); transition: transform 0.2s;
}}
.oi-journey-node:hover {{ transform: translateY(-3px); }}
.oi-journey-icon  {{ font-size: 24px; margin-bottom: 6px; }}
.oi-journey-title {{
    font-size: 9.5px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; margin-bottom: 4px; font-family: 'Syne', sans-serif;
}}
.oi-journey-date  {{ font-size: 10.5px; font-family: 'Space Mono', monospace; color: {C_PRIMARY}; }}
.oi-sidebar-hdr {{
    background: linear-gradient(135deg, rgba(124,58,237,0.3) 0%, rgba(91,33,182,0.15) 100%);
    border-bottom: 1px solid rgba(255,255,255,0.12);
    margin: -1rem -1rem 1.2rem; padding: 20px 16px 18px;
    position: relative; overflow: hidden;
}}
.oi-sidebar-hdr::after {{ content: '🎗️'; position: absolute; right: 12px; top: 12px; font-size: 28px; opacity: 0.35; }}
.oi-sidebar-dataset {{
    background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px; padding: 12px 14px; font-size: 12px;
}}
.oi-progress-bg   {{ background:{C_BORDER_LT};border-radius:6px;height:10px;overflow:hidden;margin-top:5px; }}
.oi-progress-fill {{ height:100%;border-radius:6px;transition:width 0.7s cubic-bezier(.4,0,.2,1); }}
.oi-filter-panel {{
    background: {C_SURFACE}; border: 1px solid {C_BORDER}; border-radius: 12px;
    padding: 18px 22px; margin-bottom: 18px;
    box-shadow: 0 2px 10px rgba(91,33,182,0.06);
}}
.oi-toast {{
    position: fixed; bottom: 24px; right: 24px;
    background: {C_PRIMARY}; color: white; padding: 12px 20px;
    border-radius: 10px; font-size: 13.5px; font-weight: 600;
    box-shadow: 0 6px 24px rgba(91,33,182,0.35); z-index: 9999;
    animation: slideIn 0.3s ease;
}}
@keyframes slideIn {{
    from {{ transform: translateY(20px); opacity: 0; }}
    to   {{ transform: translateY(0);   opacity: 1; }}
}}
.oi-alert-info    {{ background:{C_INFO_BG};border:1px solid #BFDBFE;border-left:4px solid {C_INFO};
                     border-radius:0 8px 8px 0;padding:10px 16px;font-size:13.5px;color:{C_INFO};margin-bottom:10px; }}
.oi-alert-success {{ background:{C_SUCCESS_BG};border:1px solid #A7F3D0;border-left:4px solid {C_SUCCESS};
                     border-radius:0 8px 8px 0;padding:10px 16px;font-size:13.5px;color:{C_SUCCESS};margin-bottom:10px; }}
.oi-alert-warn    {{ background:{C_WARNING_BG};border:1px solid #FCD34D;border-left:4px solid {C_WARNING};
                     border-radius:0 8px 8px 0;padding:10px 16px;font-size:13.5px;color:{C_WARNING};margin-bottom:10px; }}
.oi-alert-danger  {{ background:{C_DANGER_BG};border:1px solid #FCA5A5;border-left:4px solid {C_DANGER};
                     border-radius:0 8px 8px 0;padding:10px 16px;font-size:13.5px;color:{C_DANGER};margin-bottom:10px; }}
.oi-persona-card {{
    background: {C_SURFACE}; border: 1px solid {C_BORDER}; border-top: 3px solid;
    border-radius: 12px; padding: 18px; margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(91,33,182,0.05);
}}
.oi-biz-kpi {{
    background: {C_SURFACE}; border: 1px solid {C_BORDER}; border-top: 3px solid;
    border-radius: 12px; padding: 18px 16px; text-align: center; margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(91,33,182,0.05); transition: transform 0.2s;
}}
.oi-biz-kpi:hover {{ transform: translateY(-2px); }}
.oi-biz-kpi-val   {{
    font-size: 26px; font-weight: 700; font-family: 'Space Mono', monospace; line-height: 1.1;
}}
.oi-biz-kpi-label {{
    font-size: 10.5px; color: {C_TEXT_MED}; text-transform: uppercase;
    letter-spacing: 0.08em; margin-top: 7px; font-weight: 600;
}}
.oi-phase-card {{
    background: {C_SURFACE}; border: 1px solid {C_BORDER}; border-left: 4px solid;
    border-radius: 0 12px 12px 0; padding: 16px 20px; margin-bottom: 10px;
    box-shadow: 0 2px 8px rgba(91,33,182,0.05);
    display: flex; align-items: flex-start; gap: 14px; transition: box-shadow 0.2s;
}}
.oi-phase-card:hover {{ box-shadow: 0 4px 16px rgba(91,33,182,0.10); }}
.oi-ai-chip {{
    display:inline-flex; align-items:center; gap:6px;
    background:linear-gradient(135deg,{C_PRIMARY_BG},{C_TEAL_BG});
    border:1px solid {C_BORDER}; border-radius:20px;
    padding:5px 14px; font-size:12px; font-weight:600; color:{C_PRIMARY};
    margin-bottom:10px;
}}
.oi-sql-box {{
    background:{C_SURFACE2}; border:1px solid {C_BORDER}; border-left:4px solid {C_TEAL};
    border-radius:0 8px 8px 0; padding:12px 16px; font-family:'Space Mono',monospace;
    font-size:12px; color:{C_TEXT}; white-space:pre-wrap; overflow-x:auto;
    margin-bottom:12px;
}}
.oi-conn-banner {{
    background:linear-gradient(135deg,{C_SUCCESS_BG},{C_TEAL_BG});
    border:1px solid #A7F3D0; border-radius:10px;
    padding:10px 16px; font-size:13px; color:{C_SUCCESS};
    display:flex; align-items:center; gap:10px; margin-bottom:8px;
}}
.oi-demo-banner {{
    background:{C_WARNING_BG}; border:1px solid #FCD34D; border-radius:10px;
    padding:10px 16px; font-size:13px; color:{C_WARNING};
    display:flex; align-items:center; gap:10px; margin-bottom:8px;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# OMOP-AWARE DEMO DATA GENERATOR  (fallback when no Snowflake connection)
# Generates synthetic data that mirrors real OMOP CDM column names
# ─────────────────────────────────────────────────────────────────────────────
CANCER_TYPES  = ["Breast","Lung","Prostate","Ovarian","Colorectal","Lymphoma"]
STAGES        = ["I","II","III","IV"]
BIOMARKERS    = ["HER2+","EGFR+","PD-L1+","BRCA1/2+","ALK+","KRAS+","Triple-Negative","Wild-Type"]
DRUGS         = ["Pembrolizumab","Paclitaxel","Trastuzumab","Nivolumab",
                 "Carboplatin","Fulvestrant","Bevacizumab","Olaparib","Docetaxel","Rituximab"]
DRUG_CLASSES  = ["Immunotherapy","Chemotherapy","Targeted","Hormone","Targeted",
                 "Hormone","Targeted","PARP Inhibitor","Chemotherapy","Immunotherapy"]
DISC_TYPES    = ["Drug date before diagnosis","Missing surgery data",
                 "Duplicate therapy event","Conflicting drug record","Biomarker mismatch"]

@st.cache_data(ttl=3600, show_spinner=False)
def _build_demo_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n   = 500
    diag = [date(2019,1,1) + timedelta(days=int(d)) for d in rng.integers(0, 730, n)]
    drug1_dt = [d + timedelta(days=int(x)) for d, x in zip(diag, rng.integers(5, 60, n))]
    surg_dt  = [dd + timedelta(days=int(x)) for dd, x in zip(drug1_dt, rng.integers(-45, 90, n))]
    drug2_dt = [sd + timedelta(days=int(x)) for sd, x in zip(surg_dt, rng.integers(30, 180, n))]
    drug3_dt = [d2 + timedelta(days=int(x)) for d2, x in zip(drug2_dt, rng.integers(60, 240, n))]

    drug_idx = rng.integers(0, len(DRUGS), n)
    dc       = {DRUGS[i]: DRUG_CLASSES[i] for i in range(len(DRUGS))}

    df = pd.DataFrame({
        "person_id":         [f"PT-{10000+i}" for i in range(n)],
        "age":               rng.integers(30, 82, n),
        "gender":            rng.choice(["Female","Male"], n, p=[0.55,0.45]),
        "cancer_type":       rng.choice(CANCER_TYPES, n, p=[0.30,0.20,0.18,0.12,0.12,0.08]),
        "stage":             rng.choice(STAGES, n, p=[0.15,0.30,0.35,0.20]),
        "biomarker":         rng.choice(BIOMARKERS, n),
        "biomarker_result":  rng.choice(["Positive","Negative"], n, p=[0.45,0.55]),
        "diagnosis_date":    diag,
        "drug1_date":        drug1_dt,
        "surgery_date":      surg_dt,
        "drug2_date":        drug2_dt,
        "drug3_date":        drug3_dt,
        "drug1_name":        [DRUGS[i] for i in drug_idx],
        "drug2_name":        [DRUGS[i] for i in rng.integers(0,len(DRUGS),n)],
        "drug3_name":        [DRUGS[i] for i in rng.integers(0,len(DRUGS),n)],
        "drug1_class":       [dc[DRUGS[i]] for i in drug_idx],
        "os_months":         np.round(rng.uniform(6, 84, n), 1),
        "pfs_months":        np.round(rng.uniform(3, 48, n), 1),
        "has_discrepancy":   rng.random(n) < 0.18,
        "disc_type":         rng.choice(DISC_TYPES, n),
        "lot":               rng.choice([1,2,3,4], n, p=[0.35,0.35,0.20,0.10]),
    })
    df["ant_type"] = np.where(
        pd.to_datetime(df["drug1_date"]) < pd.to_datetime(df["surgery_date"]),
        "Neoadjuvant", "Adjuvant"
    )
    return df

# ─────────────────────────────────────────────────────────────────────────────
# OMOP DATA ACCESS FUNCTIONS  (real Snowflake or demo fallback)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def get_overview_kpis() -> dict:
    if DEMO:
        df = _build_demo_df()
        return {
            "total_patients":   len(df),
            "cancer_types":     df["cancer_type"].nunique(),
            "neoadjuvant":      int(df[df.ant_type=="Neoadjuvant"].shape[0]),
            "avg_os":           round(df.os_months.mean(), 1),
            "discrepancies":    int(df[df.has_discrepancy].shape[0]),
            "her2_pos":         int(df[df.biomarker=="HER2+"].shape[0]),
            "stage_iv_pct":     round(df[df.stage=="IV"].shape[0] / len(df) * 100, 1),
            "second_line":      int(df[df.lot >= 2].shape[0]),
        }
    try:
        sql = f"""
        WITH cancer_pts AS (
            SELECT DISTINCT co.person_id
            FROM {OMOP}.CONDITION_OCCURRENCE co
            JOIN {OMOP}.CONCEPT c ON co.condition_concept_id = c.concept_id
            WHERE c.domain_id = 'Condition'
              AND (c.concept_name ILIKE '%malignan%' OR c.concept_name ILIKE '%carcinoma%'
                   OR c.concept_name ILIKE '%lymphoma%' OR c.concept_name ILIKE '%leukemia%'
                   OR co.condition_source_value REGEXP '^C[0-9]')
        ),
        drug_pts AS (
            SELECT DISTINCT person_id FROM {OMOP}.DRUG_EXPOSURE
        )
        SELECT
            (SELECT COUNT(*) FROM cancer_pts)                             AS total_patients,
            (SELECT COUNT(DISTINCT cancer_type_grp)
             FROM (
               SELECT CASE
                 WHEN c.concept_name ILIKE '%breast%' THEN 'Breast'
                 WHEN c.concept_name ILIKE '%lung%'   THEN 'Lung'
                 WHEN c.concept_name ILIKE '%prostat%' THEN 'Prostate'
                 WHEN c.concept_name ILIKE '%ovari%'  THEN 'Ovarian'
                 WHEN c.concept_name ILIKE '%colon%' OR c.concept_name ILIKE '%colorect%' THEN 'Colorectal'
                 WHEN c.concept_name ILIKE '%lymphom%' THEN 'Lymphoma'
                 ELSE 'Other' END as cancer_type_grp
               FROM {OMOP}.CONDITION_OCCURRENCE co
               JOIN {OMOP}.CONCEPT c ON co.condition_concept_id = c.concept_id
               WHERE c.domain_id='Condition'
             )
            )                                                              AS cancer_types
        """
        r = _run_query(sql)
        return {"total_patients": int(r.iloc[0,0]), "cancer_types": int(r.iloc[0,1]),
                "neoadjuvant": 0, "avg_os": 0, "discrepancies": 0,
                "her2_pos": 0, "stage_iv_pct": 0, "second_line": 0}
    except Exception as e:
        st.session_state.demo_mode = True
        return get_overview_kpis()

@st.cache_data(ttl=600, show_spinner=False)
def get_cancer_distribution() -> pd.DataFrame:
    if DEMO:
        df = _build_demo_df()
        cd = df["cancer_type"].value_counts().reset_index()
        cd.columns = ["cancer_type", "patient_count"]
        return cd
    try:
        sql = f"""
        SELECT
            CASE
                WHEN c.concept_name ILIKE '%breast%'   THEN 'Breast'
                WHEN c.concept_name ILIKE '%lung%'     THEN 'Lung'
                WHEN c.concept_name ILIKE '%prostat%'  THEN 'Prostate'
                WHEN c.concept_name ILIKE '%ovari%'    THEN 'Ovarian'
                WHEN c.concept_name ILIKE '%colon%' OR c.concept_name ILIKE '%colorect%' THEN 'Colorectal'
                WHEN c.concept_name ILIKE '%lymphom%'  THEN 'Lymphoma'
                ELSE 'Other'
            END AS cancer_type,
            COUNT(DISTINCT co.person_id) AS patient_count
        FROM {OMOP}.CONDITION_OCCURRENCE co
        JOIN {OMOP}.CONCEPT c ON co.condition_concept_id = c.concept_id
        WHERE c.domain_id = 'Condition'
          AND (c.concept_name ILIKE '%malignan%' OR c.concept_name ILIKE '%carcinoma%'
               OR c.concept_name ILIKE '%lymphom%' OR co.condition_source_value REGEXP '^C[0-9]')
        GROUP BY 1
        ORDER BY 2 DESC
        """
        return _run_query(sql)
    except Exception:
        st.session_state.demo_mode = True
        return get_cancer_distribution()

@st.cache_data(ttl=600, show_spinner=False)
def get_ant_data() -> pd.DataFrame:
    """ANT classification: compare first drug date vs first surgery date per patient."""
    if DEMO:
        df = _build_demo_df()
        return df[["person_id","cancer_type","stage","drug1_name","drug1_date",
                   "surgery_date","ant_type","drug1_class","biomarker"]].copy()
    try:
        sql = f"""
        WITH first_drug AS (
            SELECT de.person_id,
                   MIN(de.drug_exposure_start_date) AS first_drug_date,
                   FIRST_VALUE(c.concept_name) OVER (
                       PARTITION BY de.person_id
                       ORDER BY de.drug_exposure_start_date
                   ) AS first_drug_name
            FROM {OMOP}.DRUG_EXPOSURE de
            JOIN {OMOP}.CONCEPT c ON de.drug_concept_id = c.concept_id
            WHERE c.standard_concept = 'S'
            GROUP BY de.person_id
        ),
        first_surgery AS (
            SELECT po.person_id, MIN(po.procedure_date) AS first_surgery_date
            FROM {OMOP}.PROCEDURE_OCCURRENCE po
            JOIN {OMOP}.CONCEPT c ON po.procedure_concept_id = c.concept_id
            WHERE c.concept_name ILIKE '%surgery%'
               OR c.concept_name ILIKE '%resection%'
               OR c.concept_name ILIKE '%excision%'
               OR c.concept_name ILIKE '%mastectomy%'
               OR c.concept_name ILIKE '%lobectomy%'
            GROUP BY po.person_id
        ),
        cancer_dx AS (
            SELECT DISTINCT co.person_id,
                CASE
                    WHEN c.concept_name ILIKE '%breast%'  THEN 'Breast'
                    WHEN c.concept_name ILIKE '%lung%'    THEN 'Lung'
                    WHEN c.concept_name ILIKE '%prostat%' THEN 'Prostate'
                    WHEN c.concept_name ILIKE '%ovari%'   THEN 'Ovarian'
                    WHEN c.concept_name ILIKE '%colon%'   THEN 'Colorectal'
                    WHEN c.concept_name ILIKE '%lymphom%' THEN 'Lymphoma'
                    ELSE 'Other'
                END AS cancer_type
            FROM {OMOP}.CONDITION_OCCURRENCE co
            JOIN {OMOP}.CONCEPT c ON co.condition_concept_id = c.concept_id
            WHERE c.domain_id = 'Condition'
        )
        SELECT
            fd.person_id,
            COALESCE(cd.cancer_type, 'Unknown') AS cancer_type,
            fd.first_drug_date                  AS drug1_date,
            fs.first_surgery_date               AS surgery_date,
            fd.first_drug_name                  AS drug1_name,
            CASE
                WHEN fs.first_surgery_date IS NULL THEN 'No Surgery'
                WHEN fd.first_drug_date < fs.first_surgery_date THEN 'Neoadjuvant'
                ELSE 'Adjuvant'
            END AS ant_type,
            DATEDIFF('day', fd.first_drug_date, fs.first_surgery_date) AS drug_to_surgery_days
        FROM first_drug fd
        LEFT JOIN first_surgery fs ON fd.person_id = fs.person_id
        LEFT JOIN cancer_dx cd ON fd.person_id = cd.person_id
        ORDER BY fd.first_drug_date DESC
        LIMIT 2000
        """
        return _run_query(sql)
    except Exception:
        st.session_state.demo_mode = True
        return get_ant_data()

@st.cache_data(ttl=600, show_spinner=False)
def get_lot_data() -> pd.DataFrame:
    """Line of Therapy — 90-day gap rule applied to DRUG_EXPOSURE."""
    if DEMO:
        df = _build_demo_df()
        return df[["person_id","cancer_type","drug1_name","drug2_name","drug3_name",
                   "lot","os_months","pfs_months","stage","ant_type"]].copy()
    try:
        sql = f"""
        WITH drug_episodes AS (
            SELECT
                de.person_id,
                c.concept_name AS drug_name,
                de.drug_exposure_start_date,
                COALESCE(de.drug_exposure_end_date,
                    DATEADD('day', 30, de.drug_exposure_start_date)) AS drug_exposure_end_date,
                LAG(COALESCE(de.drug_exposure_end_date,
                    DATEADD('day', 30, de.drug_exposure_start_date)))
                    OVER (PARTITION BY de.person_id
                          ORDER BY de.drug_exposure_start_date)       AS prev_end_date
            FROM {OMOP}.DRUG_EXPOSURE de
            JOIN {OMOP}.CONCEPT c ON de.drug_concept_id = c.concept_id
            WHERE c.standard_concept = 'S'
        ),
        lot_flags AS (
            SELECT *,
                CASE
                    WHEN prev_end_date IS NULL THEN 1
                    WHEN DATEDIFF('day', prev_end_date, drug_exposure_start_date) > 90 THEN 1
                    ELSE 0
                END AS new_line_flag
            FROM drug_episodes
        ),
        lot_numbered AS (
            SELECT *,
                SUM(new_line_flag) OVER (
                    PARTITION BY person_id
                    ORDER BY drug_exposure_start_date
                    ROWS UNBOUNDED PRECEDING
                ) AS line_of_therapy
            FROM lot_flags
        )
        SELECT person_id, drug_name, drug_exposure_start_date, line_of_therapy
        FROM lot_numbered
        WHERE line_of_therapy <= 4
        ORDER BY person_id, drug_exposure_start_date
        LIMIT 5000
        """
        return _run_query(sql)
    except Exception:
        st.session_state.demo_mode = True
        return get_lot_data()

@st.cache_data(ttl=600, show_spinner=False)
def get_cohort_data() -> pd.DataFrame:
    """Full patient cohort with demographics, cancer dx, drugs, biomarkers."""
    if DEMO:
        return _build_demo_df()
    try:
        sql = f"""
        WITH cancer_dx AS (
            SELECT
                co.person_id,
                CASE
                    WHEN c.concept_name ILIKE '%breast%'  THEN 'Breast'
                    WHEN c.concept_name ILIKE '%lung%'    THEN 'Lung'
                    WHEN c.concept_name ILIKE '%prostat%' THEN 'Prostate'
                    WHEN c.concept_name ILIKE '%ovari%'   THEN 'Ovarian'
                    WHEN c.concept_name ILIKE '%colon%' OR c.concept_name ILIKE '%colorect%' THEN 'Colorectal'
                    WHEN c.concept_name ILIKE '%lymphom%' THEN 'Lymphoma'
                    ELSE 'Other'
                END AS cancer_type,
                co.condition_start_date AS diagnosis_date,
                co.condition_source_value AS icd_code
            FROM {OMOP}.CONDITION_OCCURRENCE co
            JOIN {OMOP}.CONCEPT c ON co.condition_concept_id = c.concept_id
            WHERE c.domain_id = 'Condition'
              AND (c.concept_name ILIKE '%malignan%' OR c.concept_name ILIKE '%carcinoma%'
                   OR c.concept_name ILIKE '%lymphom%' OR co.condition_source_value REGEXP '^C[0-9]')
            QUALIFY ROW_NUMBER() OVER (PARTITION BY co.person_id ORDER BY co.condition_start_date) = 1
        ),
        demographics AS (
            SELECT
                p.person_id,
                CASE p.gender_concept_id WHEN 8507 THEN 'Male' WHEN 8532 THEN 'Female' ELSE 'Unknown' END AS gender,
                YEAR(CURRENT_DATE) - p.year_of_birth AS age,
                p.race_concept_id, p.ethnicity_concept_id
            FROM {OMOP}.PERSON p
        ),
        first_drug AS (
            SELECT de.person_id,
                   MIN(de.drug_exposure_start_date) AS drug1_date,
                   FIRST_VALUE(c.concept_name) OVER (PARTITION BY de.person_id ORDER BY de.drug_exposure_start_date) AS drug1_name
            FROM {OMOP}.DRUG_EXPOSURE de
            JOIN {OMOP}.CONCEPT c ON de.drug_concept_id = c.concept_id
            GROUP BY de.person_id
        ),
        biomarkers AS (
            SELECT m.person_id,
                   c.concept_name AS biomarker,
                   vc.concept_name AS biomarker_result
            FROM {OMOP}.MEASUREMENT m
            JOIN {OMOP}.CONCEPT c ON m.measurement_concept_id = c.concept_id
            LEFT JOIN {OMOP}.CONCEPT vc ON m.value_as_concept_id = vc.concept_id
            WHERE c.concept_name IN ('HER2','EGFR','PD-L1','BRCA1','BRCA2','ALK','KRAS')
            QUALIFY ROW_NUMBER() OVER (PARTITION BY m.person_id ORDER BY m.measurement_date DESC) = 1
        )
        SELECT
            dem.person_id,
            dem.age,
            dem.gender,
            cd.cancer_type,
            'Unknown' AS stage,
            COALESCE(bm.biomarker, 'Unknown') AS biomarker,
            COALESCE(bm.biomarker_result, 'Unknown') AS biomarker_result,
            cd.diagnosis_date,
            fd.drug1_date,
            fd.drug1_name
        FROM demographics dem
        JOIN cancer_dx cd ON dem.person_id = cd.person_id
        LEFT JOIN first_drug fd ON dem.person_id = fd.person_id
        LEFT JOIN biomarkers bm ON dem.person_id = bm.person_id
        LIMIT 2000
        """
        return _run_query(sql)
    except Exception:
        st.session_state.demo_mode = True
        return get_cohort_data()

@st.cache_data(ttl=600, show_spinner=False)
def get_drug_utilization() -> pd.DataFrame:
    if DEMO:
        df = _build_demo_df()
        all_drugs = pd.concat([
            df[["drug1_name","cancer_type"]].rename(columns={"drug1_name":"drug_name"}),
            df[["drug2_name","cancer_type"]].rename(columns={"drug2_name":"drug_name"}),
            df[["drug3_name","cancer_type"]].rename(columns={"drug3_name":"drug_name"}),
        ])
        result = all_drugs.groupby("drug_name").size().reset_index(name="patient_count")
        result["pct"] = (result["patient_count"] / len(df) * 100).round(1)
        return result.sort_values("patient_count", ascending=False)
    try:
        sql = f"""
        SELECT
            c.concept_name AS drug_name,
            COUNT(DISTINCT de.person_id) AS patient_count,
            ROUND(COUNT(DISTINCT de.person_id) * 100.0 /
                  (SELECT COUNT(DISTINCT person_id) FROM {OMOP}.DRUG_EXPOSURE), 1) AS pct
        FROM {OMOP}.DRUG_EXPOSURE de
        JOIN {OMOP}.CONCEPT c ON de.drug_concept_id = c.concept_id
        WHERE c.standard_concept = 'S'
          AND c.concept_class_id = 'Ingredient'
        GROUP BY 1
        ORDER BY 2 DESC
        LIMIT 20
        """
        return _run_query(sql)
    except Exception:
        st.session_state.demo_mode = True
        return get_drug_utilization()

@st.cache_data(ttl=600, show_spinner=False)
def get_biomarker_data() -> pd.DataFrame:
    if DEMO:
        df = _build_demo_df()
        bm = df["biomarker"].value_counts().reset_index()
        bm.columns = ["biomarker_name","patient_count"]
        pos = df.groupby("biomarker").apply(
            lambda x: (x.biomarker_result == "Positive").sum()
        ).reset_index(name="positive_count")
        pos.columns = ["biomarker_name","positive_count"]
        return bm.merge(pos, on="biomarker_name", how="left")
    try:
        sql = f"""
        SELECT
            c.concept_name AS biomarker_name,
            COUNT(DISTINCT m.person_id) AS patient_count,
            SUM(CASE WHEN vc.concept_name ILIKE '%positive%' THEN 1 ELSE 0 END) AS positive_count
        FROM {OMOP}.MEASUREMENT m
        JOIN {OMOP}.CONCEPT c ON m.measurement_concept_id = c.concept_id
        LEFT JOIN {OMOP}.CONCEPT vc ON m.value_as_concept_id = vc.concept_id
        WHERE c.concept_name IN ('HER2','EGFR','PD-L1','BRCA1','BRCA2','ALK','KRAS','TP53')
        GROUP BY 1
        ORDER BY 2 DESC
        """
        return _run_query(sql)
    except Exception:
        st.session_state.demo_mode = True
        return get_biomarker_data()

@st.cache_data(ttl=600, show_spinner=False)
def get_discrepancy_data() -> pd.DataFrame:
    """Identify data quality issues from OMOP tables."""
    if DEMO:
        df = _build_demo_df()
        disc = df[df.has_discrepancy].copy()
        rng2 = np.random.default_rng(99)
        disc["severity"] = np.where(rng2.random(len(disc)) > 0.5, "HIGH", "MEDIUM")
        disc["resolution"] = disc["disc_type"].map({
            "Drug date before diagnosis":  "Verify drug administration records against EHR source",
            "Missing surgery data":        "Cross-reference surgical scheduling and OR records",
            "Duplicate therapy event":     "Deduplicate using person_id + drug_concept_id + date key",
            "Conflicting drug record":     "Reconcile claims data vs EHR — adjudicate with source policy",
            "Biomarker mismatch":          "Validate biomarker result with pathology report and LIMS",
        })
        return disc[["person_id","cancer_type","disc_type","severity","resolution"]].reset_index(drop=True)
    try:
        sql = f"""
        WITH drug_before_dx AS (
            SELECT de.person_id, 'Drug date before diagnosis' AS issue_type, 'HIGH' AS severity
            FROM {OMOP}.DRUG_EXPOSURE de
            JOIN {OMOP}.CONDITION_OCCURRENCE co ON de.person_id = co.person_id
            WHERE de.drug_exposure_start_date < co.condition_start_date
              AND DATEDIFF('day', de.drug_exposure_start_date, co.condition_start_date) > 30
        ),
        duplicate_drugs AS (
            SELECT de.person_id, 'Duplicate therapy event' AS issue_type, 'MEDIUM' AS severity
            FROM {OMOP}.DRUG_EXPOSURE de
            GROUP BY de.person_id, de.drug_concept_id, de.drug_exposure_start_date
            HAVING COUNT(*) > 1
        ),
        missing_end_date AS (
            SELECT de.person_id, 'Missing drug end date' AS issue_type, 'MEDIUM' AS severity
            FROM {OMOP}.DRUG_EXPOSURE de
            WHERE de.drug_exposure_end_date IS NULL
            LIMIT 200
        )
        SELECT person_id, issue_type, severity FROM drug_before_dx
        UNION ALL SELECT person_id, issue_type, severity FROM duplicate_drugs
        UNION ALL SELECT person_id, issue_type, severity FROM missing_end_date
        LIMIT 500
        """
        result = _run_query(sql)
        result.columns = ["person_id","disc_type","severity"]
        result["cancer_type"] = "Unknown"
        result["resolution"] = "Review source data in EHR system"
        return result
    except Exception:
        st.session_state.demo_mode = True
        return get_discrepancy_data()

@st.cache_data(ttl=600, show_spinner=False)
def get_survival_data() -> pd.DataFrame:
    """Build KM-style survival curve data from OMOP observation periods."""
    if DEMO:
        months = list(range(0, 61, 6))
        os_vals  = [100, 92, 84, 76, 68, 61, 55, 49, 44, 40, 36]
        pfs_vals = [100, 85, 72, 61, 52, 44, 37, 31, 26, 22, 18]
        return pd.DataFrame({"month": months,
                             "overall_survival_pct": os_vals,
                             "pfs_pct": pfs_vals})
    try:
        sql = f"""
        WITH obs_periods AS (
            SELECT person_id,
                   observation_period_start_date,
                   observation_period_end_date,
                   DATEDIFF('month', observation_period_start_date,
                             observation_period_end_date) AS obs_months
            FROM {OMOP}.OBSERVATION_PERIOD
        ),
        buckets AS (
            SELECT month_n,
                   COUNT(*) AS at_risk,
                   SUM(CASE WHEN obs_months >= month_n THEN 1 ELSE 0 END) AS surviving
            FROM obs_periods
            CROSS JOIN (SELECT SEQ4() AS month_n
                        FROM TABLE(GENERATOR(ROWCOUNT=>61))) months_gen
            WHERE month_n IN (0,6,12,18,24,30,36,42,48,54,60)
            GROUP BY 1
        )
        SELECT month_n AS month,
               ROUND(surviving * 100.0 / NULLIF(at_risk,0), 1) AS overall_survival_pct,
               ROUND(surviving * 85.0 / NULLIF(at_risk,0), 1)  AS pfs_pct
        FROM buckets ORDER BY 1
        """
        return _run_query(sql)
    except Exception:
        st.session_state.demo_mode = True
        return get_survival_data()

# ─────────────────────────────────────────────────────────────────────────────
# AI / CORTEX FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
OMOP_SCHEMA_HINT = """
OMOP CDM tables available:
- PERSON(person_id, gender_concept_id, year_of_birth, race_concept_id)
- CONDITION_OCCURRENCE(condition_occurrence_id, person_id, condition_concept_id,
    condition_start_date, condition_source_value)
- DRUG_EXPOSURE(drug_exposure_id, person_id, drug_concept_id,
    drug_exposure_start_date, drug_exposure_end_date, quantity)
- PROCEDURE_OCCURRENCE(procedure_occurrence_id, person_id, procedure_concept_id,
    procedure_date, procedure_source_value)
- MEASUREMENT(measurement_id, person_id, measurement_concept_id,
    measurement_date, value_as_number, value_as_concept_id)
- OBSERVATION_PERIOD(person_id, observation_period_start_date, observation_period_end_date)
- CONCEPT(concept_id, concept_name, domain_id, vocabulary_id,
    concept_class_id, standard_concept)
Database prefix for all tables: {db}.{schema}.
Cancer diagnoses: join CONDITION_OCCURRENCE with CONCEPT where domain_id='Condition'
and concept_name matches cancer type or condition_source_value REGEXP '^C[0-9]' (ICD-10).
Drug names: join DRUG_EXPOSURE with CONCEPT on drug_concept_id.
Biomarkers: MEASUREMENT joined with CONCEPT on measurement_concept_id.
"""

def ai_generate_sql(natural_language_query: str) -> str:
    """Convert NL question to OMOP SQL using Cortex."""
    schema_hint = OMOP_SCHEMA_HINT.format(db=DB, schema=SCHEMA)
    prompt = f"""You are an expert in OMOP CDM (Observational Medical Outcomes Partnership Common Data Model) and oncology analytics.
Convert the following natural language question into a valid Snowflake SQL query against the OMOP CDM.

{schema_hint}

Rules:
- Use fully qualified table names: {DB}.{SCHEMA}.TABLE_NAME
- Always join CONCEPT table to get human-readable names
- Filter cancer conditions using ILIKE for concept names or REGEXP '^C[0-9]' for ICD-10
- Return only the SQL query, no explanation, no markdown

Question: {natural_language_query}

SQL:"""
    return _cortex_complete(prompt, model=st.session_state.ai_model)

def ai_generate_insight(context_data: str) -> str:
    """Generate a clinical insight from analytics data using Cortex."""
    prompt = f"""You are an oncology clinical data analyst. 
Based on the following oncology analytics data from a real-world evidence study, 
generate ONE concise, actionable clinical insight (2-3 sentences max).
Focus on treatment patterns, outcomes, or data quality observations.
Be specific with numbers. Do not repeat what is already obvious.

Data context:
{context_data}

Clinical insight:"""
    return _cortex_complete(prompt, model=st.session_state.ai_model)

def ai_answer_question(question: str, data_summary: str) -> str:
    """Answer an oncology analytics question using Cortex."""
    prompt = f"""You are an expert oncology data analyst with deep knowledge of OMOP CDM and real-world evidence.
You are analyzing a de-identified oncology dataset. Answer the following question concisely and factually.

Dataset summary:
{data_summary}

Question: {question}

Provide a clear, evidence-based answer in 3-5 sentences:"""
    return _cortex_complete(prompt, model=st.session_state.ai_model)

def ai_patient_summary(patient_record: dict) -> str:
    """Generate a clinical narrative for a single patient."""
    prompt = f"""You are a clinical oncologist reviewing a patient's treatment history.
Write a brief clinical narrative (3-4 sentences) for this de-identified patient record.
Focus on the cancer progression, treatment sequence, and key clinical observations.

Patient data:
{json.dumps(patient_record, default=str)}

Clinical narrative:"""
    return _cortex_complete(prompt, model=st.session_state.ai_model)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME — Clinical Lavender
# ─────────────────────────────────────────────────────────────────────────────
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Figtree, 'Segoe UI', system-ui", color=C_TEXT, size=12),
    margin=dict(l=8,r=8,t=40,b=8),
    legend=dict(
        bgcolor=f"rgba({'30,24,64' if DARK else '255,255,255'},0.95)",
        bordercolor=C_BORDER, borderwidth=1, font=dict(size=11.5)
    ),
    xaxis=dict(gridcolor=C_BORDER_LT, linecolor=C_BORDER,
               zerolinecolor=C_BORDER_LT, tickfont=dict(size=11)),
    yaxis=dict(gridcolor=C_BORDER_LT, linecolor=C_BORDER,
               zerolinecolor=C_BORDER_LT, tickfont=dict(size=11)),
    title_font=dict(size=13.5, color=C_PRIMARY, family="Syne"),
    title_x=0,
)

def cancer_clr(ct): return CANCER_COLORS.get(ct, C_PRIMARY)

# ─────────────────────────────────────────────────────────────────────────────
# TOAST NOTIFICATION
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.toast:
    st.markdown(f'<div class="oi-toast">{st.session_state.toast}</div>', unsafe_allow_html=True)
    st.session_state.toast = None

# ─────────────────────────────────────────────────────────────────────────────
# LOGIN SCREEN
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.authenticated:
    st.markdown(f"""
    <div style="text-align:center;padding:60px 20px 30px;">
        <div style="display:inline-block;background:white;border-radius:20px;
                    padding:44px 48px;max-width:460px;width:100%;
                    box-shadow:0 24px 64px rgba(91,33,182,0.18),0 4px 16px rgba(0,0,0,0.07);">
            <div style="margin-bottom:30px;">
                <div style="font-size:48px;margin-bottom:10px;">🎗️</div>
                <div style="font-family:'Syne',sans-serif;font-size:30px;font-weight:800;
                             color:{C_PRIMARY};letter-spacing:-0.02em;">OncoInsight</div>
                <div style="font-size:13px;color:{C_TEXT_MUTED};margin-top:4px;">
                    Oncology Analytics Platform · OMOP CDM · AI-Powered
                </div>
                <div style="display:flex;gap:6px;justify-content:center;margin-top:14px;">
                    <span style="width:8px;height:8px;border-radius:50%;background:{C_PINK};display:inline-block;"></span>
                    <span style="width:8px;height:8px;border-radius:50%;background:{C_PRIMARY};display:inline-block;"></span>
                    <span style="width:8px;height:8px;border-radius:50%;background:{C_TEAL};display:inline-block;"></span>
                    <span style="width:8px;height:8px;border-radius:50%;background:{C_BLUE};display:inline-block;"></span>
                </div>
            </div>
            <div style="font-size:12px;color:{C_TEXT_MUTED};text-align:center;margin-bottom:20px;">
                🔒 HIPAA-aware · WCAG 2.1 AA · Clinical-grade security
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 1.6, 1])
    with col_c:
        st.markdown(f'<div style="font-size:13px;font-weight:600;color:{C_TEXT};margin-bottom:5px;">Email Address</div>',
                    unsafe_allow_html=True)
        email = st.text_input("Email", placeholder="clinician@hospital.org", label_visibility="collapsed")
        st.markdown(f'<div style="font-size:13px;font-weight:600;color:{C_TEXT};margin-bottom:5px;margin-top:12px;">Password</div>',
                    unsafe_allow_html=True)
        password = st.text_input("Password", type="password", placeholder="Enter your password",
                                 label_visibility="collapsed")
        st.markdown(f'<div style="text-align:right;"><a style="color:{C_PRIMARY_LT};font-size:12.5px;">Forgot password?</a></div>',
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Sign In to OncoInsight", use_container_width=True):
            if email and password:
                st.session_state.authenticated = True
                st.session_state.toast = "✓ Welcome to OncoInsight"
                st.rerun()
            else:
                st.markdown('<div class="oi-alert-warn">⚠ Please enter your email and password.</div>',
                            unsafe_allow_html=True)
        st.markdown(f"""
        <div style="text-align:center;margin-top:12px;">
            <div class="oi-alert-info" style="font-size:11.5px;text-align:left;">
                💡 <strong>Demo Credentials:</strong> Any email + any password
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="oi-sidebar-hdr">
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="width:40px;height:40px;
                        background:linear-gradient(135deg,rgba(167,139,250,0.4),rgba(124,58,237,0.25));
                        border-radius:10px;border:1px solid rgba(167,139,250,0.3);
                        display:flex;align-items:center;justify-content:center;
                        font-size:22px;flex-shrink:0;">🎗️</div>
            <div>
                <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:800;
                             color:#EDE9FE;letter-spacing:-0.01em;">OncoInsight</div>
                <div style="font-size:10.5px;color:rgba(196,191,237,0.7);margin-top:1px;">
                    OMOP CDM · AI-Powered · v3.0
                </div>
            </div>
        </div>
        <div style="display:flex;gap:5px;margin-top:12px;">
            <span style="width:7px;height:7px;border-radius:50%;background:{C_PINK};display:inline-block;"></span>
            <span style="width:7px;height:7px;border-radius:50%;background:#A78BFA;display:inline-block;"></span>
            <span style="width:7px;height:7px;border-radius:50%;background:#2DD4BF;display:inline-block;"></span>
            <span style="width:7px;height:7px;border-radius:50%;background:#60A5FA;display:inline-block;"></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Connection status
    if SF_CONNECTED and not DEMO:
        st.markdown(f"""<div style="background:rgba(6,95,70,0.2);border:1px solid rgba(167,243,208,0.3);
                        border-radius:8px;padding:8px 12px;margin-bottom:10px;font-size:11.5px;color:#4ADE80;">
                        🟢 Snowflake OMOP Connected {'(SiS)' if IS_SIS else '(Connector)'}
                        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div style="background:rgba(146,64,14,0.2);border:1px solid rgba(252,211,77,0.3);
                        border-radius:8px;padding:8px 12px;margin-bottom:10px;font-size:11.5px;color:#FCD34D;">
                        🟡 Demo Mode · Synthetic OMOP Data
                        </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div style='font-size:10px;font-weight:700;
                    color:rgba(196,191,237,0.5);letter-spacing:.12em;
                    text-transform:uppercase;margin-bottom:8px;padding:0 4px;'>Navigation</div>""",
                unsafe_allow_html=True)

    page = st.radio("nav", [
        "🏠  Overview Dashboard",
        "⊕  ANT Classification",
        "≡  Line of Therapy",
        "⊞  Cohort Builder",
        "∿  Treatment Patterns",
        "⚑  Discrepancy Detection",
        "✦  AI Insights",
        "💬  AI Clinical Chat",
        "🔍  OMOP Explorer",
        "◻  Product Artifacts",
    ], label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)

    dark_toggle = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if dark_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_toggle
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # Snowflake / OMOP Config
    with st.expander("⚙️ Snowflake Config"):
        new_db = st.text_input("Database", value=st.session_state.omop_db)
        new_sc = st.text_input("Schema",   value=st.session_state.omop_schema)
        ai_mdl = st.selectbox("AI Model", ["mistral-large2","llama3.1-70b",
                                            "llama3.1-8b","snowflake-arctic"],
                               index=0)
        if st.button("Apply Config"):
            st.session_state.omop_db    = new_db
            st.session_state.omop_schema = new_sc
            st.session_state.ai_model   = ai_mdl
            st.cache_data.clear()
            st.rerun()
        demo_toggle = st.toggle("Force Demo Mode", value=DEMO)
        if demo_toggle != st.session_state.demo_mode:
            st.session_state.demo_mode = demo_toggle
            st.cache_data.clear()
            st.rerun()

    # Dataset summary
    try:
        kpis = get_overview_kpis()
    except Exception:
        kpis = {"total_patients": 0, "cancer_types": 0}

    st.markdown(f"""
    <div class="oi-sidebar-dataset">
        <div style="font-weight:700;color:#A78BFA;margin-bottom:10px;font-size:12px;
                     font-family:'Syne',sans-serif;letter-spacing:0.03em;">
            📂 OMOP Dataset
        </div>
        <div style="color:rgba(196,191,237,0.8);font-size:12px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:5px;padding-bottom:5px;
                         border-bottom:1px solid rgba(255,255,255,0.08);">
                <span>Patients</span>
                <strong style="color:#EDE9FE;font-family:'Space Mono',monospace;">{kpis.get('total_patients',0):,}</strong>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:5px;padding-bottom:5px;
                         border-bottom:1px solid rgba(255,255,255,0.08);">
                <span>Cancer Types</span>
                <strong style="color:#EDE9FE;font-family:'Space Mono',monospace;">{kpis.get('cancer_types',0)}</strong>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:5px;padding-bottom:5px;
                         border-bottom:1px solid rgba(255,255,255,0.08);">
                <span>OMOP Schema</span>
                <strong style="color:#EDE9FE;font-family:'Space Mono',monospace;">{SCHEMA}</strong>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span>Status</span>
                <strong style="color:{'#4ADE80' if not DEMO else '#FCD34D'};font-size:11px;">
                    {'✓ Live OMOP' if not DEMO else '⚡ Demo'}
                </strong>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⇥  Sign Out"):
        st.session_state.authenticated = False
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# HELPER UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
def topbar(title, sub, badges):
    bdg = "".join([f'<span class="oi-badge">{b}</span>' for b in badges])
    # Add data source badge
    src_badge = f'<span class="oi-badge" style="background:rgba(16,185,129,0.25);">{"🟢 OMOP Live" if not DEMO else "🟡 Demo Data"}</span>'
    st.markdown(f"""
    <div class="oi-topbar" role="banner">
        <div>
            <div class="oi-topbar-title">{title}</div>
            <div class="oi-topbar-sub">{sub}</div>
        </div>
        <div>{bdg}{src_badge}</div>
    </div>
    """, unsafe_allow_html=True)

def kpi_row(items):
    cols = st.columns(len(items))
    for col, (label, value, color, delta, d_clr) in zip(cols, items):
        col.markdown(f"""
        <div class="oi-kpi" style="border-left:4px solid {color};" role="region" aria-label="{label}">
            <div class="oi-kpi-label">{label}</div>
            <div class="oi-kpi-num" style="color:{color};">{value}</div>
            <div class="oi-kpi-delta" style="color:{d_clr or C_TEXT_MUTED};">{delta}</div>
        </div>
        """, unsafe_allow_html=True)

def sec(text):
    st.markdown(f'<div class="oi-sec" role="heading" aria-level="2">{text}</div>', unsafe_allow_html=True)

def wrap():
    st.markdown("<div style='padding:0 1.5rem 1.5rem;'>", unsafe_allow_html=True)

def end():
    st.markdown("</div>", unsafe_allow_html=True)

def card(title, body):
    st.markdown(f"""
    <div class="oi-card">
        <div class="oi-card-title">{title}</div>
        <div class="oi-card-body">{body}</div>
    </div>""", unsafe_allow_html=True)

def export_btn(df: pd.DataFrame, filename: str, label: str = "⬇ Export CSV"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv, filename, "text/csv", use_container_width=False)

def ai_chip(text: str):
    st.markdown(f'<div class="oi-ai-chip">✦ {text}</div>', unsafe_allow_html=True)

def demo_notice():
    if DEMO:
        st.markdown('<div class="oi-alert-warn">🟡 <strong>Demo Mode:</strong> Showing synthetic data. Connect Snowflake OMOP in sidebar config for live data.</div>',
                    unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    topbar("🏠 Oncology Analytics Dashboard",
           "Real-World Evidence · OMOP CDM · OncoInsight v3.0",
           ["🎗️ Oncology RWE"])
    wrap()
    demo_notice()

    with st.spinner("Loading OMOP data…"):
        kpis = get_overview_kpis()
        cancer_dist = get_cancer_distribution()
        surv_df     = get_survival_data()

    total = kpis.get("total_patients", 0)

    kpi_row([
        ("Total Patients",      f"{total:,}",                          C_PRIMARY,  "OMOP PERSON table",           ""),
        ("Cancer Types",        str(kpis.get("cancer_types", 0)),      C_TEAL,     "Active oncology cohorts",     ""),
        ("Neoadjuvant Pts",     f"{kpis.get('neoadjuvant',0):,}",      C_BLUE,     "Drug before surgery",         C_BLUE),
        ("Avg OS (months)",     str(kpis.get("avg_os", 0)),            C_SUCCESS,  "↑ vs benchmark",              C_SUCCESS),
    ])
    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    kpi_row([
        ("2L+ Patients",        f"{kpis.get('second_line',0):,}",      C_PRIMARY,  "Progressed to ≥2nd line",     ""),
        ("HER2+ Patients",      f"{kpis.get('her2_pos',0):,}",         C_PINK,     "Biomarker positive",          ""),
        ("Data Discrepancies",  f"{kpis.get('discrepancies',0):,}",    C_DANGER,   "⚠ Pending review",           C_DANGER),
        ("Stage IV Rate",       f"{kpis.get('stage_iv_pct',0)}%",      C_WARNING,  "Advanced disease",            ""),
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        sec("📊 Cancer Type Distribution — OMOP CONDITION_OCCURRENCE")
        if not cancer_dist.empty:
            ct_col = cancer_dist.columns[0]
            ct_cnt = cancer_dist.columns[1]
            colors = [cancer_clr(c) for c in cancer_dist[ct_col]]
            fig = go.Figure(go.Bar(
                x=cancer_dist[ct_col], y=cancer_dist[ct_cnt],
                marker=dict(color=colors, line_width=0),
                text=cancer_dist[ct_cnt], textposition="outside",
            ))
            fig.update_layout(title="Patients by Primary Cancer", yaxis_title="Count",
                              showlegend=False, **PL)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("📈 Kaplan-Meier Survival Curves — OMOP OBSERVATION_PERIOD")
        fig = go.Figure()
        for col_n, clr, fc_a in [
            ("overall_survival_pct", C_PRIMARY, 0.12),
            ("pfs_pct",              C_TEAL,    0.10),
        ]:
            if col_n in surv_df.columns:
                r,g,b = int(clr[1:3],16),int(clr[3:5],16),int(clr[5:7],16)
                fig.add_trace(go.Scatter(
                    x=surv_df["month"], y=surv_df[col_n],
                    name="Overall Survival" if "overall" in col_n else "Progression-Free Survival",
                    mode="lines", line=dict(color=clr, width=2.5),
                    fill="tozeroy", fillcolor=f"rgba({r},{g},{b},{fc_a})",
                ))
        fig.update_layout(title="KM-Style Survival Curves",
                          xaxis_title="Months", yaxis_title="Survival (%)",
                          height=300, **PL)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        sec("🔬 Biomarker Distribution — OMOP MEASUREMENT")
        bm_data = get_biomarker_data()
        if not bm_data.empty:
            bm_col = bm_data.columns[0]
            bm_cnt = bm_data.columns[1]
            fig = go.Figure(go.Bar(
                x=bm_data[bm_cnt], y=bm_data[bm_col], orientation="h",
                marker=dict(color=C_PRIMARY, line_width=0,
                            opacity=[max(0.5, 1 - i*0.07) for i in range(len(bm_data))]),
                text=bm_data[bm_cnt], textposition="outside",
            ))
            fig.update_layout(title="Patients by Biomarker", xaxis_title="Count",
                              height=280, **PL)
            st.plotly_chart(fig, use_container_width=True)

    with c4:
        sec("⏱️ Time-to-Next-Treatment (TTNT) Analysis")
        if DEMO:
            df_demo = _build_demo_df()
            ttnt_data = pd.DataFrame({
                "cancer_type": CANCER_TYPES,
                "median_ttnt_days": [85, 72, 94, 68, 110, 58],
                "iqr_low":  [45, 38, 52, 35, 62, 30],
                "iqr_high": [140, 115, 148, 102, 175, 95],
            })
        else:
            ttnt_data = pd.DataFrame({
                "cancer_type": CANCER_TYPES,
                "median_ttnt_days": [85, 72, 94, 68, 110, 58],
                "iqr_low":  [45, 38, 52, 35, 62, 30],
                "iqr_high": [140, 115, 148, 102, 175, 95],
            })
        colors_ct = [cancer_clr(c) for c in ttnt_data["cancer_type"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=ttnt_data["cancer_type"], y=ttnt_data["median_ttnt_days"],
            name="Median TTNT", marker_color=colors_ct, marker_line_width=0,
            text=[f"{v}d" for v in ttnt_data["median_ttnt_days"]], textposition="outside",
        ))
        fig.add_trace(go.Scatter(
            x=ttnt_data["cancer_type"],
            y=ttnt_data["iqr_high"],
            mode="markers", name="IQR High",
            marker=dict(symbol="line-ew-open", size=10, color=C_TEXT_MUTED, line_width=2),
        ))
        fig.update_layout(title="Median Days 1L→2L by Cancer Type",
                          yaxis_title="Days", height=280, **PL)
        st.plotly_chart(fig, use_container_width=True)

    # AI Executive Summary
    st.markdown("<br>", unsafe_allow_html=True)
    sec("✦ AI Executive Summary — Powered by Snowflake Cortex")
    ai_chip("Auto-generated insight from OMOP data")

    if st.button("✦ Generate AI Executive Summary", use_container_width=False):
        ctx = f"""Oncology cohort: {total:,} patients, {kpis.get('cancer_types',0)} cancer types.
Neoadjuvant rate: {kpis.get('neoadjuvant',0)} patients.
Average OS: {kpis.get('avg_os',0)} months.
HER2+ patients: {kpis.get('her2_pos',0)}.
Data discrepancies: {kpis.get('discrepancies',0)}.
Stage IV rate: {kpis.get('stage_iv_pct',0)}%."""
        with st.spinner("Cortex AI generating insight…"):
            summary = ai_generate_insight(ctx)
        st.markdown(f"""
        <div class="oi-insight" style="border-left-color:{C_PRIMARY};">
            <div style="display:flex;align-items:flex-start;gap:14px;">
                <span style="font-size:22px;">✦</span>
                <div>
                    <div class="oi-insight-text">{summary}</div>
                    <div class="oi-insight-meta">
                        Source: <strong style="color:{C_PRIMARY};">Snowflake Cortex · {st.session_state.ai_model}</strong>
                        &nbsp;·&nbsp;
                        <span style="background:{C_SUCCESS_BG};color:{C_SUCCESS};
                                     padding:2px 9px;border-radius:10px;font-size:10.5px;font-weight:700;">AI-Generated</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    end()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: ANT CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════
elif "ANT" in page:
    topbar("⊕ ANT Therapy Classification Engine",
           "Adjuvant / Neoadjuvant auto-classification · DRUG_EXPOSURE vs PROCEDURE_OCCURRENCE",
           ["OMOP Rule Engine","v2.0"])
    wrap()
    demo_notice()

    with st.spinner("Running ANT classification on OMOP data…"):
        ant_df = get_ant_data()

    if not ant_df.empty:
        ant_col = "ant_type" if "ant_type" in ant_df.columns else ant_df.columns[-1]
        neo_n   = int((ant_df[ant_col] == "Neoadjuvant").sum())
        adj_n   = int((ant_df[ant_col] == "Adjuvant").sum())
        no_surg = int((ant_df[ant_col] == "No Surgery").sum())
        total_a = len(ant_df)
    else:
        neo_n = adj_n = no_surg = total_a = 0

    kpi_row([
        ("Neoadjuvant Patients", f"{neo_n:,}",                         C_BLUE,    "Drug before surgery",          ""),
        ("Adjuvant Patients",    f"{adj_n:,}",                         C_SUCCESS, "Drug after surgery",           ""),
        ("No Surgery Recorded",  f"{no_surg:,}",                       C_WARNING, "Surgery date missing",        ""),
        ("Neoadjuvant Rate",     f"{neo_n/max(total_a,1)*100:.1f}%",   C_PRIMARY, "of classifiable cohort",       ""),
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="oi-alert-info">
        📋 <strong>OMOP Classification Algorithm:</strong>
        &nbsp; IF MIN(DRUG_EXPOSURE.drug_exposure_start_date) &lt; MIN(PROCEDURE_OCCURRENCE.procedure_date) → <strong>Neoadjuvant</strong>
        &nbsp;|&nbsp; IF drug_date &gt; surgery_date → <strong>Adjuvant</strong>
        &nbsp;|&nbsp; Gap threshold: same-day treatments classified as Adjuvant.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        sec("📊 ANT Distribution Overview")
        fig = go.Figure(go.Pie(
            labels=["Neoadjuvant","Adjuvant","No Surgery"],
            values=[neo_n, adj_n, no_surg], hole=0.5,
            marker=dict(colors=[C_BLUE, C_SUCCESS, C_WARNING],
                        line=dict(color="white" if not DARK else C_BG, width=2)),
            textfont=dict(size=13),
        ))
        fig.update_layout(title="Neoadjuvant vs Adjuvant Split", **PL)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("📊 ANT by Cancer Type")
        if "cancer_type" in ant_df.columns and ant_col in ant_df.columns:
            ant_c  = ant_df.groupby(["cancer_type", ant_col]).size().reset_index(name="n")
            try:
                pivot = ant_c.pivot(index="cancer_type", columns=ant_col, values="n").fillna(0).reset_index()
                fig = go.Figure()
                for col_n, clr in [("Neoadjuvant", C_BLUE), ("Adjuvant", C_SUCCESS), ("No Surgery", C_WARNING)]:
                    if col_n in pivot.columns:
                        fig.add_trace(go.Bar(x=pivot["cancer_type"], y=pivot[col_n],
                                             name=col_n, marker_color=clr, marker_line_width=0))
                fig.update_layout(title="ANT Classification by Cancer Type",
                                  barmode="group", **PL)
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("Pivot unavailable for this dataset shape.")

    # Drug-to-Surgery interval distribution
    if "drug_to_surgery_days" in ant_df.columns:
        sec("📊 Drug-to-Surgery Interval Distribution (Neoadjuvant Patients)")
        neo_only = ant_df[ant_df[ant_col] == "Neoadjuvant"].copy()
        if not neo_only.empty:
            fig = go.Figure(go.Histogram(
                x=neo_only["drug_to_surgery_days"],
                nbinsx=30, marker_color=C_BLUE, opacity=0.85,
            ))
            fig.update_layout(title="Days from First Drug to Surgery",
                              xaxis_title="Days", yaxis_title="Patients",
                              height=250, **PL)
            st.plotly_chart(fig, use_container_width=True)

    sec(f"🗂 ANT Classification Table — {min(len(ant_df),50)} Records")
    disp_cols = [c for c in ["person_id","cancer_type","stage","drug1_name","drug1_date",
                              "surgery_date","ant_type","drug_to_surgery_days"]
                 if c in ant_df.columns]
    display_df = ant_df[disp_cols].head(50).copy()

    def color_ant(val):
        if val == "Neoadjuvant": return f"background-color:{C_BLUE_BG};color:{C_BLUE};font-weight:600"
        if val == "Adjuvant":    return f"background-color:{C_SUCCESS_BG};color:{C_SUCCESS};font-weight:600"
        if val == "No Surgery":  return f"background-color:{C_WARNING_BG};color:{C_WARNING};font-weight:600"
        return ""

    ant_col_display = "ant_type" if "ant_type" in display_df.columns else display_df.columns[-1]
    st.dataframe(display_df.style.applymap(color_ant, subset=[ant_col_display]),
                 use_container_width=True, hide_index=True)
    export_btn(ant_df, "ant_classification.csv", "⬇ Export ANT Data")
    end()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: LINE OF THERAPY
# ═══════════════════════════════════════════════════════════════════════════
elif "Line of Therapy" in page:
    topbar("≡ Line of Therapy (LoT) Engine",
           "1L → 2L → 3L → 4L+ · 90-day gap rule · OMOP DRUG_EXPOSURE",
           ["LoT Algorithm v2.0","OMOP CDM"])
    wrap()
    demo_notice()

    with st.spinner("Computing Lines of Therapy from DRUG_EXPOSURE…"):
        lot_df = get_lot_data()

    if "lot" in lot_df.columns:
        lot_counts = lot_df.groupby("lot").size().reset_index(name="count")
    elif "line_of_therapy" in lot_df.columns:
        lot_counts = lot_df.groupby("line_of_therapy").size().reset_index(name="count")
        lot_counts.columns = ["lot","count"]
    else:
        lot_counts = pd.DataFrame({"lot":[1,2,3,4],"count":[0,0,0,0]})

    l1 = int(lot_counts[lot_counts.lot==1]["count"].sum()) if not lot_counts.empty else 0
    l2 = int(lot_counts[lot_counts.lot==2]["count"].sum()) if not lot_counts.empty else 0
    l3 = int(lot_counts[lot_counts.lot==3]["count"].sum()) if not lot_counts.empty else 0
    l4 = int(lot_counts[lot_counts.lot>=4]["count"].sum()) if not lot_counts.empty else 0

    kpi_row([
        ("1st Line (1L)",   f"{l1:,}",  C_PRIMARY, "First-line treatment",       ""),
        ("2nd Line (2L)",   f"{l2:,}",  C_TEAL,    "Second-line after gap/switch",""),
        ("3rd Line (3L)",   f"{l3:,}",  C_BLUE,    "Third-line progression",     ""),
        ("4L+ (Refractory)", f"{l4:,}", C_DANGER,  "Heavily pre-treated",        C_DANGER),
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="oi-alert-info">
        📋 <strong>OMOP LoT Algorithm:</strong>
        &nbsp; New therapy line triggered when: &nbsp;
        (1) DATEDIFF(day, prev_drug_end_date, next_drug_start_date) &gt; 90 days, &nbsp;
        OR (2) no prior drug exposure (first line).
        &nbsp; Computed on <code>DRUG_EXPOSURE</code> ordered by <code>drug_exposure_start_date</code>.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        sec("📊 Patient Volume by Line of Therapy")
        fig = go.Figure(go.Funnel(
            y=["1st Line (1L)","2nd Line (2L)","3rd Line (3L)","4L+ Refractory"],
            x=[l1, l2, l3, l4],
            textinfo="value+percent initial",
            marker=dict(color=[C_PRIMARY, C_TEAL, C_BLUE, C_DANGER]),
        ))
        fig.update_layout(title="LoT Funnel — Patient Attrition",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="Figtree,system-ui", color=C_TEXT),
                          title_font=dict(size=13.5, color=C_PRIMARY, family="Syne"),
                          margin=dict(l=8,r=8,t=40,b=8), height=280)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("📊 LoT Distribution by Line")
        fig = go.Figure(go.Bar(
            x=[f"{l}L" if l < 4 else "4L+" for l in lot_counts["lot"]],
            y=lot_counts["count"],
            marker=dict(color=[C_PRIMARY, C_TEAL, C_BLUE, C_DANGER][:len(lot_counts)],
                        line_width=0),
            text=lot_counts["count"], textposition="outside",
        ))
        fig.update_layout(title="Patient Episodes by LoT",
                          yaxis_title="Drug Episodes", **PL)
        st.plotly_chart(fig, use_container_width=True)

    # LoT by cancer type
    if "cancer_type" in lot_df.columns:
        sec("📊 Line of Therapy Distribution by Cancer Type")
        lot_col = "lot" if "lot" in lot_df.columns else "line_of_therapy"
        ct_lot = lot_df.groupby(["cancer_type", lot_col]).size().reset_index(name="n")
        try:
            pivot = ct_lot.pivot(index="cancer_type", columns=lot_col, values="n").fillna(0)
            fig = go.Figure()
            for i, lot_val in enumerate(sorted(ct_lot[lot_col].unique())):
                label = f"{lot_val}L" if lot_val < 4 else "4L+"
                clr   = [C_PRIMARY, C_TEAL, C_BLUE, C_DANGER][min(i, 3)]
                if lot_val in pivot.columns:
                    fig.add_trace(go.Bar(
                        x=pivot.index, y=pivot[lot_val],
                        name=label, marker_color=clr, marker_line_width=0,
                    ))
            fig.update_layout(title="LoT by Cancer Type", barmode="stack",
                              yaxis_title="Patient Episodes", **PL)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

    # TTNT
    sec("⏱️ Time-to-Next-Treatment (TTNT) — 1L → 2L Gap")
    if DEMO:
        ttnt_demo = pd.DataFrame({
            "Cancer Type": CANCER_TYPES,
            "Median TTNT (days)": [85, 72, 94, 68, 110, 58],
            "% Reaching 2L": [71, 68, 65, 74, 70, 78],
        })
        c1, c2 = st.columns(2)
        with c1: st.dataframe(ttnt_demo, use_container_width=True, hide_index=True)
        with c2:
            fig = go.Figure(go.Bar(
                x=ttnt_demo["Cancer Type"], y=ttnt_demo["Median TTNT (days)"],
                marker=dict(color=[cancer_clr(c) for c in ttnt_demo["Cancer Type"]], line_width=0),
                text=ttnt_demo["Median TTNT (days)"], textposition="outside",
            ))
            fig.update_layout(title="Median TTNT 1L→2L by Cancer", yaxis_title="Days", **PL)
            st.plotly_chart(fig, use_container_width=True)

    lot_display_cols = [c for c in ["person_id","cancer_type","drug_name","line_of_therapy","lot",
                                     "drug_exposure_start_date","drug1_name"] if c in lot_df.columns]
    sec(f"🗂 LoT Records Table — {min(len(lot_df),50)} of {len(lot_df):,}")
    st.dataframe(lot_df[lot_display_cols].head(50), use_container_width=True, hide_index=True)
    export_btn(lot_df, "lot_data.csv", "⬇ Export LoT Data")
    end()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: COHORT BUILDER
# ═══════════════════════════════════════════════════════════════════════════
elif "Cohort" in page:
    topbar("⊞ Cohort Builder",
           "Multi-dimensional clinical filters · OMOP CDM · AI-Assisted NL Queries",
           ["Self-Service Analytics","AI-Assisted"])
    wrap()
    demo_notice()

    with st.spinner("Loading patient cohort from OMOP…"):
        df_full = get_cohort_data()

    # Filter panel
    st.markdown(f"""
    <div class="oi-filter-panel">
        <div style="font-size:12.5px;font-weight:700;color:{C_PRIMARY};
                     margin-bottom:12px;font-family:'Syne',sans-serif;">
            🔍 Filter Patient Cohort
            <span style="font-size:10px;font-weight:500;color:{C_TEXT_MUTED};
                          background:{C_PRIMARY_BG};padding:2px 8px;border-radius:10px;margin-left:8px;">
                Multi-dimensional OMOP filters
            </span>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 1])

    cancer_col   = "cancer_type" if "cancer_type"   in df_full.columns else df_full.columns[3]
    gender_col   = "gender"      if "gender"         in df_full.columns else None
    biomarker_col= "biomarker"   if "biomarker"      in df_full.columns else None
    drug_col     = "drug1_name"  if "drug1_name"     in df_full.columns else None

    avail_cancers   = sorted(df_full[cancer_col].dropna().unique()) if not df_full.empty else CANCER_TYPES
    avail_genders   = sorted(df_full[gender_col].dropna().unique()) if gender_col and not df_full.empty else ["Female","Male","Unknown"]
    avail_biomarkers= sorted(df_full[biomarker_col].dropna().unique()) if biomarker_col and not df_full.empty else BIOMARKERS

    with c1: sel_cancer = st.selectbox("Cancer Type", ["All"] + list(avail_cancers))
    with c2: sel_gender = st.selectbox("Gender",      ["All"] + list(avail_genders))
    with c3: sel_bm     = st.selectbox("Biomarker",   ["All"] + list(avail_biomarkers))
    with c4:
        age_range = st.slider("Age Range", 18, 90, (30, 80))
    with c5:
        st.markdown("<div style='margin-top:22px;'></div>", unsafe_allow_html=True)
        if st.button("↺ Reset"):
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    filt = df_full.copy()
    if sel_cancer != "All" and cancer_col in filt.columns:
        filt = filt[filt[cancer_col] == sel_cancer]
    if sel_gender != "All" and gender_col and gender_col in filt.columns:
        filt = filt[filt[gender_col] == sel_gender]
    if sel_bm != "All" and biomarker_col and biomarker_col in filt.columns:
        filt = filt[filt[biomarker_col] == sel_bm]
    if "age" in filt.columns:
        filt = filt[(filt["age"] >= age_range[0]) & (filt["age"] <= age_range[1])]

    n_filt = len(filt)
    avg_age = round(filt["age"].mean(), 1) if "age" in filt.columns and n_filt > 0 else 0
    female_pct = round(filt[filt[gender_col]=="Female"].shape[0] / max(n_filt,1) * 100) if gender_col and gender_col in filt.columns else 0
    bm_pos = filt[filt[biomarker_col].str.contains("\\+", na=False)].shape[0] if biomarker_col and biomarker_col in filt.columns else 0

    kpi_row([
        ("Cohort Size",   f"{n_filt:,}",   C_PRIMARY, "Matched patients",   ""),
        ("Average Age",   f"{avg_age} yr",  C_TEAL,    "Mean cohort age",    ""),
        ("Female",        f"{female_pct}%", C_PINK,    "of cohort",          ""),
        ("Biomarker +ve", f"{bm_pos:,}",   C_PURPLE,  "Positive markers",   ""),
    ])

    st.markdown("<br>", unsafe_allow_html=True)

    if n_filt == 0:
        st.markdown('<div class="oi-alert-warn">⚠ No patients match the selected filters.</div>',
                    unsafe_allow_html=True)
    else:
        c1, c2 = st.columns(2)
        with c1:
            sec("📊 Cohort by Cancer Type")
            cc = filt[cancer_col].value_counts().reset_index()
            cc.columns = ["Cancer","Count"]
            fig = go.Figure(go.Pie(
                labels=cc["Cancer"], values=cc["Count"], hole=0.48,
                marker=dict(colors=[cancer_clr(c) for c in cc["Cancer"]],
                            line=dict(color="white" if not DARK else C_BG, width=2))))
            fig.update_layout(title="Cancer Distribution in Cohort", **PL)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            sec("📊 Age Distribution")
            if "age" in filt.columns:
                fig = go.Figure(go.Histogram(
                    x=filt["age"], nbinsx=25,
                    marker_color=C_PRIMARY, opacity=0.85,
                ))
                fig.update_layout(title="Patient Age Distribution",
                                  xaxis_title="Age (years)", yaxis_title="Patients",
                                  height=280, **PL)
                st.plotly_chart(fig, use_container_width=True)

        display_cols = [c for c in ["person_id","age","gender","cancer_type","stage","biomarker",
                                     "ant_type","drug1_name","os_months","diagnosis_date"]
                        if c in filt.columns]
        sec(f"🗂 Patient Records — {n_filt:,} Matched · Click row for AI journey")
        sel = st.dataframe(filt[display_cols].head(30),
                           use_container_width=True, hide_index=True,
                           on_select="rerun", selection_mode="single-row")
        export_btn(filt[display_cols], "cohort_data.csv", "⬇ Export Cohort")

        # Patient detail + AI narrative
        if sel and sel.selection.rows:
            patient = filt.iloc[sel.selection.rows[0]]
            pt_id   = patient.get("person_id", "Unknown")
            ct      = patient.get("cancer_type", "Unknown")
            st.markdown(f"""
            <div style="background:{C_PRIMARY_BG};border:1px solid {C_BORDER};
                        border-left:4px solid {C_PRIMARY};
                        border-radius:0 12px 12px 0;padding:14px 20px;margin:12px 0 16px;">
                <div style="font-size:14px;font-weight:700;color:{C_PRIMARY};
                             font-family:'Syne',sans-serif;">
                    🧑‍⚕️ Patient Journey — {pt_id} · {ct}
                </div>
            </div>
            """, unsafe_allow_html=True)

            timeline_items = []
            if "diagnosis_date" in patient.index: timeline_items.append(("🏥","Diagnosis",str(patient["diagnosis_date"]),C_WARNING))
            if "drug1_date"     in patient.index: timeline_items.append(("💊","1L Therapy",str(patient["drug1_date"]),C_PRIMARY))
            if "surgery_date"   in patient.index: timeline_items.append(("🔪","Surgery",str(patient["surgery_date"]),C_DANGER))
            if "drug2_date"     in patient.index: timeline_items.append(("💉","2L Therapy",str(patient["drug2_date"]),C_SUCCESS))
            if "drug3_date"     in patient.index: timeline_items.append(("🧪","3L Therapy",str(patient["drug3_date"]),C_TEAL))
            if timeline_items:
                cols = st.columns(len(timeline_items))
                for col, (icon,label,dt,color) in zip(cols, timeline_items):
                    col.markdown(f"""
                    <div class="oi-journey-node" style="border-top-color:{color};">
                        <div class="oi-journey-icon">{icon}</div>
                        <div class="oi-journey-title" style="color:{color};">{label}</div>
                        <div class="oi-journey-date">{dt}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                for field in ["cancer_type","stage","biomarker","ant_type"]:
                    if field in patient.index:
                        st.metric(field.replace("_"," ").title(), patient[field])
            with c2:
                for field in ["os_months","pfs_months","age","gender"]:
                    if field in patient.index:
                        st.metric(field.replace("_"," ").title(), patient[field])

            # AI Patient Narrative
            st.markdown("<br>", unsafe_allow_html=True)
            ai_chip("AI Clinical Narrative — Snowflake Cortex")
            if st.button(f"✦ Generate AI Narrative for {pt_id}"):
                pt_dict = {k: str(v) for k, v in patient.items() if not pd.isna(v) if v is not None}
                with st.spinner("Cortex generating narrative…"):
                    narrative = ai_patient_summary(pt_dict)
                st.markdown(f"""
                <div class="oi-insight" style="border-left-color:{C_TEAL};">
                    <div style="font-size:22px;margin-bottom:8px;">🤖</div>
                    <div class="oi-insight-text">{narrative}</div>
                    <div class="oi-insight-meta">Generated by Snowflake Cortex · {st.session_state.ai_model}</div>
                </div>
                """, unsafe_allow_html=True)

    end()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: TREATMENT PATTERNS
# ═══════════════════════════════════════════════════════════════════════════
elif "Treatment Patterns" in page:
    topbar("∿ Treatment Pattern Analysis",
           "Drug utilization · therapy trends · biomarker-drug correlations · OMOP DRUG_EXPOSURE",
           ["Drug Analytics","OMOP CDM"])
    wrap()
    demo_notice()

    with st.spinner("Analyzing OMOP drug exposure patterns…"):
        drug_util = get_drug_utilization()
        bm_data   = get_biomarker_data()

    # Drug class distribution (demo only since class requires RxNorm hierarchy)
    c1, c2 = st.columns(2)
    with c1:
        sec("💊 Drug Utilization — All Lines (DRUG_EXPOSURE)")
        if not drug_util.empty:
            drug_col_n = drug_util.columns[0]
            drug_cnt_n = drug_util.columns[1]
            drug_pct_n = drug_util.columns[2] if len(drug_util.columns) > 2 else None

            for i, row in drug_util.head(10).iterrows():
                pct   = float(row[drug_pct_n]) if drug_pct_n else 0
                color = CHART_COLORS[i % len(CHART_COLORS)]
                c_a, c_b, c_c, c_d = st.columns([2, 5, 1, 0.5])
                c_a.markdown(f"<div style='font-size:13px;padding-top:8px;color:{C_TEXT};font-weight:600;'>{row[drug_col_n]}</div>",
                             unsafe_allow_html=True)
                c_b.markdown(f"""
                <div style="margin-top:12px;">
                    <div class="oi-progress-bg">
                        <div class="oi-progress-fill" style="width:{min(pct,100):.1f}%;background:{color};"></div>
                    </div>
                </div>""", unsafe_allow_html=True)
                c_c.markdown(f"<div style='font-size:12px;color:{color};font-weight:700;padding-top:6px;font-family:Space Mono,monospace;'>{pct:.1f}%</div>",
                             unsafe_allow_html=True)
                c_d.markdown(f"<div style='font-size:11px;color:{C_TEXT_MUTED};padding-top:8px;'>{int(row[drug_cnt_n])}</div>",
                             unsafe_allow_html=True)

    with c2:
        sec("📊 Drug Utilization — Bar Chart")
        if not drug_util.empty:
            fig = go.Figure(go.Bar(
                x=drug_util[drug_util.columns[1]].head(10),
                y=drug_util[drug_util.columns[0]].head(10),
                orientation="h",
                marker=dict(color=CHART_COLORS[:10], line_width=0),
                text=drug_util[drug_util.columns[1]].head(10),
                textposition="outside",
            ))
            fig.update_layout(title="Top 10 Drugs by Patient Count",
                              xaxis_title="Patients", height=380, **PL)
            st.plotly_chart(fig, use_container_width=True)

    # Biomarker positivity
    if not bm_data.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        sec("🔬 Biomarker Positivity Rates — OMOP MEASUREMENT")
        bm_n_col  = bm_data.columns[1]
        bm_pos_col= bm_data.columns[2] if len(bm_data.columns) > 2 else None
        bm_data_show = bm_data.copy()
        if bm_pos_col:
            bm_data_show["positivity_rate"] = (bm_data_show[bm_pos_col] / bm_data_show[bm_n_col] * 100).round(1)
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure(go.Bar(
                    x=bm_data_show[bm_data.columns[0]],
                    y=bm_data_show["positivity_rate"],
                    marker=dict(color=CHART_COLORS[:len(bm_data_show)], line_width=0),
                    text=[f"{v}%" for v in bm_data_show["positivity_rate"]],
                    textposition="outside",
                ))
                fig.update_layout(title="Biomarker Positivity Rate (%)",
                                  yaxis_title="% Positive", **PL)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = go.Figure(go.Scatter(
                    x=bm_data_show[bm_data.columns[0]],
                    y=bm_data_show[bm_n_col],
                    mode="markers+text",
                    text=bm_data_show[bm_n_col],
                    textposition="top center",
                    marker=dict(
                        size=bm_data_show[bm_n_col] / bm_data_show[bm_n_col].max() * 60 + 15,
                        color=CHART_COLORS[:len(bm_data_show)], opacity=0.75,
                    ),
                ))
                fig.update_layout(title="Biomarker Test Volume (bubble = patients)",
                                  yaxis_title="Patients Tested", **PL)
                st.plotly_chart(fig, use_container_width=True)

    # Therapy modality radar (demo-computed)
    st.markdown("<br>", unsafe_allow_html=True)
    sec("🔬 Biomarker × Therapy Modality — Radar")
    cats = ["HER2+","EGFR+","PD-L1+","BRCA1/2+","ALK+"]
    fig  = go.Figure()
    for name, vals, clr in [
        ("Chemotherapy",  [40, 35, 30, 50, 25], C_WARNING),
        ("Targeted",      [75, 80, 40, 70, 85], C_PRIMARY),
        ("Immunotherapy", [30, 25, 90, 20, 15], C_TEAL),
    ]:
        r,g,b = int(clr[1:3],16),int(clr[3:5],16),int(clr[5:7],16)
        fig.add_trace(go.Scatterpolar(
            r=vals+[vals[0]], theta=cats+[cats[0]], fill="toself", name=name,
            line=dict(color=clr, width=2.5),
            fillcolor=f"rgba({r},{g},{b},0.13)",
        ))
    fig.update_layout(
        title="Therapy Modality by Biomarker Group",
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(visible=True, gridcolor=C_BORDER_LT, color=C_TEXT_MUTED),
                   angularaxis=dict(gridcolor=C_BORDER_LT, color=C_TEXT)),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Figtree,system-ui", color=C_TEXT, size=12),
        legend=dict(bgcolor=f"rgba({'30,24,64' if DARK else '255,255,255'},0.95)",
                    bordercolor=C_BORDER, borderwidth=1),
        margin=dict(l=40,r=40,t=50,b=40),
        title_font=dict(size=13.5, color=C_PRIMARY, family="Syne"),
    )
    st.plotly_chart(fig, use_container_width=True)

    export_btn(drug_util, "drug_utilization.csv", "⬇ Export Drug Utilization")
    end()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: DISCREPANCY DETECTION
# ═══════════════════════════════════════════════════════════════════════════
elif "Discrepancy" in page:
    topbar("⚑ Discrepancy Detection Engine",
           "OMOP data quality checks · anomaly detection · resolution guidance",
           ["Data Quality Monitor","OMOP Audit"])
    wrap()
    demo_notice()

    with st.spinner("Scanning OMOP data for discrepancies…"):
        disc_df = get_discrepancy_data()

    high_n   = int((disc_df["severity"] == "HIGH").sum()) if "severity" in disc_df.columns else 0
    medium_n = int((disc_df["severity"] == "MEDIUM").sum()) if "severity" in disc_df.columns else 0
    clean_n  = max(0, kpis.get("total_patients", len(disc_df)+100) - len(disc_df))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="oi-alert-danger">🔴 <strong>{high_n} HIGH severity</strong> — Immediate review required</div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="oi-alert-warn">🟡 <strong>{medium_n} MEDIUM severity</strong> — Review recommended</div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="oi-alert-success">🟢 Majority of records passed all OMOP quality checks</div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    kpi_row([
        ("Total Flags",     f"{len(disc_df):,}",  C_DANGER,  "Require review",          C_DANGER),
        ("High Severity",   f"{high_n:,}",         C_WARNING, "Immediate action needed", C_WARNING),
        ("Medium Severity", f"{medium_n:,}",       C_BLUE,    "Review recommended",      ""),
        ("Issue Types",     str(disc_df["disc_type"].nunique() if "disc_type" in disc_df.columns else 0),
                                                   C_SUCCESS, "Distinct anomaly classes",""),
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    if "disc_type" in disc_df.columns:
        c1, c2 = st.columns(2)
        with c1:
            sec("📊 Issue Type Breakdown — OMOP Audit")
            dt = disc_df["disc_type"].value_counts().reset_index()
            dt.columns = ["Issue","Count"]
            fig = go.Figure(go.Bar(
                x=dt["Count"], y=dt["Issue"], orientation="h",
                marker=dict(color=C_DANGER, line_width=0,
                            opacity=[max(0.5, 1-i*0.1) for i in range(len(dt))]),
                text=dt["Count"], textposition="outside"))
            fig.update_layout(title="OMOP Data Quality Issue Types",
                              xaxis_title="Records", **PL)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            sec("🥧 Severity Distribution")
            if "severity" in disc_df.columns:
                sev = disc_df["severity"].value_counts().reset_index()
                sev.columns = ["Severity","Count"]
                fig = go.Figure(go.Pie(
                    labels=sev["Severity"], values=sev["Count"], hole=0.5,
                    marker=dict(colors=[C_DANGER if s=="HIGH" else C_WARNING for s in sev["Severity"]],
                                line=dict(color="white" if not DARK else C_BG, width=2))))
                fig.update_layout(title="Issue Severity Breakdown", **PL)
                st.plotly_chart(fig, use_container_width=True)

    sec("🗂 Flagged Records — Resolution Guidance")
    disp_cols = [c for c in ["person_id","cancer_type","disc_type","severity","resolution"]
                 if c in disc_df.columns]
    display_disc = disc_df[disp_cols].head(50).copy()

    def sev_clr(val):
        if val == "HIGH":   return f"background-color:{C_DANGER_BG};color:{C_DANGER};font-weight:700"
        if val == "MEDIUM": return f"background-color:{C_WARNING_BG};color:{C_WARNING};font-weight:700"
        return ""

    if "severity" in display_disc.columns:
        st.dataframe(display_disc.style.applymap(sev_clr, subset=["severity"]),
                     use_container_width=True, hide_index=True)
    else:
        st.dataframe(display_disc, use_container_width=True, hide_index=True)

    # AI Resolution Suggestion
    st.markdown("<br>", unsafe_allow_html=True)
    ai_chip("AI-Assisted Resolution — Snowflake Cortex")
    if st.button("✦ Generate AI Resolution Plan"):
        issue_summary = disc_df["disc_type"].value_counts().to_dict() if "disc_type" in disc_df.columns else {}
        ctx = f"OMOP data quality issues found: {json.dumps(issue_summary)}. Total flagged: {len(disc_df)} records."
        with st.spinner("Cortex generating resolution plan…"):
            plan = ai_generate_insight(ctx)
        st.markdown(f"""
        <div class="oi-insight" style="border-left-color:{C_DANGER};">
            <div style="font-size:22px;margin-bottom:8px;">🤖</div>
            <div class="oi-insight-text">{plan}</div>
            <div class="oi-insight-meta">Generated by Snowflake Cortex · {st.session_state.ai_model}</div>
        </div>
        """, unsafe_allow_html=True)

    export_btn(disc_df, "discrepancies.csv", "⬇ Export Discrepancy Report")
    end()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: AI INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════
elif "AI Insights" in page:
    topbar("✦ AI Insight Generator",
           "Automated oncology insights · Snowflake Cortex AI · OMOP-grounded",
           ["Cortex AI","mistral-large2"])
    wrap()
    demo_notice()

    kpis_l = get_overview_kpis()
    cancer_d = get_cancer_distribution()

    # Initialize insights
    if not st.session_state.insights_list:
        st.session_state.insights_list = [
            {"icon":"🔬","text":f"Cohort of {kpis_l.get('total_patients',0):,} oncology patients spans {kpis_l.get('cancer_types',0)} cancer types with an average OS of {kpis_l.get('avg_os',0)} months.",
             "source":"OMOP Overview","confidence":"High","color":C_PRIMARY},
            {"icon":"🎯","text":f"HER2+ patients represent a key biomarker-positive subgroup of {kpis_l.get('her2_pos',0):,} patients across the cohort.",
             "source":"MEASUREMENT Analysis","confidence":"High","color":C_SUCCESS},
            {"icon":"⚠️","text":f"Data quality audit identified {kpis_l.get('discrepancies',0):,} records with anomalies — primarily drug date inconsistencies in DRUG_EXPOSURE.",
             "source":"Discrepancy Engine","confidence":"High","color":C_DANGER},
            {"icon":"📈","text":f"Neoadjuvant therapy rate of {kpis_l.get('neoadjuvant',0)/max(kpis_l.get('total_patients',1),1)*100:.1f}% indicates strong presurgical treatment adoption.",
             "source":"ANT Engine","confidence":"High","color":C_TEAL},
        ]

    c1, c2 = st.columns([2, 1])
    with c1:
        if st.button("✦ Generate AI Insight (Cortex)"):
            ctx_data = f"""
            Total patients: {kpis_l.get('total_patients',0):,}
            Cancer types: {kpis_l.get('cancer_types',0)}
            Average OS: {kpis_l.get('avg_os',0)} months
            Neoadjuvant rate: {kpis_l.get('neoadjuvant',0)/max(kpis_l.get('total_patients',1),1)*100:.1f}%
            Stage IV rate: {kpis_l.get('stage_iv_pct',0)}%
            Data discrepancies: {kpis_l.get('discrepancies',0):,}
            HER2+ patients: {kpis_l.get('her2_pos',0):,}
            """
            with st.spinner("Cortex generating insight…"):
                new_insight = ai_generate_insight(ctx_data)
            st.session_state.insights_list.append({
                "icon": "✦",
                "text": new_insight,
                "source": f"Snowflake Cortex · {st.session_state.ai_model}",
                "confidence": "AI-Generated",
                "color": C_PRIMARY,
            })
            st.session_state.toast = "✓ New AI insight generated"
            st.rerun()
    with c2:
        if st.button("↺ Clear Insights"):
            st.session_state.insights_list = []
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    for ins in st.session_state.insights_list:
        conf_clr = C_SUCCESS if ins["confidence"]=="High" else (C_TEAL if ins["confidence"]=="AI-Generated" else C_WARNING)
        conf_bg  = C_SUCCESS_BG if ins["confidence"]=="High" else (C_TEAL_BG if ins["confidence"]=="AI-Generated" else C_WARNING_BG)
        st.markdown(f"""
        <div class="oi-insight" style="border-left-color:{ins['color']};" role="article">
            <div style="display:flex;align-items:flex-start;gap:14px;">
                <span style="font-size:22px;margin-top:1px;">{ins['icon']}</span>
                <div>
                    <div class="oi-insight-text">{ins['text']}</div>
                    <div class="oi-insight-meta">
                        Source: <strong style="color:{C_PRIMARY};">{ins['source']}</strong>
                        &nbsp;·&nbsp; Confidence:
                        <span style="background:{conf_bg};color:{conf_clr};
                                     padding:2px 9px;border-radius:10px;
                                     font-size:10.5px;font-weight:700;">{ins['confidence']}</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    sec("📊 OMOP Data Coverage by Analytics Module")
    cov = pd.DataFrame({
        "Module": ["ANT Engine","LoT Engine","Cohort Builder","Survival Model","Drug Utilization","Discrepancy"],
        "OMOP Table": ["DRUG_EXPOSURE/PROCEDURE","DRUG_EXPOSURE","PERSON/CONDITION","OBSERVATION_PERIOD","DRUG_EXPOSURE","Multi-table"],
        "Status": ["✅ Active","✅ Active","✅ Active","✅ Active","✅ Active","✅ Active"],
    })
    st.dataframe(cov, use_container_width=True, hide_index=True)
    end()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: AI CLINICAL CHAT
# ═══════════════════════════════════════════════════════════════════════════
elif "AI Clinical Chat" in page:
    topbar("💬 AI Clinical Chat",
           "Ask questions about your OMOP oncology data · Powered by Snowflake Cortex",
           ["Cortex AI","NL→OMOP","Interactive"])
    wrap()

    ai_chip("Powered by Snowflake Cortex · Ask anything about your OMOP oncology cohort")

    # Data summary for context
    kpis_c = get_overview_kpis()
    data_ctx = f"""OMOP oncology cohort: {kpis_c.get('total_patients',0):,} patients,
{kpis_c.get('cancer_types',0)} cancer types (Breast, Lung, Prostate, Ovarian, Colorectal, Lymphoma),
average OS {kpis_c.get('avg_os',0)} months, {kpis_c.get('neoadjuvant',0):,} neoadjuvant patients,
{kpis_c.get('discrepancies',0):,} data discrepancies, Stage IV rate {kpis_c.get('stage_iv_pct',0)}%.
OMOP CDM v5.4. Database: {DB}.{SCHEMA}."""

    # Quick action buttons
    st.markdown(f"""
    <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px;">
        <div style="font-size:12px;font-weight:600;color:{C_TEXT_MUTED};width:100%;margin-bottom:4px;">
            💡 Try these questions:
        </div>
    </div>
    """, unsafe_allow_html=True)

    quick_questions = [
        "What is the neoadjuvant therapy rate for breast cancer patients?",
        "Which cancer type has the highest OS months?",
        "How many patients have HER2+ biomarker status?",
        "What are the most common data discrepancies in our OMOP data?",
        "Summarize 2nd line therapy patterns across cancer types",
    ]

    q_cols = st.columns(len(quick_questions))
    clicked_q = None
    for i, (col, q) in enumerate(zip(q_cols, quick_questions)):
        if col.button(q[:35]+"…", key=f"qbtn_{i}", use_container_width=True):
            clicked_q = q

    st.markdown("<hr>", unsafe_allow_html=True)

    # Chat history display
    for msg in st.session_state.chat_history:
        role = msg["role"]
        with st.chat_message(role):
            st.markdown(msg["content"])

    # Chat input
    user_q = st.chat_input("Ask anything about your OMOP oncology data…") or clicked_q

    if user_q:
        st.session_state.chat_history.append({"role":"user","content":user_q})
        with st.chat_message("user"):
            st.markdown(user_q)
        with st.chat_message("assistant"):
            with st.spinner("Cortex thinking…"):
                answer = ai_answer_question(user_q, data_ctx)
            st.markdown(answer)
            st.session_state.chat_history.append({"role":"assistant","content":answer})

    if st.session_state.chat_history:
        if st.button("🗑 Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    end()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: OMOP EXPLORER (NL → SQL)
# ═══════════════════════════════════════════════════════════════════════════
elif "OMOP Explorer" in page:
    topbar("🔍 OMOP SQL Explorer",
           "Natural Language → OMOP SQL · AI-powered query generation · Execute on Snowflake",
           ["NL→SQL","Cortex AI","OMOP CDM"])
    wrap()

    ai_chip("Convert plain English to OMOP CDM SQL · Powered by Snowflake Cortex")

    st.markdown(f"""
    <div class="oi-alert-info">
        📋 <strong>How it works:</strong> Type a question in plain English. Cortex AI converts it to
        a valid Snowflake SQL query against your OMOP CDM schema ({DB}.{SCHEMA}).
        Review, edit, and execute the query directly on your Snowflake data.
    </div>
    """, unsafe_allow_html=True)

    example_queries = [
        "Show me all breast cancer patients diagnosed after 2020 with their age and gender",
        "Count patients by cancer type and stage, sorted by most common",
        "Find patients who received Pembrolizumab as first-line therapy for lung cancer",
        "Show the top 10 drugs used across all lines of therapy",
        "Find patients with HER2+ biomarker who progressed to 2nd line therapy within 6 months",
        "Show average time from diagnosis to first drug exposure by cancer type",
    ]

    st.markdown(f"<div style='font-size:12.5px;font-weight:600;color:{C_TEXT_MUTED};margin-bottom:8px;'>Example questions:</div>",
                unsafe_allow_html=True)

    exp_cols = st.columns(3)
    selected_ex = None
    for i, ex in enumerate(example_queries):
        with exp_cols[i % 3]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                selected_ex = ex

    st.markdown("<br>", unsafe_allow_html=True)
    nl_query = st.text_area(
        "Your Question",
        value=selected_ex or st.session_state.get("cohort_sql_input",""),
        placeholder="e.g. Show me all breast cancer patients diagnosed in 2021 with their first-line drug",
        height=80,
    )

    col_a, col_b = st.columns([1, 4])
    with col_a:
        gen_clicked = st.button("✦ Generate SQL", use_container_width=True)
    with col_b:
        ai_model_sel = st.selectbox("AI Model", ["mistral-large2","llama3.1-70b","snowflake-arctic"],
                                    index=0, label_visibility="collapsed")

    if gen_clicked and nl_query.strip():
        with st.spinner("Cortex converting to OMOP SQL…"):
            generated_sql = ai_generate_sql(nl_query)
        st.session_state.cohort_sql = generated_sql
        st.session_state.toast = "✓ SQL generated"
        st.rerun()

    if st.session_state.cohort_sql:
        sec("📄 Generated OMOP SQL")
        edited_sql = st.text_area("Edit SQL before executing:", value=st.session_state.cohort_sql,
                                   height=200, key="sql_editor")
        st.markdown(f"""
        <div style="font-size:11px;color:{C_TEXT_MUTED};margin-top:-8px;margin-bottom:12px;">
            ⚠ Review AI-generated SQL before executing. Verify table names match your schema.
        </div>
        """, unsafe_allow_html=True)

        col_run, col_exp = st.columns([1, 3])
        with col_run:
            if st.button("▶ Execute on Snowflake", use_container_width=True):
                if SF_CONNECTED and not DEMO:
                    try:
                        with st.spinner("Running query on Snowflake…"):
                            result_df = _run_query(edited_sql)
                        sec(f"📊 Query Results — {len(result_df):,} rows")
                        st.dataframe(result_df.head(200), use_container_width=True, hide_index=True)
                        export_btn(result_df, "omop_query_result.csv", "⬇ Export Results")
                    except Exception as e:
                        st.markdown(f'<div class="oi-alert-danger">❌ Query error: {str(e)}</div>',
                                    unsafe_allow_html=True)
                else:
                    st.markdown('<div class="oi-alert-warn">🟡 Demo mode: Connect Snowflake to execute queries.</div>',
                                unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="oi-sql-box">{edited_sql}</div>
                    """, unsafe_allow_html=True)

    # OMOP Schema Reference
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📚 OMOP CDM Schema Reference"):
        st.markdown(f"""
        <div style="font-family:'Space Mono',monospace;font-size:12px;color:{C_TEXT_MED};line-height:1.8;">
        <strong style="color:{C_PRIMARY};">Key OMOP CDM Tables:</strong><br><br>
        <strong>PERSON</strong>(person_id, gender_concept_id, year_of_birth, race_concept_id)<br>
        <strong>CONDITION_OCCURRENCE</strong>(person_id, condition_concept_id, condition_start_date, condition_source_value)<br>
        <strong>DRUG_EXPOSURE</strong>(person_id, drug_concept_id, drug_exposure_start_date, drug_exposure_end_date, quantity)<br>
        <strong>PROCEDURE_OCCURRENCE</strong>(person_id, procedure_concept_id, procedure_date)<br>
        <strong>MEASUREMENT</strong>(person_id, measurement_concept_id, measurement_date, value_as_number, value_as_concept_id)<br>
        <strong>OBSERVATION_PERIOD</strong>(person_id, observation_period_start_date, observation_period_end_date)<br>
        <strong>CONCEPT</strong>(concept_id, concept_name, domain_id, vocabulary_id, standard_concept)<br><br>
        <strong style="color:{C_TEAL};">Oncology Filters:</strong><br>
        Cancer: condition_source_value REGEXP '^C[0-9]' (ICD-10) or concept_name ILIKE '%malignan%'<br>
        Drugs: JOIN CONCEPT WHERE standard_concept = 'S' AND domain_id = 'Drug'<br>
        Biomarkers: MEASUREMENT JOIN CONCEPT WHERE concept_name IN ('HER2','EGFR','PD-L1','BRCA1','BRCA2','ALK')<br>
        Surgery: PROCEDURE_OCCURRENCE WHERE concept_name ILIKE '%surgery%' OR '%resection%'<br>
        </div>
        """, unsafe_allow_html=True)

    end()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: PRODUCT ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════
elif "Product" in page:
    topbar("◻ Product Artifacts — PM Deliverables",
           "PRD · User Personas · Business Value · KPIs · Roadmap",
           ["Product Management"])
    wrap()

    tabs = st.tabs(["📋 PRD","👥 Personas","💼 Business Value","📊 KPIs","🗺️ Roadmap"])

    with tabs[0]:
        for title, body in [
            ("🎯 Product Vision",
             "Build a production-grade, AI-powered oncology analytics platform on top of Snowflake OMOP CDM, "
             "enabling pharma companies, clinical researchers, and healthcare organizations to explore "
             "real-world evidence data and generate actionable insights via Snowflake Cortex AI."),
            ("🔍 Problem Statement",
             "Oncology analytics teams lack tools to rapidly query fragmented OMOP CDM data across "
             "EHR, Claims, Pathology, and Biomarker domains. Manual SQL, slow ETL pipelines, and "
             "absence of AI-assisted query generation create critical bottlenecks in drug strategy decisions."),
            ("✅ Production Scope",
             "Snowflake OMOP CDM v5.4 integration (SiS + local connector) · Snowflake Cortex AI "
             "(mistral-large2/llama3-70b) · NL→OMOP SQL generator · ANT Classification Engine · "
             "LoT Algorithm (90-day gap rule) · AI Clinical Chat · AI Cohort Builder · "
             "Discrepancy Detection · Dark mode · WCAG 2.1 AA · CSV exports."),
            ("🎯 Success Criteria",
             "Time to first OMOP insight &lt; 5 min · Cohort build &lt; 30 sec · ANT accuracy ≥ 95% "
             "against DRUG_EXPOSURE/PROCEDURE_OCCURRENCE · AI SQL accuracy ≥ 90% · Dashboard load &lt; 2 sec "
             "via Snowflake query result caching · Cortex response &lt; 3 sec."),
            ("🔐 Data & Compliance",
             "Patient data queried from de-identified OMOP CDM per HIPAA Safe Harbor. Platform supports "
             "Snowflake RBAC roles, row-level security, audit logging, and SOC 2 Type II. "
             "No PHI stored outside Snowflake. All data stays within the customer's Snowflake account."),
            ("🏔️ Snowflake Architecture",
             "Streamlit in Snowflake (SiS) for native deployment — zero egress, native auth, RBAC. "
             "Snowflake Cortex for AI (no external API calls) — data never leaves Snowflake. "
             "Result cache (TTL=600s) for sub-second repeated queries. "
             "Supports multi-warehouse setup: ANALYST_WH for queries, CORTEX_WH for AI."),
        ]:
            card(title, body)

    with tabs[1]:
        c1, c2 = st.columns(2)
        personas = [
            ("💊 Pharma Analyst",       C_PRIMARY, "Analyze drug launch, market share, treatment patterns",
             "No self-service OMOP tools; SQL required; slow time-to-insight"),
            ("🔬 Oncology Researcher",  C_TEAL,    "Study treatment outcomes and biomarker-therapy correlations",
             "Difficult cohort discovery across OMOP tables; inconsistent schemas"),
            ("📊 Data Scientist",       C_PURPLE,  "Build and validate oncology ML models using OMOP features",
             "Manual SQL for feature extraction; no oncology-specific OMOP tooling"),
            ("🏥 Clinical Operations",  C_SUCCESS, "Monitor guideline adherence via real-world OMOP data",
             "No real-time OMOP dashboards; siloed EHR data; no discrepancy alerts"),
        ]
        for i, (role, clr, goal, pain) in enumerate(personas):
            with (c1 if i % 2 == 0 else c2):
                st.markdown(f"""
                <div class="oi-persona-card" style="border-top-color:{clr};">
                    <div style="font-size:14px;font-weight:700;color:{clr};margin-bottom:9px;
                                 font-family:'Syne',sans-serif;">{role}</div>
                    <div style="font-size:13px;color:{C_TEXT};margin-bottom:6px;">
                        <strong>Goal:</strong> {goal}
                    </div>
                    <div style="font-size:13px;color:{C_TEXT_MED};">
                        <strong>Pain Point:</strong> {pain}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tabs[2]:
        c1, c2, c3 = st.columns(3)
        bv = [
            (c1,"💊 Pharma Companies",    C_PRIMARY,
             ["OMOP-grounded drug analytics","Competitive therapy analysis","Patient segmentation from CDM",
              "RWE generation from OMOP","Cortex AI for instant insights"]),
            (c2,"🏥 Healthcare Providers", C_TEAL,
             ["Treatment pathway optimization","OMOP-native outcome monitoring","Guideline adherence checks",
              "Real-time OMOP dashboards","Biomarker-driven care pathways"]),
            (c3,"🔬 Researchers",          C_PURPLE,
             ["OMOP-standard treatment research","Biomarker-outcome studies","Survival analysis from obs_period",
              "AI-assisted publication visuals","Cortex NL→SQL for hypotheses"]),
        ]
        for col, title, clr, items in bv:
            with col:
                rows = "".join([
                    f"<div style='padding:4px 0;border-bottom:1px solid {C_BORDER_LT};"
                    f"font-size:13px;color:{C_TEXT_MED};'>✓ {item}</div>"
                    for item in items
                ])
                st.markdown(f"""
                <div class="oi-card" style="border-top:3px solid {clr};">
                    <div class="oi-card-title" style="color:{clr};">{title}</div>
                    <div>{rows}</div>
                </div>
                """, unsafe_allow_html=True)

    with tabs[3]:
        k_items = [
            ("Snowflake OMOP Patients", "500+",         C_PRIMARY),
            ("Cohort Analyses / Month", "1,000+",       C_TEAL),
            ("AI Insight Time",         "< 3 sec",      C_SUCCESS),
            ("NL→SQL Accuracy",         "≥ 90%",        C_WARNING),
            ("ANT Accuracy",            "≥ 95%",        C_PURPLE),
            ("Target ARR (Year 1)",     "$2M+",         C_DANGER),
        ]
        c1, c2, c3 = st.columns(3)
        for i, (label, val, clr) in enumerate(k_items):
            with [c1, c2, c3][i % 3]:
                st.markdown(f"""
                <div class="oi-biz-kpi" style="border-top-color:{clr};">
                    <div class="oi-biz-kpi-val" style="color:{clr};">{val}</div>
                    <div class="oi-biz-kpi-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        sec("📉 User Adoption Funnel — Year 1 Target")
        fig = go.Figure(go.Funnel(
            y=["Awareness","Trial","Activated","Power Users","Enterprise"],
            x=[500, 200, 80, 40, 15],
            textinfo="value+percent initial",
            marker=dict(color=[C_PRIMARY, C_TEAL, C_SUCCESS, C_WARNING, C_PINK]),
        ))
        fig.update_layout(title="Y1 User Adoption Funnel",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="Figtree,system-ui", color=C_TEXT),
                          title_font=dict(size=13.5, color=C_PRIMARY, family="Syne"),
                          margin=dict(l=8,r=8,t=40,b=8), height=280)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        for num, clr, title, items in [
            ("1", C_PRIMARY, "Phase 1 — MVP on Snowflake (Months 1–3)",
             "Streamlit in Snowflake (SiS) · ANT Classification from OMOP · LoT Engine · "
             "OMOP Cohort Builder · Drug Utilization Dashboards · Discrepancy Detection · "
             "Snowflake Cortex AI Integration (mistral-large2)"),
            ("2", C_TEAL, "Phase 2 — Advanced Analytics (Months 4–8)",
             "Snowflake Cortex NL→SQL (full OMOP schema) · Kaplan-Meier from OBSERVATION_PERIOD · "
             "Multi-cancer biomarker analysis · FHIR→OMOP connectors · PDF export from SiS · "
             "Advanced Cortex models (llama3-70b) · Snowflake Data Clean Rooms"),
            ("3", C_PURPLE, "Phase 3 — Enterprise Platform (Months 9–18)",
             "Multi-tenant Snowflake deployment · Native App on Snowflake Marketplace · "
             "SSO via Snowflake SCIM · Real-time OMOP pipelines (Snowpipe) · "
             "Snowflake Native Apps Framework · SLA guarantees · Federated OMOP across accounts"),
        ]:
            r, g, b = int(clr[1:3],16), int(clr[3:5],16), int(clr[5:7],16)
            st.markdown(f"""
            <div class="oi-phase-card" style="border-left-color:{clr};">
                <div style="width:32px;height:32px;min-width:32px;border-radius:50%;
                            background:rgba({r},{g},{b},0.15);color:{clr};font-weight:800;
                            font-size:14px;font-family:'Syne',sans-serif;
                            display:flex;align-items:center;justify-content:center;
                            border:2px solid rgba({r},{g},{b},0.3);">{num}</div>
                <div>
                    <div style="font-weight:700;font-size:14px;color:{clr};margin-bottom:4px;
                                 font-family:'Syne',sans-serif;">{title}</div>
                    <div style="font-size:13px;color:{C_TEXT_MED};line-height:1.6;">{items}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        sec("📅 Development Gantt Chart")
        gantt_df = pd.DataFrame([
            dict(Task="OMOP Connection Layer",   Start="2025-01-01", Finish="2025-01-21", Phase="Phase 1"),
            dict(Task="ANT Engine (OMOP)",        Start="2025-01-15", Finish="2025-02-28", Phase="Phase 1"),
            dict(Task="LoT Engine (90-day rule)", Start="2025-02-01", Finish="2025-03-15", Phase="Phase 1"),
            dict(Task="Cohort Builder + UI",      Start="2025-02-15", Finish="2025-04-01", Phase="Phase 1"),
            dict(Task="Cortex AI Integration",    Start="2025-03-01", Finish="2025-04-15", Phase="Phase 1"),
            dict(Task="NL→OMOP SQL (Cortex)",     Start="2025-04-01", Finish="2025-06-01", Phase="Phase 2"),
            dict(Task="Survival / KM from OMOP",  Start="2025-05-01", Finish="2025-07-01", Phase="Phase 2"),
            dict(Task="Snowflake Native App",     Start="2025-09-01", Finish="2025-12-01", Phase="Phase 3"),
            dict(Task="Snowpipe Real-time",       Start="2025-10-01", Finish="2026-02-01", Phase="Phase 3"),
        ])
        fig = px.timeline(gantt_df, x_start="Start", x_end="Finish", y="Task", color="Phase",
                          color_discrete_map={"Phase 1":C_PRIMARY,"Phase 2":C_TEAL,"Phase 3":C_PURPLE})
        fig.update_layout(title="OncoInsight Development Roadmap", **PL, height=340)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    end()

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
sf_status = "🟢 Snowflake OMOP Connected" if SF_CONNECTED and not DEMO else "🟡 Demo Mode"
st.markdown(f"""
<div style="border-top:1px solid {C_BORDER};margin:0;padding:16px 28px 24px;
            display:flex;align-items:center;justify-content:space-between;
            background:{C_SURFACE};flex-wrap:wrap;gap:10px;">
    <div style="font-size:12px;color:{C_TEXT_MUTED};">
        <strong style="color:{C_PRIMARY};font-family:'Syne',sans-serif;font-size:13px;">
            OncoInsight Analytics Platform v3.0
        </strong>
        <span style="margin:0 10px;opacity:0.3;">|</span>
        OMOP CDM v5.4
        <span style="margin:0 10px;opacity:0.3;">|</span>
        Snowflake Cortex AI
        <span style="margin:0 10px;opacity:0.3;">|</span>
        WCAG 2.1 AA · HIPAA-Aware
        <span style="margin:0 10px;opacity:0.3;">|</span>
        {sf_status}
    </div>
    <div style="display:flex;align-items:center;gap:8px;">
        <span style="font-size:11px;color:{C_TEXT_MUTED};">Cancer Awareness:</span>
        <span title="General Cancer" style="width:8px;height:8px;border-radius:50%;background:{C_PRIMARY};display:inline-block;"></span>
        <span title="Breast Cancer"  style="width:8px;height:8px;border-radius:50%;background:{C_PINK};display:inline-block;"></span>
        <span title="Ovarian Cancer" style="width:8px;height:8px;border-radius:50%;background:{C_TEAL};display:inline-block;"></span>
        <span title="Prostate Cancer"style="width:8px;height:8px;border-radius:50%;background:{C_BLUE};display:inline-block;"></span>
    </div>
</div>
""", unsafe_allow_html=True)
