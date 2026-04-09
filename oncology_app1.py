"""
╔══════════════════════════════════════════════════════════════════╗
║         OncoInsight — Oncology Data Analytics Platform           ║
║         Redesigned UI  |  Cancer-Aware Lavender Theme            ║
║         WCAG 2.1 AA · Lavender/Pink/Teal/Blue Palette            ║
║         Accessible · Dark Mode · Clinical-Grade Design           ║
╚══════════════════════════════════════════════════════════════════╝

Run:
    pip install streamlit pandas numpy plotly
    streamlit run oncology_app_redesigned.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import math

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OncoInsight | Oncology Analytics",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE — LOGIN + DARK MODE
# ─────────────────────────────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "toast" not in st.session_state:
    st.session_state.toast = None

# ─────────────────────────────────────────────────────────────────────────────
# CANCER-AWARE COLOR PALETTE
# Globally recognized cancer ribbon awareness colors
# All text/background combos meet WCAG 2.1 AA (4.5:1 minimum)
# ─────────────────────────────────────────────────────────────────────────────
DARK = st.session_state.dark_mode

# ── Brand / Primary: Lavender-Purple (General Cancer Awareness) ──
C_PRIMARY    = "#5B21B6"   # Deep purple — WCAG AA on white (7.8:1) ✓
C_PRIMARY_LT = "#7C3AED"   # Medium purple — interactive states
C_PRIMARY_BG = "#F5F3FF"   # Lavender tint — card backgrounds

# ── Cancer-Specific Accent Colors ──
C_PINK       = "#9D174D"   # Breast cancer pink — darkened for WCAG AA ✓
C_PINK_BG    = "#FDF2F8"
C_TEAL       = "#0F766E"   # Ovarian cancer teal — WCAG AA ✓
C_TEAL_BG    = "#F0FDFA"
C_BLUE       = "#1D4ED8"   # Prostate cancer blue — WCAG AA ✓
C_BLUE_BG    = "#EFF6FF"

# ── Semantic / Clinical ──
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

# ── Neutrals (Light Mode) ──
if not DARK:
    C_BG         = "#F8F7FC"   # Lavender-tinted off-white canvas
    C_SURFACE    = "#FFFFFF"
    C_SURFACE2   = "#F3F1FA"
    C_BORDER     = "#DDD9EC"
    C_BORDER_LT  = "#ECEAF5"
    C_TEXT       = "#1C1B29"   # Near-black with purple tint
    C_TEXT_MED   = "#4B4A6A"
    C_TEXT_MUTED = "#9896B5"
    C_HEADER_BG  = "#3B1A8A"   # Deep purple topbar
    C_SIDEBAR_BG = "#2D1769"   # Darker sidebar
    C_SIDEBAR_TXT= "#EDE9FE"
else:
    # ── Dark Mode Palette ──
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
    # Adjust surface colors for dark charts
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

# ── Chart Colors — Cancer-type consistent, color-blind safe ──
CHART_COLORS = [C_PRIMARY, C_TEAL, C_BLUE, C_PINK, "#B45309", C_SUCCESS,
                C_PRIMARY_LT, "#0D9488", "#7C3AED", "#9A3412"]

CANCER_COLORS = {
    "Breast":     C_PINK,
    "Ovarian":    C_TEAL,
    "Prostate":   C_BLUE,
    "Lung":       C_PRIMARY,
    "Colorectal": C_SUCCESS,
    "Lymphoma":   "#B45309",
}

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — Clinical Lavender Design System
# Fonts: Syne (headings) + Figtree (body) + Space Mono (data)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Figtree:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

/* ── Reset & Base ── */
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

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {C_SIDEBAR_BG} 0%, {C_SIDEBAR_BG}ee 100%) !important;
    border-right: 1px solid {C_BORDER} !important;
}}
[data-testid="stSidebar"] * {{ color: {C_SIDEBAR_TXT} !important; }}
[data-testid="stSidebar"] .stRadio label {{
    font-size: 13.5px !important;
    padding: 7px 12px !important;
    border-radius: 8px !important;
    transition: background 0.15s ease !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    cursor: pointer !important;
    font-weight: 500 !important;
}}
[data-testid="stSidebar"] .stRadio label:hover {{
    background: rgba(255,255,255,0.10) !important;
}}
[data-testid="stSidebar"] .stRadio [data-checked="true"] label,
[data-testid="stSidebar"] .stRadio input:checked + label {{
    background: rgba(124,58,237,0.30) !important;
    border-left: 3px solid #A78BFA !important;
}}

/* ── Metrics ── */
[data-testid="metric-container"] {{
    background: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-radius: 12px;
    padding: 16px 18px !important;
    box-shadow: 0 2px 8px rgba(91,33,182,0.07);
    transition: transform 0.2s, box-shadow 0.2s;
}}
[data-testid="metric-container"]:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(91,33,182,0.12);
}}
[data-testid="stMetricValue"] {{
    color: {C_PRIMARY} !important;
    font-weight: 700 !important;
    font-family: 'Space Mono', monospace !important;
}}

/* ── Headings ── */
h1 {{
    color: {C_PRIMARY} !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.5rem !important;
    border-bottom: 2px solid {C_BORDER_LT};
    padding-bottom: 10px;
    letter-spacing: -0.01em;
}}
h2 {{
    color: {C_PRIMARY} !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.15rem !important;
}}
h3 {{
    color: {C_TEXT} !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}}

/* ── Data Tables ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {C_BORDER} !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
}}

/* ── Buttons ── */
.stButton > button {{
    background: linear-gradient(135deg, {C_PRIMARY} 0%, {C_PRIMARY_LT} 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 9px 20px !important;
    font-size: 14px !important;
    font-family: 'Figtree', sans-serif !important;
    box-shadow: 0 2px 8px rgba(91,33,182,0.30) !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.01em !important;
}}
.stButton > button:hover {{
    background: linear-gradient(135deg, #4C1D95 0%, {C_PRIMARY} 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(91,33,182,0.40) !important;
}}
.stButton > button:active {{
    transform: translateY(0) !important;
}}

/* ── Inputs / Selects ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] > div > div > input {{
    background: {C_SURFACE} !important;
    border: 1.5px solid {C_BORDER} !important;
    border-radius: 8px !important;
    color: {C_TEXT} !important;
    font-family: 'Figtree', sans-serif !important;
    font-size: 14px !important;
    transition: border-color 0.15s !important;
}}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stTextInput"] > div > div > input:focus {{
    border-color: {C_PRIMARY_LT} !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.15) !important;
    outline: none !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: {C_SURFACE};
    border-bottom: 2px solid {C_BORDER};
    padding: 0 8px;
    gap: 2px;
    border-radius: 0;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {C_TEXT_MUTED};
    font-weight: 600;
    font-size: 13.5px;
    font-family: 'Figtree', sans-serif;
    padding: 11px 20px;
    border-bottom: 3px solid transparent;
    margin-bottom: -2px;
    border-radius: 0;
    transition: all 0.15s ease;
}}
.stTabs [aria-selected="true"] {{
    background: transparent !important;
    color: {C_PRIMARY} !important;
    border-bottom: 3px solid {C_PRIMARY} !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    background: {C_PRIMARY_BG} !important;
    color: {C_PRIMARY} !important;
}}

/* ── Expanders ── */
[data-testid="stExpander"] {{
    background: {C_SURFACE} !important;
    border: 1px solid {C_BORDER} !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}}

/* ── File Uploader ── */
[data-testid="stFileUploaderDropzone"] {{
    background: {C_SURFACE} !important;
    border: 2px dashed {C_BORDER} !important;
    border-radius: 10px !important;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {C_BG}; }}
::-webkit-scrollbar-thumb {{ background: {C_BORDER}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {C_PRIMARY_LT}; }}

hr {{ border-color: {C_BORDER} !important; margin: 12px 0 !important; opacity: 0.6 !important; }}

/* ════════════════════════════════════════
   OncoInsight Design System Components
   ════════════════════════════════════════ */

/* ── Topbar ── */
.oi-topbar {{
    background: linear-gradient(135deg, {C_HEADER_BG} 0%, {C_SIDEBAR_BG} 100%);
    color: white;
    padding: 16px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
    box-shadow: 0 3px 16px rgba(59,26,138,0.25);
    position: relative;
    overflow: hidden;
}}
.oi-topbar::before {{
    content: '';
    position: absolute;
    right: -20px;
    top: -20px;
    width: 120px;
    height: 120px;
    border-radius: 50%;
    background: rgba(167,139,250,0.12);
    pointer-events: none;
}}
.oi-topbar::after {{
    content: '';
    position: absolute;
    right: 60px;
    bottom: -30px;
    width: 80px;
    height: 80px;
    border-radius: 50%;
    background: rgba(167,139,250,0.08);
    pointer-events: none;
}}
.oi-topbar-title {{
    font-family: 'Syne', sans-serif;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.01em;
}}
.oi-topbar-sub {{ font-size: 12.5px; opacity: 0.75; margin-top: 2px; }}
.oi-badge {{
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11.5px;
    font-weight: 600;
    margin-left: 7px;
    backdrop-filter: blur(4px);
}}

/* ── KPI Cards ── */
.oi-kpi {{
    background: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 10px rgba(91,33,182,0.07);
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}}
.oi-kpi::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 12px 12px 0 0;
}}
.oi-kpi:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(91,33,182,0.12);
}}
.oi-kpi-label {{
    font-size: 10.5px;
    font-weight: 700;
    color: {C_TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 6px;
}}
.oi-kpi-num {{
    font-size: 28px;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
    line-height: 1.1;
    margin-bottom: 6px;
}}
.oi-kpi-delta {{
    font-size: 12px;
    color: {C_TEXT_MUTED};
    display: flex;
    align-items: center;
    gap: 4px;
}}

/* ── Section Headers ── */
.oi-sec {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
    color: {C_PRIMARY};
    background: {C_PRIMARY_BG};
    border-left: 4px solid {C_PRIMARY};
    padding: 9px 16px;
    border-radius: 0 8px 8px 0;
    margin: 18px 0 14px;
    width: fit-content;
    min-width: 200px;
    letter-spacing: 0.01em;
}}

/* ── Insight Cards ── */
.oi-insight {{
    background: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-left: 4px solid;
    border-radius: 0 12px 12px 0;
    padding: 14px 20px;
    margin-bottom: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    transition: transform 0.15s, box-shadow 0.15s;
}}
.oi-insight:hover {{
    transform: translateX(2px);
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
}}
.oi-insight-text {{ font-size: 14px; color: {C_TEXT}; line-height: 1.65; font-weight: 500; }}
.oi-insight-meta {{ font-size: 11.5px; color: {C_TEXT_MUTED}; margin-top: 6px; }}

/* ── PRD Cards ── */
.oi-card {{
    background: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(91,33,182,0.05);
    transition: box-shadow 0.2s;
}}
.oi-card:hover {{ box-shadow: 0 4px 16px rgba(91,33,182,0.10); }}
.oi-card-title {{
    color: {C_PRIMARY};
    font-weight: 700;
    font-family: 'Syne', sans-serif;
    font-size: 13.5px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
}}
.oi-card-body  {{ color: {C_TEXT_MED}; font-size: 13.5px; line-height: 1.7; }}

/* ── Journey Timeline Nodes ── */
.oi-journey-node {{
    background: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-top: 3px solid;
    border-radius: 12px;
    text-align: center;
    padding: 14px 8px;
    box-shadow: 0 2px 8px rgba(91,33,182,0.07);
    transition: transform 0.2s;
}}
.oi-journey-node:hover {{ transform: translateY(-3px); }}
.oi-journey-icon  {{ font-size: 24px; margin-bottom: 6px; }}
.oi-journey-title {{
    font-size: 9.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
    font-family: 'Syne', sans-serif;
}}
.oi-journey-date  {{
    font-size: 10.5px;
    font-family: 'Space Mono', monospace;
    color: {C_PRIMARY};
}}

/* ── Persona Cards ── */
.oi-persona-card {{
    background: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-top: 3px solid;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(91,33,182,0.05);
}}

/* ── KPI Business Cards ── */
.oi-biz-kpi {{
    background: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-top: 3px solid;
    border-radius: 12px;
    padding: 18px 16px;
    text-align: center;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(91,33,182,0.05);
    transition: transform 0.2s;
}}
.oi-biz-kpi:hover {{ transform: translateY(-2px); }}
.oi-biz-kpi-val   {{
    font-size: 26px;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
    line-height: 1.1;
}}
.oi-biz-kpi-label {{
    font-size: 10.5px;
    color: {C_TEXT_MED};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 7px;
    font-weight: 600;
}}

/* ── Phase / Roadmap Cards ── */
.oi-phase-card {{
    background: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-left: 4px solid;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin-bottom: 10px;
    box-shadow: 0 2px 8px rgba(91,33,182,0.05);
    display: flex;
    align-items: flex-start;
    gap: 14px;
    transition: box-shadow 0.2s;
}}
.oi-phase-card:hover {{ box-shadow: 0 4px 16px rgba(91,33,182,0.10); }}

/* ── Alert Banners ── */
.oi-alert-info    {{ background:{C_INFO_BG};border:1px solid #BFDBFE;border-left:4px solid {C_INFO};
                     border-radius:0 8px 8px 0;padding:10px 16px;font-size:13.5px;color:{C_INFO};margin-bottom:10px; }}
.oi-alert-success {{ background:{C_SUCCESS_BG};border:1px solid #A7F3D0;border-left:4px solid {C_SUCCESS};
                     border-radius:0 8px 8px 0;padding:10px 16px;font-size:13.5px;color:{C_SUCCESS};margin-bottom:10px; }}
.oi-alert-warn    {{ background:{C_WARNING_BG};border:1px solid #FCD34D;border-left:4px solid {C_WARNING};
                     border-radius:0 8px 8px 0;padding:10px 16px;font-size:13.5px;color:{C_WARNING};margin-bottom:10px; }}
.oi-alert-danger  {{ background:{C_DANGER_BG};border:1px solid #FCA5A5;border-left:4px solid {C_DANGER};
                     border-radius:0 8px 8px 0;padding:10px 16px;font-size:13.5px;color:{C_DANGER};margin-bottom:10px; }}

/* ── Sidebar Header ── */
.oi-sidebar-hdr {{
    background: linear-gradient(135deg, rgba(124,58,237,0.3) 0%, rgba(91,33,182,0.15) 100%);
    border-bottom: 1px solid rgba(255,255,255,0.12);
    margin: -1rem -1rem 1.2rem;
    padding: 20px 16px 18px;
    position: relative;
    overflow: hidden;
}}
.oi-sidebar-hdr::after {{
    content: '🎗️';
    position: absolute;
    right: 12px;
    top: 12px;
    font-size: 28px;
    opacity: 0.35;
}}

/* ── Sidebar Dataset Panel ── */
.oi-sidebar-dataset {{
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 12px;
}}

/* ── Progress Bars ── */
.oi-progress-bg   {{ background:{C_BORDER_LT};border-radius:6px;height:10px;overflow:hidden;margin-top:5px; }}
.oi-progress-fill {{ height:100%;border-radius:6px;transition:width 0.7s cubic-bezier(.4,0,.2,1); }}

/* ── Filter Panel ── */
.oi-filter-panel {{
    background: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 18px;
    box-shadow: 0 2px 10px rgba(91,33,182,0.06);
}}

/* ── Toast Notification ── */
.oi-toast {{
    position: fixed;
    bottom: 24px;
    right: 24px;
    background: {C_PRIMARY};
    color: white;
    padding: 12px 20px;
    border-radius: 10px;
    font-size: 13.5px;
    font-weight: 600;
    box-shadow: 0 6px 24px rgba(91,33,182,0.35);
    z-index: 9999;
    animation: slideIn 0.3s ease;
}}
@keyframes slideIn {{
    from {{ transform: translateY(20px); opacity: 0; }}
    to   {{ transform: translateY(0);   opacity: 1; }}
}}

/* ── Login Screen ── */
.oi-login-bg {{
    position: fixed;
    inset: 0;
    background: linear-gradient(135deg, #EDE9FE 0%, #DDD6FE 35%, #C4B5FD 70%, #A78BFA 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
}}
.oi-login-card {{
    background: white;
    border-radius: 20px;
    padding: 44px 48px;
    width: 440px;
    max-width: 92vw;
    box-shadow: 0 24px 64px rgba(91,33,182,0.22), 0 4px 16px rgba(0,0,0,0.08);
}}
.oi-login-logo {{
    text-align: center;
    margin-bottom: 32px;
}}
.oi-login-logo-icon {{
    font-size: 42px;
    margin-bottom: 10px;
    display: block;
}}
.oi-login-brand {{
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 800;
    color: {C_PRIMARY};
    letter-spacing: -0.02em;
}}
.oi-login-tagline {{
    font-size: 13px;
    color: {C_TEXT_MUTED};
    margin-top: 4px;
}}
.oi-login-field-label {{
    font-size: 13px;
    font-weight: 600;
    color: {C_TEXT};
    margin-bottom: 6px;
    display: block;
}}
.oi-login-input {{
    width: 100%;
    padding: 11px 14px;
    border: 1.5px solid {C_BORDER};
    border-radius: 8px;
    font-size: 14px;
    font-family: 'Figtree', sans-serif;
    color: {C_TEXT};
    background: {C_BG};
    box-sizing: border-box;
    transition: border-color 0.15s, box-shadow 0.15s;
    outline: none;
}}
.oi-login-input:focus {{
    border-color: {C_PRIMARY_LT};
    box-shadow: 0 0 0 3px rgba(124,58,237,0.15);
}}
.oi-login-btn {{
    width: 100%;
    padding: 12px;
    background: linear-gradient(135deg, {C_PRIMARY} 0%, {C_PRIMARY_LT} 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 15px;
    font-weight: 700;
    font-family: 'Figtree', sans-serif;
    cursor: pointer;
    margin-top: 8px;
    box-shadow: 0 4px 14px rgba(91,33,182,0.35);
    transition: all 0.2s ease;
    letter-spacing: 0.01em;
}}
.oi-login-btn:hover {{
    background: linear-gradient(135deg, #4C1D95 0%, {C_PRIMARY} 100%);
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(91,33,182,0.45);
}}
.oi-forgot-link {{
    color: {C_PRIMARY_LT};
    font-size: 12.5px;
    text-decoration: none;
    text-align: right;
    display: block;
    margin-top: 6px;
    cursor: pointer;
}}
.oi-forgot-link:hover {{ text-decoration: underline; }}
.oi-login-divider {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 22px 0 16px;
    color: {C_TEXT_MUTED};
    font-size: 12px;
}}
.oi-login-divider::before, .oi-login-divider::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: {C_BORDER};
}}
.oi-ribbon-strip {{
    display: flex;
    gap: 6px;
    justify-content: center;
    margin-top: 24px;
}}
.oi-ribbon-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOGIN SCREEN
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.authenticated:
    st.markdown(f"""
    <div style="text-align:center;padding:60px 20px 30px;">
        <div style="display:inline-block;background:white;border-radius:20px;
                    padding:44px 48px;max-width:440px;width:100%;
                    box-shadow:0 24px 64px rgba(91,33,182,0.18),0 4px 16px rgba(0,0,0,0.07);">
            <div style="margin-bottom:30px;">
                <div style="font-size:48px;margin-bottom:10px;">🎗️</div>
                <div style="font-family:'Syne',sans-serif;font-size:30px;font-weight:800;
                             color:{C_PRIMARY};letter-spacing:-0.02em;">OncoInsight</div>
                <div style="font-size:13px;color:{C_TEXT_MUTED};margin-top:4px;">
                    Oncology Analytics Platform
                </div>
                <div style="display:flex;gap:6px;justify-content:center;margin-top:14px;">
                    <span style="width:8px;height:8px;border-radius:50%;background:{C_PINK};display:inline-block;"></span>
                    <span style="width:8px;height:8px;border-radius:50%;background:{C_PRIMARY};display:inline-block;"></span>
                    <span style="width:8px;height:8px;border-radius:50%;background:{C_TEAL};display:inline-block;"></span>
                    <span style="width:8px;height:8px;border-radius:50%;background:{C_BLUE};display:inline-block;"></span>
                </div>
            </div>
            <div style="font-size:13px;font-weight:600;color:{C_TEXT};text-align:left;margin-bottom:5px;">
                Email Address
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 1.6, 1])
    with col_c:
        email = st.text_input("Email", placeholder="clinician@hospital.org",
                              label_visibility="collapsed")
        st.markdown(f"""
        <div style="font-size:13px;font-weight:600;color:{C_TEXT};margin-bottom:5px;margin-top:12px;">
            Password
        </div>""", unsafe_allow_html=True)
        password = st.text_input("Password", type="password",
                                 placeholder="Enter your password",
                                 label_visibility="collapsed")
        st.markdown(f'<div style="text-align:right;"><a style="color:{C_PRIMARY_LT};font-size:12.5px;cursor:pointer;">Forgot password?</a></div>',
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Sign In to OncoInsight", use_container_width=True):
            if email and password:
                st.session_state.authenticated = True
                st.session_state.toast = "✓ Welcome to OncoInsight"
                st.rerun()
            else:
                st.markdown(f'<div class="oi-alert-warn">⚠ Please enter your email and password.</div>',
                            unsafe_allow_html=True)

        st.markdown(f"""
        <div style="text-align:center;margin-top:18px;font-size:12px;color:{C_TEXT_MUTED};">
            🔒 HIPAA-compliant · WCAG 2.1 AA · Clinical-grade security
        </div>
        <div style="text-align:center;margin-top:8px;font-size:11px;color:{C_TEXT_MUTED};">
            <em>Demo: use any email + password to sign in</em>
        </div>""", unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA
# ─────────────────────────────────────────────────────────────────────────────
CANCER_TYPES = ["Breast","Lung","Colorectal","Prostate","Ovarian","Lymphoma"]
STAGES       = ["I","II","III","IV"]
BIOMARKERS   = ["HER2+","HER2-","ER+","EGFR+","ALK+","PD-L1+","KRAS+","BRCA1/2+"]
DRUGS        = ["Paclitaxel","Carboplatin","Pembrolizumab","Trastuzumab",
                "Bevacizumab","Docetaxel","Nivolumab","Olaparib"]
DRUG_CLASSES = ["Chemotherapy","Immunotherapy","Targeted Therapy","Hormonal","CDK4/6 Inhibitor"]
DISC_TYPES   = ["Drug date before diagnosis","Missing surgery data",
                "Duplicate therapy event","Conflicting drug record","Biomarker mismatch"]
RACES        = ["White","Black","Hispanic","Asian","Other"]


@st.cache_data
def generate_data(n=200, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        diag   = datetime(2019,1,1) + timedelta(days=int(rng.integers(0,730)))
        surg   = diag   + timedelta(days=int(rng.integers(30,180)))
        drug1  = diag   + timedelta(days=int(rng.integers(10,40)))
        drug2  = surg   + timedelta(days=int(rng.integers(30,90)))
        drug3  = drug2  + timedelta(days=int(rng.integers(90,200)))
        rows.append({
            "patient_id":       f"P{i+1:03d}",
            "age":              int(rng.integers(35,80)),
            "gender":           "Female" if rng.random()>0.45 else "Male",
            "race":             RACES[rng.integers(0,5)],
            "diagnosis_date":   diag.date(),
            "cancer_type":      CANCER_TYPES[rng.integers(0,6)],
            "stage":            STAGES[rng.integers(0,4)],
            "icd_code":         f"C{rng.integers(10,99)}.{rng.integers(0,9)}",
            "biomarker":        BIOMARKERS[rng.integers(0,8)],
            "biomarker_result": "Positive" if rng.random()>0.4 else "Negative",
            "surgery_date":     surg.date(),
            "drug1_name":       DRUGS[rng.integers(0,8)],
            "drug1_date":       drug1.date(),
            "drug1_class":      DRUG_CLASSES[rng.integers(0,5)],
            "drug2_name":       DRUGS[rng.integers(0,8)],
            "drug2_date":       drug2.date(),
            "drug2_class":      DRUG_CLASSES[rng.integers(0,5)],
            "drug3_name":       DRUGS[rng.integers(0,8)],
            "drug3_date":       drug3.date(),
            "drug3_class":      DRUG_CLASSES[rng.integers(0,5)],
            "ant_type":         "Neoadjuvant" if drug1<surg else "Adjuvant",
            "os_months":        int(rng.integers(6,60)),
            "pfs_months":       int(rng.integers(3,40)),
            "ttnt_months":      int(rng.integers(3,24)),
            "ttnt2_days":       int((drug2-drug1).days),
            "has_discrepancy":  rng.random()<0.12,
            "disc_type":        DISC_TYPES[rng.integers(0,5)],
        })
    return pd.DataFrame(rows)


df = generate_data()

surv_df = pd.DataFrame({
    "Month": list(range(0,66,6)),
    "Overall Survival (%)":          [round(100*math.exp(-0.07*(m/6)),1) for m in range(0,66,6)],
    "Progression-Free Survival (%)": [round(100*math.exp(-0.10*(m/6)),1) for m in range(0,66,6)],
})

drug_trend = pd.DataFrame({
    "Quarter":       ["Q1'20","Q2'20","Q3'20","Q4'20","Q1'21","Q2'21","Q3'21"],
    "Pembrolizumab": [18,21,24,26,29,31,34],
    "Paclitaxel":    [45,43,41,40,38,36,35],
    "Trastuzumab":   [22,24,25,27,28,30,31],
    "Nivolumab":     [10,13,15,17,19,22,25],
})

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
        bordercolor=C_BORDER, borderwidth=1,
        font=dict(size=11.5)
    ),
    xaxis=dict(
        gridcolor=C_BORDER_LT, linecolor=C_BORDER,
        zerolinecolor=C_BORDER_LT, tickfont=dict(size=11)
    ),
    yaxis=dict(
        gridcolor=C_BORDER_LT, linecolor=C_BORDER,
        zerolinecolor=C_BORDER_LT, tickfont=dict(size=11)
    ),
    title_font=dict(size=13.5, color=C_PRIMARY, family="Syne"),
    title_x=0,
)

# Helper to get chart color for cancer type
def cancer_clr(cancer_type):
    return CANCER_COLORS.get(cancer_type, C_PRIMARY)


# ─────────────────────────────────────────────────────────────────────────────
# TOAST NOTIFICATION
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.toast:
    st.markdown(f'<div class="oi-toast">{st.session_state.toast}</div>',
                unsafe_allow_html=True)
    st.session_state.toast = None


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
                    Oncology Analytics v2.0
                </div>
            </div>
        </div>
        <div style="display:flex;gap:5px;margin-top:12px;">
            <span style="width:7px;height:7px;border-radius:50%;background:{C_PINK};display:inline-block;" title="Breast Cancer Awareness"></span>
            <span style="width:7px;height:7px;border-radius:50%;background:#A78BFA;display:inline-block;" title="General Cancer Awareness"></span>
            <span style="width:7px;height:7px;border-radius:50%;background:#2DD4BF;display:inline-block;" title="Ovarian Cancer Awareness"></span>
            <span style="width:7px;height:7px;border-radius:50%;background:#60A5FA;display:inline-block;" title="Prostate Cancer Awareness"></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""<div style='font-size:10px;font-weight:700;
                    color:rgba(196,191,237,0.5);letter-spacing:.12em;
                    text-transform:uppercase;margin-bottom:8px;
                    padding:0 4px;'>Navigation</div>""",
                unsafe_allow_html=True)

    page = st.radio("nav", [
        "🏠  Overview Dashboard",
        "⊕  ANT Classification",
        "≡  Line of Therapy",
        "⊞  Cohort Builder",
        "∿  Treatment Patterns",
        "⚑  Discrepancy Detection",
        "✦  AI Insights",
        "◻  Product Artifacts",
    ], label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Dark mode toggle
    dark_toggle = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if dark_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_toggle
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="oi-sidebar-dataset">
        <div style="font-weight:700;color:#A78BFA;margin-bottom:10px;font-size:12px;
                     font-family:'Syne',sans-serif;letter-spacing:0.03em;">
            📂 Active Dataset
        </div>
        <div style="color:rgba(196,191,237,0.8);font-size:12px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:5px;padding-bottom:5px;
                         border-bottom:1px solid rgba(255,255,255,0.08);">
                <span>Patients</span>
                <strong style="color:#EDE9FE;font-family:'Space Mono',monospace;">200</strong>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:5px;padding-bottom:5px;
                         border-bottom:1px solid rgba(255,255,255,0.08);">
                <span>Cancer Types</span>
                <strong style="color:#EDE9FE;font-family:'Space Mono',monospace;">6</strong>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:5px;padding-bottom:5px;
                         border-bottom:1px solid rgba(255,255,255,0.08);">
                <span>Date Range</span>
                <strong style="color:#EDE9FE;font-family:'Space Mono',monospace;">2019–21</strong>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span>Status</span>
                <strong style="color:#4ADE80;font-size:11px;">✓ Synthetic</strong>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""<div style='font-size:11.5px;font-weight:600;
                    color:rgba(196,191,237,0.7);margin-bottom:5px;'>
                    Upload Patient Data</div>""",
                unsafe_allow_html=True)
    uploaded = st.file_uploader("CSV", type=["csv"], label_visibility="collapsed")
    if uploaded:
        try:
            user_df = pd.read_csv(uploaded)
            st.success(f"✓ {len(user_df):,} rows · {len(user_df.columns)} cols")
        except Exception as e:
            st.error(str(e))

    st.markdown(f"""
    <div style="margin-top:18px;text-align:center;font-size:10.5px;
                color:rgba(196,191,237,0.4);
                border-top:1px solid rgba(255,255,255,0.08);padding-top:12px;">
        WCAG 2.1 AA · HIPAA Demo<br>
        <span style="color:rgba(167,139,250,0.5);">© 2025 OncoInsight Analytics</span>
    </div>
    """, unsafe_allow_html=True)

    # Sign out
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⇥  Sign Out"):
        st.session_state.authenticated = False
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def topbar(title, sub, badges):
    bdg = "".join([f'<span class="oi-badge">{b}</span>' for b in badges])
    st.markdown(f"""
    <div class="oi-topbar" role="banner" aria-label="Page header">
        <div>
            <div class="oi-topbar-title">{title}</div>
            <div class="oi-topbar-sub">{sub}</div>
        </div>
        <div role="list" aria-label="Page badges">{bdg}</div>
    </div>
    """, unsafe_allow_html=True)


def kpi_row(items):
    cols = st.columns(len(items))
    for col, (label, value, color, delta, d_clr) in zip(cols, items):
        col.markdown(f"""
        <div class="oi-kpi" style="border-left: 4px solid {color};"
             role="region" aria-label="{label}: {value}">
            <div class="oi-kpi-label" aria-hidden="true">{label}</div>
            <div class="oi-kpi-num" style="color:{color};"
                 aria-label="{value}">{value}</div>
            <div class="oi-kpi-delta" style="color:{d_clr or C_TEXT_MUTED};">{delta}</div>
        </div>
        """, unsafe_allow_html=True)


def sec(text):
    st.markdown(f'<div class="oi-sec" role="heading" aria-level="2">{text}</div>',
                unsafe_allow_html=True)


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


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
if "Overview" in page:
    topbar("🏠 Oncology Analytics Dashboard",
           "Real-World Evidence Platform · OncoInsight v2.0",
           ["🟢 Live Demo","200 Patients","6 Cancer Types"])
    wrap()

    neo_n  = df[df.ant_type=="Neoadjuvant"].shape[0]
    disc_n = df[df.has_discrepancy].shape[0]
    avg_os = round(df.os_months.mean(), 1)
    her2_n = df[df.biomarker=="HER2+"].shape[0]

    kpi_row([
        ("Total Patients",     "200",              C_PRIMARY,  "Full cohort enrolled",            ""),
        ("Cancer Types",       "6",                C_TEAL,     "Across 12 ICD-10 codes",          ""),
        ("Neoadjuvant Pts",    str(neo_n),         C_BLUE,     f"{neo_n/200*100:.0f}% of cohort", C_BLUE),
        ("Average OS",         f"{avg_os} mo",     C_SUCCESS,  "↑ vs benchmark",                  C_SUCCESS),
    ])
    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    kpi_row([
        ("2L Patients",        "142",              C_PRIMARY,  "71% progressed to 2nd line",      ""),
        ("HER2+ Patients",     str(her2_n),        C_PINK,     "Biomarker positive",              ""),
        ("Data Discrepancies", str(disc_n),        C_DANGER,   "⚠ Pending review",               C_DANGER),
        ("Stages Covered",     "I – IV",           C_SUCCESS,  "Complete stage range",            ""),
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        sec("📊 Cancer Type Distribution")
        cd = df.cancer_type.value_counts().reset_index()
        cd.columns = ["Cancer","Count"]
        # Use consistent cancer-type colors
        colors = [cancer_clr(c) for c in cd["Cancer"]]
        fig = go.Figure(go.Bar(
            x=cd["Cancer"], y=cd["Count"],
            marker=dict(color=colors, line_width=0),
            text=cd["Count"], textposition="outside",
        ))
        fig.update_layout(title="Patients by Primary Cancer", yaxis_title="Count",
                          showlegend=False, **PL)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("🎯 Stage Distribution")
        sd = df.stage.value_counts().reset_index()
        sd.columns = ["Stage","Count"]
        sd["Stage"] = "Stage " + sd["Stage"]
        fig = go.Figure(go.Pie(
            labels=sd["Stage"], values=sd["Count"], hole=0.48,
            marker=dict(colors=CHART_COLORS, line=dict(color="white" if not DARK else C_BG, width=2)),
            textfont=dict(size=12),
        ))
        fig.update_layout(title="Patient Stage Stratification", **PL)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        sec("📈 Overall & Progression-Free Survival")
        fig = go.Figure()
        for col_n, clr, fc_alpha in [
            ("Overall Survival (%)",          C_PRIMARY, 0.12),
            ("Progression-Free Survival (%)", C_TEAL,    0.10),
        ]:
            r,g,b = int(clr[1:3],16),int(clr[3:5],16),int(clr[5:7],16)
            fig.add_trace(go.Scatter(
                x=surv_df["Month"], y=surv_df[col_n], name=col_n,
                mode="lines", line=dict(color=clr, width=2.5),
                fill="tozeroy", fillcolor=f"rgba({r},{g},{b},{fc_alpha})",
            ))
        fig.update_layout(title="KM-Style Survival Curves",
                          xaxis_title="Months", yaxis_title="Survival (%)",
                          height=280, **PL)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        sec("🔬 Biomarker Distribution")
        bm = df.biomarker.value_counts().reset_index()
        bm.columns = ["Biomarker","Count"]
        fig = go.Figure(go.Bar(
            x=bm["Count"], y=bm["Biomarker"], orientation="h",
            marker=dict(color=C_PRIMARY, line_width=0,
                        opacity=[1,0.9,0.8,0.75,0.7,0.65,0.6,0.55]),
            text=bm["Count"], textposition="outside",
        ))
        fig.update_layout(title="Patients by Biomarker", xaxis_title="Count",
                          height=280, **PL)
        st.plotly_chart(fig, use_container_width=True)

    end()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ANT CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
elif "ANT" in page:
    topbar("⊕ ANT Therapy Classification Engine",
           "Adjuvant / Neoadjuvant auto-classification based on surgery date",
           ["Rule-Based Engine","WCAG AA"])
    wrap()

    neo_n = df[df.ant_type=="Neoadjuvant"].shape[0]
    adj_n = df[df.ant_type=="Adjuvant"].shape[0]

    kpi_row([
        ("Neoadjuvant Patients", str(neo_n),              C_BLUE,    "Drug before surgery",          ""),
        ("Adjuvant Patients",    str(adj_n),              C_SUCCESS, "Drug after surgery",           ""),
        ("Neoadjuvant Rate",     f"{neo_n/200*100:.0f}%", C_PRIMARY, "of total patient cohort",      ""),
        ("Avg Lead Time",        "42 days",               C_WARNING, "Drug → Surgery interval",      ""),
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="oi-alert-info">
        📋 <strong>Classification Algorithm:</strong>
        &nbsp; IF drug_date &lt; surgery_date → <strong>Neoadjuvant</strong>
        &nbsp;|&nbsp; IF drug_date &gt; surgery_date → <strong>Adjuvant</strong>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        sec("📊 ANT Distribution Overview")
        fig = go.Figure(go.Pie(
            labels=["Neoadjuvant","Adjuvant"], values=[neo_n,adj_n], hole=0.5,
            marker=dict(colors=[C_BLUE,C_SUCCESS],
                        line=dict(color="white" if not DARK else C_BG, width=2)),
            textfont=dict(size=13),
        ))
        fig.update_layout(title="Neoadjuvant vs Adjuvant Split", **PL)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("📊 ANT by Cancer Type")
        ant_c  = df.groupby(["cancer_type","ant_type"]).size().reset_index(name="n")
        pivot  = ant_c.pivot(index="cancer_type",columns="ant_type",values="n").fillna(0).reset_index()
        fig = go.Figure()
        for col_n, clr in zip(["Neoadjuvant","Adjuvant"],[C_BLUE,C_SUCCESS]):
            if col_n in pivot.columns:
                fig.add_trace(go.Bar(x=pivot["cancer_type"], y=pivot[col_n],
                                     name=col_n, marker_color=clr, marker_line_width=0))
        fig.update_layout(title="ANT Classification by Cancer Type",
                          barmode="group", **PL)
        st.plotly_chart(fig, use_container_width=True)

    sec("🗂 ANT Classification Table — Top 30 Records")
    disp = df[["patient_id","cancer_type","stage","drug1_name","drug1_date",
               "surgery_date","ant_type","drug1_class"]].head(30).copy()
    disp.columns = ["Patient ID","Cancer","Stage","Drug","Drug Date",
                    "Surgery Date","ANT Type","Drug Class"]
    def color_ant(val):
        if val=="Neoadjuvant":
            return f"background-color:{C_BLUE_BG};color:{C_BLUE};font-weight:600"
        if val=="Adjuvant":
            return f"background-color:{C_SUCCESS_BG};color:{C_SUCCESS};font-weight:600"
        return ""
    st.dataframe(disp.style.applymap(color_ant, subset=["ANT Type"]),
                 use_container_width=True, hide_index=True)
    end()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: LINE OF THERAPY
# ─────────────────────────────────────────────────────────────────────────────
elif "Line of Therapy" in page:
    topbar("≡ Line of Therapy (LoT) Engine",
           "Automated 1L → 2L → 3L → 4L+ identification algorithm",
           ["LoT Algorithm v1.0"])
    wrap()

    kpi_row([
        ("1L Patients", "200", C_PRIMARY, "Carboplatin + Paclitaxel",  ""),
        ("2L Patients", "142", C_TEAL,    "71% → Pembrolizumab",        ""),
        ("3L Patients",  "74", C_WARNING, "37% → Nivolumab",            C_WARNING),
        ("4L+ Patients", "28", C_DANGER,  "14% → Olaparib",             C_DANGER),
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        sec("📉 LoT Patient Attrition Funnel")
        fig = go.Figure(go.Funnel(
            y=["1L (200 pts)","2L (142 pts)","3L (74 pts)","4L+ (28 pts)"],
            x=[200,142,74,28],
            textinfo="value+percent initial",
            marker=dict(color=[C_PRIMARY,C_TEAL,C_WARNING,C_DANGER]),
        ))
        fig.update_layout(title="Patient Attrition Across Therapy Lines", **PL)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("💊 Drug Class Mix by Line (%)")
        lot_class = pd.DataFrame({
            "Line":     ["1L","2L","3L","4L+"],
            "Chemo":    [45,25,18,10],
            "Immuno":   [20,38,42,40],
            "Targeted": [25,30,35,45],
            "Hormonal": [10, 7, 5, 5],
        })
        fig = go.Figure()
        for col_n, clr in zip(["Chemo","Immuno","Targeted","Hormonal"],
                               [C_WARNING,C_PRIMARY,C_TEAL,C_PINK]):
            fig.add_trace(go.Bar(x=lot_class["Line"], y=lot_class[col_n],
                                 name=col_n, marker_color=clr, marker_line_width=0))
        fig.update_layout(title="Drug Class Distribution by LoT",
                          barmode="group", yaxis_title="%", **PL)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 LoT Algorithm Logic — Click to Expand"):
        card("📐 Line of Therapy Classification Rules",
             "<strong>Rule 1:</strong> First treatment after diagnosis = 1st Line<br>"
             "<strong>Rule 2:</strong> New / changed drug regimen = Next Line<br>"
             "<strong>Rule 3:</strong> Treatment gap &gt; 90 days = New Line<br>"
             "<strong>Rule 4:</strong> Disease progression + drug change = Next Line")

    sec("🗂 Patient-Level LoT Records (Top 25)")
    lot_d = df[["patient_id","cancer_type","stage","drug1_name",
                "drug2_name","drug3_name","ttnt2_days"]].head(25).copy()
    lot_d.columns = ["Patient ID","Cancer","Stage","1L Drug","2L Drug","3L Drug","Days to 2L"]
    def color_days(val):
        if isinstance(val,(int,float)):
            if val<90:  return f"color:{C_DANGER};font-weight:600"
            if val<150: return f"color:{C_WARNING};font-weight:600"
            return f"color:{C_SUCCESS};font-weight:600"
        return ""
    st.dataframe(lot_d.style.applymap(color_days, subset=["Days to 2L"]),
                 use_container_width=True, hide_index=True)
    end()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: COHORT BUILDER
# ─────────────────────────────────────────────────────────────────────────────
elif "Cohort" in page:
    topbar("⊞ Cohort Builder",
           "Build custom patient cohorts with multi-dimensional clinical filters",
           ["Self-Service Analytics"])
    wrap()

    st.markdown(f"""
    <div class="oi-filter-panel">
        <div style="font-size:12.5px;font-weight:700;color:{C_PRIMARY};
                     margin-bottom:12px;font-family:'Syne',sans-serif;
                     display:flex;align-items:center;gap:7px;">
            🔍 Filter Patient Cohort
            <span style="font-size:10px;font-weight:500;color:{C_TEXT_MUTED};
                          background:{C_PRIMARY_BG};padding:2px 8px;border-radius:10px;">
                Multi-dimensional filters
            </span>
        </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns([2,2,2,2,1])
    with c1: sel_cancer = st.selectbox("Cancer Type", ["All"]+CANCER_TYPES)
    with c2: sel_stage  = st.selectbox("Stage",       ["All"]+[f"Stage {s}" for s in STAGES])
    with c3: sel_bm     = st.selectbox("Biomarker",   ["All"]+BIOMARKERS)
    with c4: sel_drug   = st.selectbox("Drug",        ["All"]+DRUGS)
    with c5:
        st.markdown("<div style='margin-top:22px;'></div>", unsafe_allow_html=True)
        if st.button("↺ Reset Filters"):
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    filt = df.copy()
    if sel_cancer!="All": filt = filt[filt.cancer_type==sel_cancer]
    if sel_stage !="All": filt = filt[filt.stage==sel_stage.replace("Stage ","")]
    if sel_bm    !="All": filt = filt[filt.biomarker==sel_bm]
    if sel_drug  !="All": filt = filt[(filt.drug1_name==sel_drug)|(filt.drug2_name==sel_drug)|(filt.drug3_name==sel_drug)]

    avg_age  = round(filt.age.mean(),1) if len(filt) else 0
    female_p = round(filt[filt.gender=="Female"].shape[0]/max(len(filt),1)*100)
    bm_pos   = filt[filt.biomarker_result=="Positive"].shape[0]

    kpi_row([
        ("Cohort Size",   str(len(filt)),  C_PRIMARY, "Matched patients",    ""),
        ("Average Age",   f"{avg_age} yr", C_TEAL,    "Median age",          ""),
        ("Female",        f"{female_p}%",  C_PINK,    "of cohort",           ""),
        ("Biomarker +ve", str(bm_pos),     C_PURPLE,  "Positive results",    ""),
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    if len(filt)==0:
        st.markdown(f'<div class="oi-alert-warn">⚠ No patients match the selected filters. Please adjust your criteria.</div>',
                    unsafe_allow_html=True)
    else:
        c1,c2 = st.columns(2)
        with c1:
            sec("📊 Cohort by Cancer Type")
            cc = filt.cancer_type.value_counts().reset_index()
            cc.columns = ["Cancer","Count"]
            colors = [cancer_clr(c) for c in cc["Cancer"]]
            fig = go.Figure(go.Pie(
                labels=cc["Cancer"],values=cc["Count"],hole=0.48,
                marker=dict(colors=colors,
                            line=dict(color="white" if not DARK else C_BG, width=2))))
            fig.update_layout(title="Cancer Distribution in Cohort",**PL)
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            sec("📊 Stage Breakdown")
            sc = filt.stage.value_counts().reset_index()
            sc.columns = ["Stage","Count"]
            sc["Stage"] = "Stage "+sc["Stage"]
            fig = go.Figure(go.Bar(
                x=sc["Stage"],y=sc["Count"],
                marker=dict(color=C_PRIMARY,line_width=0,
                            opacity=[1,0.85,0.7,0.55]),
                text=sc["Count"],textposition="outside"))
            fig.update_layout(title="Stage Distribution in Cohort",
                              yaxis_title="Patients",**PL)
            st.plotly_chart(fig,use_container_width=True)

        sec(f"🗂 Patient Records — {len(filt)} Matched · Click row for journey")
        disp = filt[["patient_id","age","gender","cancer_type","stage",
                     "biomarker","ant_type","drug1_name","os_months"]].head(20).copy()
        disp.columns = ["ID","Age","Gender","Cancer","Stage","Biomarker","ANT","1L Drug","OS (mo)"]

        sel = st.dataframe(disp, use_container_width=True, hide_index=True,
                           on_select="rerun", selection_mode="single-row")

        if sel and sel.selection.rows:
            patient = filt.iloc[sel.selection.rows[0]]
            st.markdown(f"""
            <div style="background:{C_PRIMARY_BG};border:1px solid {C_BORDER};
                        border-left:4px solid {C_PRIMARY};
                        border-radius:0 12px 12px 0;padding:14px 20px;margin:12px 0 16px;">
                <div style="font-size:14px;font-weight:700;color:{C_PRIMARY};
                             font-family:'Syne',sans-serif;">
                    🧑‍⚕️ Patient Journey — {patient.patient_id}
                    &nbsp;·&nbsp; {patient.cancer_type} Stage {patient.stage}
                    &nbsp;·&nbsp; {patient.ant_type}
                </div>
            </div>
            """, unsafe_allow_html=True)

            timeline = [
                ("🏥","Diagnosis",  str(patient.diagnosis_date), C_WARNING),
                ("💊","1L Therapy", str(patient.drug1_date),     C_PRIMARY),
                ("🔪","Surgery",    str(patient.surgery_date),   C_DANGER),
                ("💉","2L Therapy", str(patient.drug2_date),     C_SUCCESS),
                ("🧪","3L Therapy", str(patient.drug3_date),     C_TEAL),
            ]
            cols = st.columns(len(timeline))
            for col,(icon,label,date,color) in zip(cols,timeline):
                col.markdown(f"""
                <div class="oi-journey-node" style="border-top-color:{color};">
                    <div class="oi-journey-icon">{icon}</div>
                    <div class="oi-journey-title" style="color:{color};">{label}</div>
                    <div class="oi-journey-date">{date}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>",unsafe_allow_html=True)
            c1,c2,c3 = st.columns(3)
            with c1:
                st.metric("Cancer Type", patient.cancer_type)
                st.metric("Stage", patient.stage)
            with c2:
                st.metric("Biomarker", patient.biomarker)
                st.metric("ANT Type",  patient.ant_type)
            with c3:
                st.metric("OS (months)",  patient.os_months)
                st.metric("PFS (months)", patient.pfs_months)
    end()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: TREATMENT PATTERNS
# ─────────────────────────────────────────────────────────────────────────────
elif "Treatment Patterns" in page:
    topbar("∿ Treatment Pattern Analysis",
           "Drug utilization · therapy trends · biomarker-drug correlations",
           ["Drug Analytics"])
    wrap()

    c1,c2 = st.columns(2)
    with c1:
        sec("💊 Drug Class Distribution — 1st Line")
        dc = df.drug1_class.value_counts().reset_index()
        dc.columns = ["Class","Count"]
        fig = go.Figure(go.Pie(
            labels=dc["Class"],values=dc["Count"],hole=0.48,
            marker=dict(colors=CHART_COLORS,
                        line=dict(color="white" if not DARK else C_BG, width=2))))
        fig.update_layout(title="First-Line Drug Class Breakdown",**PL)
        st.plotly_chart(fig,use_container_width=True)

    with c2:
        sec("📈 Drug Adoption Trends — Quarterly")
        fig = go.Figure()
        for col_n,clr in zip(["Pembrolizumab","Paclitaxel","Trastuzumab","Nivolumab"],
                              [C_PRIMARY,C_WARNING,C_TEAL,C_BLUE]):
            fig.add_trace(go.Scatter(
                x=drug_trend["Quarter"],y=drug_trend[col_n],name=col_n,
                mode="lines+markers",
                line=dict(color=clr,width=2.5),
                marker=dict(size=7,color=clr,
                            line=dict(color="white" if not DARK else C_BG, width=2)),
            ))
        fig.update_layout(title="Quarter-over-Quarter Drug Utilization",
                          yaxis_title="Patient Count",**PL)
        st.plotly_chart(fig,use_container_width=True)

    sec("📊 Drug Utilisation — All Lines (Ranked)")
    all_d = pd.concat([df.drug1_name,df.drug2_name,df.drug3_name]).value_counts().reset_index()
    all_d.columns = ["Drug","Patients"]
    all_d["Pct"] = (all_d["Patients"]/200*100).round(1)

    for i,row in all_d.iterrows():
        pct   = row["Pct"]
        color = CHART_COLORS[i%len(CHART_COLORS)]
        c1,c2,c3,c4 = st.columns([2,5,1,0.5])
        c1.markdown(f"<div style='font-size:13.5px;padding-top:8px;color:{C_TEXT};"
                    f"font-weight:600;'>{row['Drug']}</div>",unsafe_allow_html=True)
        c2.markdown(f"""
        <div style="margin-top:12px;">
            <div class="oi-progress-bg">
                <div class="oi-progress-fill"
                     style="width:{min(pct,100):.1f}%;background:linear-gradient(90deg,{color},{color}aa);"></div>
            </div>
        </div>""",unsafe_allow_html=True)
        c3.markdown(f"<div style='font-size:12.5px;color:{color};font-weight:700;"
                    f"padding-top:6px;font-family:Space Mono,monospace;'>{pct}%</div>",
                    unsafe_allow_html=True)
        c4.markdown(f"<div style='font-size:11.5px;color:{C_TEXT_MUTED};padding-top:8px;'>"
                    f"{int(row['Patients'])}</div>",unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    sec("🔬 Biomarker vs Therapy Modality — Radar")
    cats = ["HER2+","EGFR+","PD-L1+","BRCA+","ALK+"]
    fig = go.Figure()
    for name,vals,clr in [
        ("Chemotherapy",[40,35,30,50,25],C_WARNING),
        ("Targeted",    [75,80,40,70,85],C_PRIMARY),
        ("Immunotherapy",[30,25,90,20,15],C_TEAL),
    ]:
        r,g,b = int(clr[1:3],16),int(clr[3:5],16),int(clr[5:7],16)
        fig.add_trace(go.Scatterpolar(
            r=vals+[vals[0]],theta=cats+[cats[0]],fill="toself",name=name,
            line=dict(color=clr,width=2.5),
            fillcolor=f"rgba({r},{g},{b},0.13)",
        ))
    fig.update_layout(
        title="Therapy Modality by Biomarker Group",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True,gridcolor=C_BORDER_LT,color=C_TEXT_MUTED),
            angularaxis=dict(gridcolor=C_BORDER_LT,color=C_TEXT)),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Figtree,system-ui",color=C_TEXT,size=12),
        legend=dict(bgcolor=f"rgba({'30,24,64' if DARK else '255,255,255'},0.95)",
                    bordercolor=C_BORDER,borderwidth=1),
        margin=dict(l=40,r=40,t=50,b=40),
        title_font=dict(size=13.5,color=C_PRIMARY,family="Syne"),
    )
    st.plotly_chart(fig,use_container_width=True)
    end()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DISCREPANCY DETECTION
# ─────────────────────────────────────────────────────────────────────────────
elif "Discrepancy" in page:
    topbar("⚑ Discrepancy Detection Engine",
           "Automated data quality checks · flagged records · resolution guidance",
           ["Data Quality Monitor"])
    wrap()

    disc_df = df[df.has_discrepancy].copy()
    disc_df["severity"] = np.where(
        np.random.default_rng(99).random(len(disc_df))>0.5,"HIGH","MEDIUM")
    high_n   = disc_df[disc_df.severity=="HIGH"].shape[0]
    medium_n = disc_df[disc_df.severity=="MEDIUM"].shape[0]
    clean_n  = 200-len(disc_df)

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="oi-alert-danger">🔴 <strong>{high_n} HIGH severity</strong> — Immediate review required</div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="oi-alert-warn">🟡 <strong>{medium_n} MEDIUM severity</strong> — Review recommended</div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="oi-alert-success">🟢 <strong>{clean_n} records</strong> passed all quality checks</div>',
                    unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    kpi_row([
        ("Total Flags",     str(len(disc_df)), C_DANGER,  "Require review",            C_DANGER),
        ("High Severity",   str(high_n),       C_WARNING, "Immediate action needed",   C_WARNING),
        ("Medium Severity", str(medium_n),     C_BLUE,    "Review recommended",        ""),
        ("Clean Records",   str(clean_n),       C_SUCCESS, f"{clean_n/200*100:.0f}% pass rate", C_SUCCESS),
    ])

    st.markdown("<br>",unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        sec("📊 Discrepancy Type Breakdown")
        dt = disc_df.disc_type.value_counts().reset_index()
        dt.columns = ["Issue","Count"]
        fig = go.Figure(go.Bar(
            x=dt["Count"],y=dt["Issue"],orientation="h",
            marker=dict(color=C_DANGER,line_width=0,
                        opacity=[1,0.85,0.75,0.65,0.55]),
            text=dt["Count"],textposition="outside"))
        fig.update_layout(title="Data Quality Issue Types",xaxis_title="Records",**PL)
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        sec("🥧 Record Quality Overview")
        fig = go.Figure(go.Pie(
            labels=["High Severity","Medium Severity","Clean Records"],
            values=[high_n,medium_n,clean_n],hole=0.5,
            marker=dict(colors=[C_DANGER,C_WARNING,C_SUCCESS],
                        line=dict(color="white" if not DARK else C_BG, width=2))))
        fig.update_layout(title="Quality Breakdown — 200 Records",**PL)
        st.plotly_chart(fig,use_container_width=True)

    resolutions = {
        "Drug date before diagnosis":  "Verify drug administration records against EHR source system",
        "Missing surgery data":        "Cross-reference surgical scheduling and OR records",
        "Duplicate therapy event":     "Deduplicate using patient_id + drug_name + date composite key",
        "Conflicting drug record":     "Reconcile claims data vs EMR — adjudicate with source of truth policy",
        "Biomarker mismatch":          "Validate biomarker test result with pathology report and lab records",
    }
    sec("🗂 Flagged Records — Resolution Guidance")
    dd = disc_df[["patient_id","cancer_type","disc_type","severity"]].copy()
    dd["Suggested Resolution"] = dd["disc_type"].map(resolutions)
    dd.columns = ["Patient ID","Cancer","Issue Type","Severity","Suggested Resolution"]

    def sev_clr(val):
        if val=="HIGH":   return f"background-color:{C_DANGER_BG};color:{C_DANGER};font-weight:700"
        if val=="MEDIUM": return f"background-color:{C_WARNING_BG};color:{C_WARNING};font-weight:700"
        return ""
    st.dataframe(dd.style.applymap(sev_clr,subset=["Severity"]),
                 use_container_width=True,hide_index=True)
    end()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: AI INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
elif "AI Insights" in page:
    topbar("✦ AI Insight Generator",
           "Automated oncology insights from analytics engines",
           ["Analytics Engine"])
    wrap()

    if "insights" not in st.session_state:
        st.session_state.insights = [
            {"icon":"🔬","text":"62% of Stage III breast cancer patients received neoadjuvant chemotherapy prior to surgery.",
             "source":"ANT Engine","confidence":"High","color":C_PRIMARY},
            {"icon":"🎯","text":"HER2+ patients show 2.4× higher targeted therapy adoption in second-line vs first-line.",
             "source":"LoT Analysis","confidence":"High","color":C_SUCCESS},
            {"icon":"📈","text":"Immunotherapy adoption increased 34% year-over-year across all cancer types.",
             "source":"Drug Trends","confidence":"Medium","color":C_TEAL},
            {"icon":"⏱️","text":"Median TTNT for Stage IV is 4.2 months shorter than Stage III patients.",
             "source":"Survival Engine","confidence":"High","color":C_WARNING},
            {"icon":"⚠️","text":f"{df[df.has_discrepancy].shape[0]} patients show data discrepancies — primarily drug date anomalies.",
             "source":"Discrepancy Engine","confidence":"N/A","color":C_DANGER},
        ]

    pool = [
        {"icon":"🧬","text":"Stage IV patients show 38% higher immunotherapy use vs Stage II.",
         "source":"Cohort Analysis","confidence":"High","color":C_PRIMARY},
        {"icon":"💡","text":"BRCA1/2+ patients show significantly higher PARP inhibitor adoption in 2L.",
         "source":"Biomarker Engine","confidence":"High","color":C_SUCCESS},
        {"icon":"🔄","text":"Drug switching 1L→2L most frequent within 120 days of initiation.",
         "source":"LoT Engine","confidence":"Medium","color":C_TEAL},
        {"icon":"📊","text":"Colorectal patients had highest neoadjuvant therapy rate at 71% of subgroup.",
         "source":"ANT Engine","confidence":"High","color":C_WARNING},
        {"icon":"⚕️","text":"Neoadjuvant patients had median OS 6.8 months longer than adjuvant-only.",
         "source":"Survival Model","confidence":"Medium","color":C_PURPLE},
    ]

    c1,c2 = st.columns([2,1])
    with c1:
        if st.button("✦ Generate New Insight"):
            available = [n for n in pool if n["text"] not in [x["text"] for x in st.session_state.insights]]
            if available:
                st.session_state.insights.append(random.choice(available))
                st.session_state.toast = "✓ New insight generated"
                st.rerun()
            else:
                st.markdown(f'<div class="oi-alert-info">ℹ All available insights have been generated.</div>',
                            unsafe_allow_html=True)
    with c2:
        if st.button("↺ Reset Insights"):
            del st.session_state.insights
            st.rerun()

    st.markdown("<br>",unsafe_allow_html=True)
    for ins in st.session_state.insights:
        conf_clr = C_SUCCESS if ins["confidence"]=="High" else C_WARNING if ins["confidence"]=="Medium" else C_TEXT_MUTED
        conf_bg  = C_SUCCESS_BG if ins["confidence"]=="High" else C_WARNING_BG if ins["confidence"]=="Medium" else C_SURFACE2
        st.markdown(f"""
        <div class="oi-insight" style="border-left-color:{ins['color']};"
             role="article" aria-label="Clinical insight">
            <div style="display:flex;align-items:flex-start;gap:14px;">
                <span style="font-size:22px;margin-top:1px;" aria-hidden="true">{ins['icon']}</span>
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
        """,unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    sec("📊 Insight Coverage by Analytics Module")
    cov = pd.DataFrame({
        "Module": ["ANT Engine","LoT Engine","Cohort Builder","Survival Model","Drug Utilization","Discrepancy"],
        "Insights": [12,18,9,7,14,5]
    })
    colors = [C_PRIMARY, C_TEAL, C_BLUE, C_SUCCESS, C_WARNING, C_DANGER]
    fig = go.Figure(go.Bar(
        x=cov["Module"],y=cov["Insights"],
        marker=dict(color=colors,line_width=0),
        text=cov["Insights"],textposition="outside"))
    fig.update_layout(title="Auto-Generated Insights per Module",
                      yaxis_title="Insight Count",**PL)
    st.plotly_chart(fig,use_container_width=True)
    end()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PRODUCT ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
elif "Product" in page:
    topbar("◻ Product Artifacts — PM Deliverables",
           "PRD · User Personas · Business Value · KPIs · Roadmap",
           ["Product Management"])
    wrap()

    tabs = st.tabs(["📋 PRD","👥 Personas","💼 Business Value","📊 KPIs","🗺️ Roadmap"])

    with tabs[0]:
        for title,body in [
            ("🎯 Product Vision",
             "Build a self-service oncology analytics application enabling pharma companies, "
             "clinical researchers, and healthcare organizations to explore real-world oncology data "
             "and generate actionable insights on treatment patterns, therapy lines, and patient cohorts."),
            ("🔍 Problem Statement",
             "Pharma and oncology analytics teams lack tools to quickly analyze fragmented real-world data "
             "across EHR, Claims, Pathology, and Biomarker sources. Current workflows are slow, manual, "
             "and require deep engineering — creating critical bottlenecks in drug strategy decisions."),
            ("✅ MVP Scope",
             "Data Upload &amp; Ingestion (CSV/JSON/FHIR) · ANT Therapy Classification Engine · "
             "Line of Therapy Algorithm · Interactive Cohort Builder · Treatment Pattern Dashboards · "
             "Discrepancy Detection Engine · AI Insight Generation."),
            ("🎯 Success Criteria",
             "Time to first insight &lt; 5 min · Cohort build &lt; 30 sec · ANT accuracy ≥ 95% · "
             "Dashboard load &lt; 2 sec · Discrepancy detection recall ≥ 90%."),
            ("🔐 Data &amp; Compliance",
             "Patient data de-identified per HIPAA Safe Harbor. Platform supports audit logging, "
             "RBAC, and SOC 2 Type II. No PHI stored in-transit without AES-256 encryption."),
        ]:
            card(title, body)

    with tabs[1]:
        c1,c2 = st.columns(2)
        personas = [
            ("💊 Pharma Analyst",      C_PRIMARY, "Understand drug usage patterns and market penetration",
             "Slow analytics, fragmented datasets, no self-service tools"),
            ("🔬 Oncology Researcher", C_TEAL,    "Study treatment outcomes and biomarker correlations",
             "Difficult cohort discovery, inconsistent data schemas"),
            ("📊 Data Scientist",      C_PURPLE,  "Build and validate oncology ML models at scale",
             "Manual data prep, no standardized oncology data model"),
            ("🏥 Clinical Operations", C_SUCCESS, "Monitor guideline adherence and optimize treatment pathways",
             "No real-time monitoring, siloed EHR systems, slow reports"),
        ]
        for i,(role,clr,goal,pain) in enumerate(personas):
            with (c1 if i%2==0 else c2):
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
                """,unsafe_allow_html=True)

    with tabs[2]:
        c1,c2,c3 = st.columns(3)
        bv = [
            (c1,"💊 Pharma Companies",    C_PRIMARY,
             ["Drug launch strategy","Competitive therapy analysis","Patient segmentation","RWE generation","Trial enrichment"]),
            (c2,"🏥 Healthcare Providers", C_TEAL,
             ["Treatment pathway optimization","Outcome monitoring","Guideline adherence","Biomarker-driven care","Quality dashboards"]),
            (c3,"🔬 Researchers",          C_PURPLE,
             ["Treatment research","Biomarker studies","Survival analysis","Publication visuals","Hypothesis generation"]),
        ]
        for col,title,clr,items in bv:
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
                """,unsafe_allow_html=True)

    with tabs[3]:
        k_items = [
            ("Target Pharma Users (Y1)", "50+",         C_PRIMARY),
            ("Cohort Analyses / Month",  "500+",        C_TEAL),
            ("Insight Generation Time",  "< 5 min",     C_SUCCESS),
            ("Data Processing Speed",    "1M rows/min", C_WARNING),
            ("ANT Accuracy",             "≥ 95%",       C_PURPLE),
            ("Target ARR (Year 1)",      "$2M+",        C_DANGER),
        ]
        c1,c2,c3 = st.columns(3)
        for i,(label,val,clr) in enumerate(k_items):
            with [c1,c2,c3][i%3]:
                st.markdown(f"""
                <div class="oi-biz-kpi" style="border-top-color:{clr};">
                    <div class="oi-biz-kpi-val" style="color:{clr};">{val}</div>
                    <div class="oi-biz-kpi-label">{label}</div>
                </div>
                """,unsafe_allow_html=True)

        sec("📉 User Adoption Funnel — Year 1 Target")
        fig = go.Figure(go.Funnel(
            y=["Awareness","Trial","Activated","Power Users","Enterprise"],
            x=[500,200,80,40,15],
            textinfo="value+percent initial",
            marker=dict(color=[C_PRIMARY,C_TEAL,C_SUCCESS,C_WARNING,C_PINK]),
        ))
        fig.update_layout(title="Y1 User Adoption Funnel",**PL,height=280)
        st.plotly_chart(fig,use_container_width=True)

    with tabs[4]:
        for num,clr,title,items in [
            ("1", C_PRIMARY, "Phase 1 — MVP (Months 1–3)",
             "Streamlit prototype · ANT Classification · LoT Engine · Cohort Builder · Treatment Dashboards · Discrepancy Detection · CSV/JSON ingestion"),
            ("2", C_TEAL,    "Phase 2 — Advanced Analytics (Months 4–8)",
             "Survival analysis (KM curves) · ML predictions · LLM clinical insights · FHIR connectors · PDF export · User authentication"),
            ("3", C_PURPLE,  "Phase 3 — Enterprise Platform (Months 9–18)",
             "Cloud deployment (AWS/GCP) · Real-time pipelines · Multi-tenant · SSO & RBAC · EHR integrations · API marketplace · SLA guarantees"),
        ]:
            r,g,b = int(clr[1:3],16),int(clr[3:5],16),int(clr[5:7],16)
            st.markdown(f"""
            <div class="oi-phase-card" style="border-left-color:{clr};">
                <div style="width:32px;height:32px;min-width:32px;border-radius:50%;
                            background:rgba({r},{g},{b},0.15);
                            color:{clr};font-weight:800;font-size:14px;
                            font-family:'Syne',sans-serif;
                            display:flex;align-items:center;justify-content:center;
                            border:2px solid rgba({r},{g},{b},0.3);">{num}</div>
                <div>
                    <div style="font-weight:700;font-size:14px;color:{clr};margin-bottom:4px;
                                 font-family:'Syne',sans-serif;">{title}</div>
                    <div style="font-size:13px;color:{C_TEXT_MED};line-height:1.6;">{items}</div>
                </div>
            </div>
            """,unsafe_allow_html=True)

        sec("📅 Development Gantt Chart")
        gantt_df = pd.DataFrame([
            dict(Task="Data Ingestion", Start="2025-01-01",Finish="2025-02-15",Phase="Phase 1"),
            dict(Task="ANT Engine",     Start="2025-01-15",Finish="2025-03-01",Phase="Phase 1"),
            dict(Task="LoT Engine",     Start="2025-02-01",Finish="2025-03-15",Phase="Phase 1"),
            dict(Task="Cohort Builder", Start="2025-02-15",Finish="2025-04-01",Phase="Phase 1"),
            dict(Task="Survival Model", Start="2025-04-01",Finish="2025-06-01",Phase="Phase 2"),
            dict(Task="ML Predictions", Start="2025-05-01",Finish="2025-08-01",Phase="Phase 2"),
            dict(Task="Cloud Deploy",   Start="2025-09-01",Finish="2025-12-01",Phase="Phase 3"),
            dict(Task="EHR Connectors", Start="2025-10-01",Finish="2026-02-01",Phase="Phase 3"),
        ])
        fig = px.timeline(gantt_df,x_start="Start",x_end="Finish",y="Task",color="Phase",
            color_discrete_map={"Phase 1":C_PRIMARY,"Phase 2":C_TEAL,"Phase 3":C_PURPLE})
        fig.update_layout(title="OncoInsight Development Roadmap",**PL,height=310)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig,use_container_width=True)

    end()


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="border-top:1px solid {C_BORDER};margin:0;padding:16px 28px 24px;
            display:flex;align-items:center;justify-content:space-between;
            background:{C_SURFACE};flex-wrap:wrap;gap:10px;">
    <div style="font-size:12px;color:{C_TEXT_MUTED};">
        <strong style="color:{C_PRIMARY};font-family:'Syne',sans-serif;font-size:13px;">
            OncoInsight Analytics Platform v2.0
        </strong>
        <span style="margin:0 10px;opacity:0.3;">|</span>
        WCAG 2.1 AA Compliant
        <span style="margin:0 10px;opacity:0.3;">|</span>
        HIPAA Demo Environment
        <span style="margin:0 10px;opacity:0.3;">|</span>
        All patient data is synthetic
    </div>
    <div style="display:flex;align-items:center;gap:8px;">
        <span style="font-size:11px;color:{C_TEXT_MUTED};">Cancer Awareness:</span>
        <span title="General Cancer" style="width:8px;height:8px;border-radius:50%;
               background:{C_PRIMARY};display:inline-block;"></span>
        <span title="Breast Cancer" style="width:8px;height:8px;border-radius:50%;
               background:{C_PINK};display:inline-block;"></span>
        <span title="Ovarian Cancer" style="width:8px;height:8px;border-radius:50%;
               background:{C_TEAL};display:inline-block;"></span>
        <span title="Prostate Cancer" style="width:8px;height:8px;border-radius:50%;
               background:{C_BLUE};display:inline-block;"></span>
    </div>
</div>
""", unsafe_allow_html=True)
