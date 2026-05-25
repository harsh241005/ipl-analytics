"""
IPL Analytics Dashboard — v4 (Command Center theme)
Inspired by Bloomberg terminals / modern ML dashboards.
6 pages, real team logos, real metrics only, plain language.

Run with: streamlit run app.py
"""

import base64
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================================
# CONFIG
# ============================================================================

BASE_PATH = Path(r"C:\Users\harsh\OneDrive\Desktop\ipl data analysis\datanew")
if not BASE_PATH.exists():
    BASE_PATH = Path(__file__).parent

ASSETS_PATH = BASE_PATH / "assets" / "logos"
if not ASSETS_PATH.exists():
    alt = Path(__file__).parent / "assets" / "logos"
    if alt.exists():
        ASSETS_PATH = alt

st.set_page_config(
    page_title="IPL Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Command Center Theme
# ----------------------------------------------------------------------------
BG_PAGE     = "#0A0F1F"  # Main app background
BG_SURFACE  = "#131A2E"  # Cards
BG_DEEP     = "#0E1428"  # Recessed surfaces
BG_HOVER    = "#1A233F"

TEXT_PRIMARY   = "#FFFFFF"
TEXT_SECONDARY = "#A0AEC0"
TEXT_MUTED     = "#6B7A99"

CYAN     = "#00D9FF"          # Primary accent
CYAN_DIM = "rgba(0, 217, 255, 0.15)"
BLUE     = "#4A7AFF"
AMBER    = "#FFB300"          # Winner / champion / star
RED      = "#FF5252"          # Loss / negative
GREEN    = "#22D3A8"          # Positive trend

BORDER       = "rgba(255, 255, 255, 0.06)"
BORDER_GLOW  = "rgba(0, 217, 255, 0.35)"

TEAM_COLORS = {
    "Chennai Super Kings":         "#FFFF3C",
    "Mumbai Indians":              "#004B8D",
    "Royal Challengers Bangalore": "#DA1818",
    "Kolkata Knight Riders":       "#3A225D",
    "Rajasthan Royals":            "#EA1A85",
    "Sunrisers Hyderabad":         "#E04F16",
    "Delhi Capitals":              "#17449B",
    "Punjab Kings":                "#DD1F2D",
    "Gujarat Titans":              "#1B74BC",
    "Lucknow Super Giants":        "#0057E2",
    "Rising Pune Supergiants":     "#B7387C",
    "Gujarat Lions":               "#F7A721",
    "Kochi Tuskers Kerala":        "#502D7F",
    "Deccan Chargers":             "#6B7280",
    "Pune Warriors":               "#3F8DCC",
}

LOGO_FILES = {
    "Chennai Super Kings":         "csk.png",
    "Mumbai Indians":              "mi.png",
    "Royal Challengers Bangalore": "rcb.jpg",
    "Kolkata Knight Riders":       "kkr.png",
    "Rajasthan Royals":            "rr.png",
    "Sunrisers Hyderabad":         "srh.png",
    "Delhi Capitals":              "dc.png",
    "Punjab Kings":                "pbks.jpg",
    "Gujarat Titans":              "gt.png",
    "Lucknow Super Giants":        "lsg.png",
    "Rising Pune Supergiants":     "rps.png",
    "Gujarat Lions":               "gl.png",
    "Kochi Tuskers Kerala":        "kochi.png",
    "Deccan Chargers":             "deccan.png",
    "Pune Warriors":               "",
}

TEAM_ABBR = {
    "Chennai Super Kings":         "CSK",
    "Mumbai Indians":              "MI",
    "Royal Challengers Bangalore": "RCB",
    "Kolkata Knight Riders":       "KKR",
    "Rajasthan Royals":            "RR",
    "Sunrisers Hyderabad":         "SRH",
    "Delhi Capitals":              "DC",
    "Punjab Kings":                "PBKS",
    "Gujarat Titans":              "GT",
    "Lucknow Super Giants":        "LSG",
    "Rising Pune Supergiants":     "RPS",
    "Gujarat Lions":               "GL",
    "Kochi Tuskers Kerala":        "KTK",
    "Deccan Chargers":             "DC*",
    "Pune Warriors":               "PW",
}

ACTIVE_TEAMS_2026 = [
    "Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Rajasthan Royals", "Sunrisers Hyderabad",
    "Delhi Capitals", "Punjab Kings", "Gujarat Titans", "Lucknow Super Giants",
]

TEAM_ENC = {
    "Chennai Super Kings": 0, "Delhi Capitals": 1, "Deccan Chargers": 2,
    "Gujarat Lions": 3, "Gujarat Titans": 4, "Kolkata Knight Riders": 5,
    "Kochi Tuskers Kerala": 6, "Lucknow Super Giants": 7, "Mumbai Indians": 8,
    "Punjab Kings": 9, "Pune Warriors": 10, "Rajasthan Royals": 11,
    "Rising Pune Supergiants": 12, "Royal Challengers Bangalore": 13,
    "Sunrisers Hyderabad": 14,
}

ML_FEATURES = [
    "team_enc", "opponent_enc", "toss_won", "form",
    "h2h_win_pct", "venue_win_pct", "season",
    "overall_win_rate", "season_win_rate",
    "h2h_runs_powerplay", "h2h_runs_middle", "h2h_runs_death",
    "h2h_wkts_powerplay", "h2h_wkts_middle", "h2h_wkts_death",
]

FINAL_FIXES = {
    336019: "Rajasthan Royals", 392236: "Deccan Chargers",
    419161: "Chennai Super Kings", 501267: "Chennai Super Kings",
    548377: "Kolkata Knight Riders", 598069: "Mumbai Indians",
    734041: "Kolkata Knight Riders", 829815: "Mumbai Indians",
    981011: "Sunrisers Hyderabad", 1082646: "Mumbai Indians",
    1216495: "Chennai Super Kings", 1304116: "Gujarat Titans",
    1359544: "Chennai Super Kings", 1426307: "Kolkata Knight Riders",
}

IPL_CHAMPIONS = {
    2008: "Rajasthan Royals",
    2009: "Deccan Chargers",
    2010: "Chennai Super Kings",
    2011: "Chennai Super Kings",
    2012: "Kolkata Knight Riders",
    2013: "Mumbai Indians",
    2014: "Kolkata Knight Riders",
    2015: "Mumbai Indians",
    2016: "Sunrisers Hyderabad",
    2017: "Mumbai Indians",
    2018: "Chennai Super Kings",
    2019: "Mumbai Indians",
    2020: "Mumbai Indians",
    2021: "Chennai Super Kings",
    2022: "Gujarat Titans",
    2023: "Chennai Super Kings",
    2024: "Kolkata Knight Riders",
    2025: "Royal Challengers Bangalore",
}

# ============================================================================
# GLOBAL CSS — Command Center Theme
# ============================================================================

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap');

/* ── App canvas ─────────────────────────────────────────────────────────── */
.stApp {{
    background: {BG_PAGE};
    background-image:
      linear-gradient({BG_PAGE} 1px, transparent 1px),
      linear-gradient(90deg, {BG_PAGE} 1px, transparent 1px),
      radial-gradient(ellipse at top right, rgba(0, 217, 255, 0.04), transparent 60%),
      radial-gradient(ellipse at bottom left, rgba(74, 122, 255, 0.04), transparent 60%);
    background-size: 32px 32px, 32px 32px, 100% 100%, 100% 100%;
    color: {TEXT_PRIMARY};
    font-family: 'Inter', system-ui, sans-serif;
    overflow-x: hidden;
}}
.main .block-container {{
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
    min-width: 0;
}}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background: {BG_DEEP};
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] > div {{ padding-top: 0.5rem; }}

/* ── Typography ─────────────────────────────────────────────────────────── */
h1, h2, h3, h4 {{
    color: {TEXT_PRIMARY};
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    letter-spacing: -0.02em;
}}
h1 {{ font-weight: 700; }}
p, label, span, div {{ font-family: 'Inter', sans-serif; }}

/* ── Page hero ──────────────────────────────────────────────────────────── */
.hero-card {{
    background: linear-gradient(135deg, {BG_SURFACE} 0%, {BG_DEEP} 100%);
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}}
.hero-card::before {{
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, {CYAN}, transparent);
}}
.hero-card h1 {{
    margin: 0; font-size: 2rem; line-height: 1.1;
    background: linear-gradient(90deg, {TEXT_PRIMARY} 30%, {CYAN});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.hero-card .hero-sub {{
    color: {TEXT_SECONDARY}; font-size: 0.92rem; margin: 0.4rem 0 0;
    letter-spacing: 0.01em;
}}
.status-pill {{
    display: inline-flex; align-items: center; gap: 6px;
    background: {CYAN_DIM};
    border: 1px solid rgba(0, 217, 255, 0.3);
    color: {CYAN};
    padding: 4px 12px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 0.7rem;
}}
.status-pill::before {{
    content: ''; width: 6px; height: 6px; border-radius: 50%;
    background: {CYAN}; box-shadow: 0 0 8px {CYAN};
}}

/* ── KPI scoreboard cards ───────────────────────────────────────────────── */
.kpi-card {{
    background: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    position: relative;
    overflow: hidden;
    height: 100%;
    transition: border-color 0.2s, transform 0.2s;
}}
.kpi-card:hover {{
    border-color: {BORDER_GLOW};
    transform: translateY(-1px);
}}
.kpi-label {{
    font-size: 0.72rem; color: {TEXT_SECONDARY};
    text-transform: uppercase; letter-spacing: 0.1em;
    font-weight: 600; margin-bottom: 0.4rem;
}}
.kpi-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.9rem; font-weight: 700;
    color: {TEXT_PRIMARY}; line-height: 1;
    margin-bottom: 0.3rem;
}}
.kpi-value.cyan {{ color: {CYAN}; }}
.kpi-value.amber {{ color: {AMBER}; }}
.kpi-delta {{
    font-size: 0.72rem; color: {TEXT_MUTED};
    font-family: 'JetBrains Mono', monospace;
}}

/* ── Surface cards ──────────────────────────────────────────────────────── */
.surface-card {{
    background: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}}
.surface-card-deep {{
    background: {BG_DEEP};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}}

/* ── Info cards ─────────────────────────────────────────────────────────── */
.info-card {{
    background: rgba(0, 217, 255, 0.05);
    border: 1px solid rgba(0, 217, 255, 0.18);
    border-left: 3px solid {CYAN};
    padding: 0.85rem 1.1rem;
    border-radius: 8px;
    color: {TEXT_SECONDARY};
    font-size: 0.9rem; line-height: 1.6;
    margin: 0.5rem 0 1rem;
}}
.warn-card {{
    background: rgba(255, 179, 0, 0.04);
    border: 1px solid rgba(255, 179, 0, 0.15);
    border-left: 3px solid {AMBER};
    padding: 0.85rem 1.1rem;
    border-radius: 8px;
    color: {TEXT_SECONDARY};
    font-size: 0.88rem; line-height: 1.6;
    margin: 0.5rem 0 1rem;
}}

/* ── Section title with subtle underline ────────────────────────────────── */
.section-title {{
    margin-top: 2rem; margin-bottom: 1rem;
    display: flex; align-items: center; gap: 0.7rem;
}}
.section-title h3 {{
    margin: 0; font-size: 1.15rem; font-weight: 600;
}}
.section-title-line {{
    flex: 1; height: 1px;
    background: linear-gradient(90deg, {BORDER_GLOW}, transparent);
}}

/* ── Tech badges ────────────────────────────────────────────────────────── */
.tech-badge {{
    display: inline-block;
    background: rgba(0, 217, 255, 0.08);
    color: {CYAN};
    padding: 0.3rem 0.75rem; border-radius: 12px;
    font-size: 0.74rem; font-weight: 600;
    margin: 0.2rem 0.15rem;
    border: 1px solid rgba(0, 217, 255, 0.2);
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.02em;
}}

/* ── Streamlit element overrides ────────────────────────────────────────── */
.stSelectbox label, .stRadio label, .stSlider label {{
    color: {TEXT_SECONDARY} !important;
    font-weight: 600; font-size: 0.74rem;
    text-transform: uppercase; letter-spacing: 0.1em;
}}
div[data-baseweb="select"] > div {{
    background: {BG_SURFACE} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT_PRIMARY} !important;
    border-radius: 8px !important;
}}
div[data-baseweb="select"] > div:hover {{
    border-color: {BORDER_GLOW} !important;
}}
.stButton > button {{
    background: linear-gradient(135deg, {CYAN} 0%, {BLUE} 100%);
    color: #000000;
    border: none; border-radius: 8px;
    padding: 0.6rem 1.7rem;
    font-weight: 700;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.03em;
    box-shadow: 0 0 20px rgba(0, 217, 255, 0.25);
    transition: all 0.15s;
}}
.stButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 0 30px rgba(0, 217, 255, 0.45);
    color: #000000;
}}

/* ── Sidebar items ──────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] .stRadio > div {{
    gap: 0.25rem;
}}
section[data-testid="stSidebar"] .stRadio label {{
    color: {TEXT_SECONDARY} !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.93rem !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    padding: 0.55rem 0.85rem;
    border-radius: 8px;
    border-left: 3px solid transparent;
    transition: all 0.15s;
    display: flex !important;
    align-items: center;
    width: 100%;
}}
section[data-testid="stSidebar"] .stRadio label:hover {{
    background: {BG_HOVER}; color: {TEXT_PRIMARY} !important;
}}
section[data-testid="stSidebar"] .stRadio label > div:first-child {{
    margin-right: 0.65rem;
}}
section[data-testid="stSidebar"] .stRadio [role="radio"] {{
    visibility: visible !important;
}}

/* ── Dataframe ──────────────────────────────────────────────────────────── */
.stDataFrame {{ background: {BG_SURFACE}; border-radius: 8px; }}

/* ── Hide Streamlit chrome ──────────────────────────────────────────────── */
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}

/* ── Expander ───────────────────────────────────────────────────────────── */
.streamlit-expanderHeader, [data-testid="stExpander"] details summary {{
    background: {BG_SURFACE} !important;
    color: {TEXT_PRIMARY} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}}

/* ── Custom row separator ───────────────────────────────────────────────── */
.row-sep {{
    border-bottom: 1px solid rgba(255,255,255,0.04);
    padding: 10px 0;
}}

/* ── Team pick card (predictor) ─────────────────────────────────────────── */
.team-pick {{
    background: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 1.3rem 1rem 1.1rem;
    text-align: center;
    position: relative;
    transition: all 0.2s;
}}
.team-pick:hover {{ border-color: {BORDER_GLOW}; }}
.team-pick-label {{
    position: absolute; top: 12px; left: 14px;
    background: {BG_DEEP};
    color: {TEXT_SECONDARY};
    font-size: 0.65rem; font-weight: 600;
    padding: 3px 8px; border-radius: 4px;
    letter-spacing: 0.1em;
}}
.vs-orb {{
    display: inline-flex;
    align-items: center; justify-content: center;
    width: 60px; height: 60px;
    border-radius: 50%;
    background: {BG_DEEP};
    border: 2px solid {CYAN};
    color: {CYAN};
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem; font-weight: 700;
    box-shadow: 0 0 20px rgba(0, 217, 255, 0.25);
}}

/* ── Bracket card ───────────────────────────────────────────────────────── */
.bracket-card {{
    background: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.75rem;
    position: relative;
}}
.bracket-card.is-final {{
    border-color: rgba(255, 179, 0, 0.4);
    background: linear-gradient(135deg, {BG_SURFACE} 0%, rgba(255,179,0,0.04) 100%);
}}
.bracket-stage-label {{
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em;
    color: {CYAN}; padding: 2px 8px;
    background: {CYAN_DIM}; border-radius: 4px;
    display: inline-block;
}}
.bracket-stage-label.is-final {{
    color: {AMBER}; background: rgba(255, 179, 0, 0.1);
}}

/* ── Code block ─────────────────────────────────────────────────────────── */
code {{
    background: {BG_DEEP} !important;
    color: {CYAN} !important;
    padding: 2px 6px; border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85em;
}}
.stCodeBlock {{
    background: {BG_DEEP} !important;
    border: 1px solid {BORDER};
    border-radius: 8px;
}}

/* ── Caption text ───────────────────────────────────────────────────────── */
.stCaption, [data-testid="stCaptionContainer"] {{
    color: {TEXT_MUTED} !important;
    font-size: 0.82rem;
}}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# LOGO HELPERS
# ============================================================================

@st.cache_data
def get_logo_base64(team: str) -> str:
    fname = LOGO_FILES.get(team, "")
    if not fname:
        return ""
    path = ASSETS_PATH / fname
    if not path.exists():
        return ""
    mime = "image/png" if fname.lower().endswith(".png") else "image/jpeg"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{data}"


def logo_html(team: str, size: int = 40, with_ring: bool = False,
              ring_color: str = None) -> str:
    src = get_logo_base64(team)
    color = TEAM_COLORS.get(team, "#7c7c8a")
    abbr = TEAM_ABBR.get(team, team[:3].upper())

    ring_style = ""
    if with_ring:
        ring_color = ring_color or color
        ring_style = (f"box-shadow: 0 0 0 2px {BG_PAGE}, "
                      f"0 0 0 4px {ring_color}, "
                      f"0 0 18px {ring_color}55;")

    if src:
        return (
            f'<div style="display:inline-flex; align-items:center; '
            f'justify-content:center; width:{size}px; height:{size}px; '
            f'background:#FFFFFF; border-radius:50%; padding:{max(2, size//12)}px; '
            f'box-sizing:border-box; overflow:hidden; {ring_style}">'
            f'<img src="{src}" style="max-width:100%; max-height:100%; '
            f'object-fit:contain;"/></div>'
        )
    else:
        font_size = max(10, size // 3)
        text_color = "#000000" if color in ("#FFFF3C", "#FFEB3B", "#F7A721") else "#FFFFFF"
        return (
            f'<div style="display:inline-flex; align-items:center; '
            f'justify-content:center; width:{size}px; height:{size}px; '
            f'background:{color}; border-radius:50%; '
            f'color:{text_color}; font-weight:700; font-size:{font_size}px; '
            f'font-family:Inter,sans-serif; letter-spacing:-0.02em; {ring_style}">'
            f'{abbr}</div>'
        )


# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(show_spinner="Loading matches…")
def load_matches() -> pd.DataFrame:
    df = pd.read_csv(BASE_PATH / "matches_clean.csv", parse_dates=["match_date"])
    for mid, winner in FINAL_FIXES.items():
        df.loc[df["match_id"] == mid, "match_winner"] = winner
    return df


@st.cache_data(show_spinner="Loading ball-by-ball data (288K rows)…")
def load_deliveries() -> pd.DataFrame:
    df = pd.read_csv(BASE_PATH / "deliveries_clean.csv", low_memory=False)
    bool_cols = ["is_wicket", "is_wide_ball", "is_no_ball",
                 "is_leg_bye", "is_bye", "is_penalty", "is_super_over"]
    for col in bool_cols:
        if col not in df.columns:
            continue
        if df[col].dtype == bool:
            continue
        df[col] = (df[col].astype(str).str.strip().str.lower()
                   .isin(["true", "1", "1.0", "yes", "t"]))
    return df


@st.cache_data
def load_points_table():
    return pd.read_csv(BASE_PATH / "points_table_2026.csv")


@st.cache_data
def load_strength_table():
    return pd.read_csv(BASE_PATH / "strength_table_2026.csv")


@st.cache_data
def load_championship_prob():
    return pd.read_csv(BASE_PATH / "championship_prob_2026.csv")


@st.cache_resource
def load_model():
    try:
        with open(BASE_PATH / "rf_model.pkl", "rb") as f:
            model = pickle.load(f)
        if hasattr(model, "predict_proba"):
            return model
        return None
    except Exception as e:
        st.sidebar.warning(f"⚠️ rf_model.pkl: {e}")
        return None


# ============================================================================
# PLOTLY THEME
# ============================================================================

def plotly_cc(fig: go.Figure, height: int = 400) -> go.Figure:
    """Apply Command Center theme to a plotly figure."""
    fig.update_layout(
        paper_bgcolor=BG_SURFACE,
        plot_bgcolor=BG_SURFACE,
        font=dict(color=TEXT_PRIMARY, family="Inter, sans-serif", size=12),
        height=height,
        margin=dict(l=20, r=20, t=50, b=30),
        title=dict(font=dict(size=14, color=TEXT_PRIMARY,
                             family="Space Grotesk, sans-serif")),
        legend=dict(bgcolor="rgba(0,0,0,0)",
                    font=dict(color=TEXT_SECONDARY, size=11),
                    bordercolor=BORDER, borderwidth=0),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)",
                   zerolinecolor="rgba(255,255,255,0.08)",
                   tickfont=dict(color=TEXT_SECONDARY, size=11)),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)",
                   zerolinecolor="rgba(255,255,255,0.08)",
                   tickfont=dict(color=TEXT_SECONDARY, size=11)),
    )
    return fig


# ============================================================================
# HELPERS
# ============================================================================

def team_color(team: str) -> str:
    return TEAM_COLORS.get(team, "#7c7c8a")


def fmt_int(n) -> str:
    return f"{int(n):,}"


def fmt_compact(n) -> str:
    n = int(n)
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000: return f"{n/1_000:.0f}K"
    return str(n)


def hero(status: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero-card">'
        f'<div class="status-pill">{status}</div>'
        f'<h1>{title}</h1>'
        f'<p class="hero-sub">{subtitle}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def kpi(label: str, value: str, delta: str = "", color_class: str = "",
        delta_color: str = None) -> None:
    delta_color = delta_color or TEXT_MUTED
    delta_html = (f'<div class="kpi-delta" style="color:{delta_color};">{delta}</div>'
                  if delta else "")
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value {color_class}">{value}</div>'
        f'{delta_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def section(text: str) -> None:
    st.markdown(
        f'<div class="section-title">'
        f'<h3>{text}</h3>'
        f'<div class="section-title-line"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def detect_col(df: pd.DataFrame, candidates: list) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_lower: return cols_lower[cand]
    for cand in candidates:
        for low, orig in cols_lower.items():
            if cand in low: return orig
    return df.columns[0]


def render_points_table(df: pd.DataFrame, team_col: str, points_col: str,
                        title_cols: list = None) -> None:
    """Render a compact IPL table with team logos."""
    title_cols = title_cols or [c for c in df.columns if c != team_col]
    n_other = len(title_cols)
    header_labels = {
        "matches": "M",
        "projected_wins": "Proj Wins",
        "projected_losses": "Proj Losses",
        "projected_points": "Proj Pts",
    }
    grid_cols = f"55px minmax(260px, 1.9fr) repeat({n_other}, minmax(96px, 1fr))"
    pts_html = '<div class="surface-card" style="padding:0; overflow-x:auto;">'
    pts_html += (
        f'<div style="display:grid; grid-template-columns:{grid_cols}; min-width:980px; '
        f'gap:12px; padding:14px 18px; background:{BG_DEEP}; '
        f'color:{TEXT_SECONDARY}; font-size:0.66rem; font-weight:700; '
        f'text-transform:uppercase; letter-spacing:0.04em; '
        f'border-bottom:1px solid {BORDER};">'
        f'<div>Pos</div><div>Team</div>'
    )
    for col in title_cols:
        label = header_labels.get(col, str(col).replace("_", " ").title())
        pts_html += f'<div style="text-align:center; white-space:normal;">{label}</div>'
    pts_html += "</div>"

    for i, row in df.iterrows():
        team_name = row[team_col]
        tc = team_color(team_name)
        is_top4 = i < 4
        bg = "rgba(0, 217, 255, 0.03)" if is_top4 else "transparent"
        pts_html += (
            f'<div style="display:grid; grid-template-columns:{grid_cols}; min-width:980px; '
            f'gap:12px; padding:11px 18px; align-items:center; '
            f'border-bottom:1px solid {BORDER}; background:{bg}; '
            f'border-left:3px solid {tc};">'
            f'<div style="color:{CYAN if is_top4 else TEXT_MUTED}; '
            f'font-family:JetBrains Mono; font-weight:700; '
            f'font-size:0.95rem;">{i+1}</div>'
            f'<div style="display:grid; grid-template-columns:38px minmax(0, 1fr); '
            f'align-items:center; gap:12px;">'
            f'<div style="width:34px; height:34px; display:flex; align-items:center; '
            f'justify-content:center;">{logo_html(team_name, size=32)}</div>'
            f'<span style="color:{TEXT_PRIMARY}; font-weight:600; font-size:0.9rem; '
            f'line-height:1.25; overflow-wrap:anywhere;">{team_name}</span>'
            f'</div>'
        )
        for col in title_cols:
            val = row[col]
            if isinstance(val, float):
                val = f"{val:.0f}" if col == "projected_points" else f"{val:.1f}"
            is_pts = col == points_col or col == "projected_points"
            color = AMBER if is_pts else TEXT_PRIMARY
            pts_html += (
                f'<div style="text-align:center; font-family:JetBrains Mono; '
                f'color:{color}; font-weight:{700 if is_pts else 500}; '
                f'font-size:0.9rem;">{val}</div>'
            )
        pts_html += "</div>"
    pts_html += "</div>"
    st.markdown(pts_html, unsafe_allow_html=True)


# ============================================================================
# FEATURE ENGINEERING (live prediction)
# ============================================================================

@st.cache_data(show_spinner="Building team history…")
def build_team_history(matches: pd.DataFrame) -> pd.DataFrame:
    m = matches[(matches["match_winner"].notna()) & (matches["season"] != 2026)].copy()
    t1 = m[["match_id", "season", "match_date", "venue", "team1", "team2",
            "toss_winner", "match_winner"]].copy()
    t1["team"] = t1["team1"]; t1["opponent"] = t1["team2"]
    t1["won"] = (t1["match_winner"] == t1["team1"]).astype(int)
    t2 = m[["match_id", "season", "match_date", "venue", "team1", "team2",
            "toss_winner", "match_winner"]].copy()
    t2["team"] = t2["team2"]; t2["opponent"] = t2["team1"]
    t2["won"] = (t2["match_winner"] == t2["team2"]).astype(int)
    base = pd.concat([t1, t2], ignore_index=True)
    base["toss_won"] = (base["toss_winner"] == base["team"]).astype(int)
    return base.sort_values(["match_date", "match_id"]).reset_index(drop=True)


def latest_form(history, team, window=5):
    rows = history[history["team"] == team].sort_values("match_date").tail(window)
    return rows["won"].mean() if len(rows) else 0.5

def latest_overall_win_rate(history, team):
    rows = history[history["team"] == team]
    return rows["won"].mean() if len(rows) else 0.5

def latest_season_win_rate(history, team, season):
    last_season = history[history["team"] == team]["season"].max()
    if pd.isna(last_season): return 0.5
    rows = history[(history["team"] == team) & (history["season"] == last_season)]
    return rows["won"].mean() if len(rows) else 0.5

def latest_h2h_win_pct(history, team, opp):
    rows = history[(history["team"] == team) & (history["opponent"] == opp)]
    return rows["won"].mean() if len(rows) else 0.5

def latest_venue_win_pct(history, team, venue):
    rows = history[(history["team"] == team) & (history["venue"] == venue)]
    return rows["won"].mean() if len(rows) else 0.5


@st.cache_data(show_spinner="Computing h2h phase splits…")
def build_h2h_phase_stats(matches, deliveries):
    m = matches[(matches["match_winner"].notna())
                & (matches["season"] != 2026)][["match_id"]]
    d = deliveries.merge(m, on="match_id", how="inner")
    d["phase"] = d["over_number"].apply(
        lambda o: "powerplay" if o < 6 else ("middle" if o < 15 else "death"))
    legal = d[(~d["is_wide_ball"]) & (~d["is_no_ball"])]

    out = {}
    grp = (d.groupby(["team_batting", "team_bowling", "phase", "match_id"])
           .agg(runs=("total_runs", "sum")).reset_index()
           .groupby(["team_batting", "team_bowling", "phase"])
           .agg(runs_per_match=("runs", "mean")).reset_index())
    wkt = (legal[legal["is_wicket"]]
           .groupby(["team_batting", "team_bowling", "phase", "match_id"])
           .size().reset_index(name="wkts")
           .groupby(["team_batting", "team_bowling", "phase"])
           .agg(wkts_per_match=("wkts", "mean")).reset_index())

    for bat, bowl, ph, runs in zip(grp["team_batting"], grp["team_bowling"],
                                    grp["phase"], grp["runs_per_match"]):
        out.setdefault((bat, bowl), {})[f"h2h_runs_{ph}"] = runs
    for bat, bowl, ph, wkts in zip(wkt["team_batting"], wkt["team_bowling"],
                                    wkt["phase"], wkt["wkts_per_match"]):
        out.setdefault((bowl, bat), {})[f"h2h_wkts_{ph}"] = wkts
    return out


def get_h2h_phase(h2h_stats, team, opp):
    stats = h2h_stats.get((team, opp), {})
    defaults = {"h2h_runs_powerplay": 45.0, "h2h_runs_middle": 65.0,
                "h2h_runs_death": 50.0, "h2h_wkts_powerplay": 1.5,
                "h2h_wkts_middle": 3.0, "h2h_wkts_death": 2.5}
    return {k: stats.get(k, defaults[k]) for k in defaults}


def build_feature_row(history, h2h_stats, team, opp, venue, toss_winner,
                      season=2026):
    row = {
        "team_enc": TEAM_ENC.get(team, 0),
        "opponent_enc": TEAM_ENC.get(opp, 0),
        "toss_won": 1 if toss_winner == team else 0,
        "form": latest_form(history, team),
        "h2h_win_pct": latest_h2h_win_pct(history, team, opp),
        "venue_win_pct": latest_venue_win_pct(history, team, venue),
        "season": season,
        "overall_win_rate": latest_overall_win_rate(history, team),
        "season_win_rate": latest_season_win_rate(history, team, season),
    }
    row.update(get_h2h_phase(h2h_stats, team, opp))
    return pd.DataFrame([row])[ML_FEATURES]


def fallback_match_probability(row: pd.DataFrame) -> float:
    """Estimate win probability when the trained model pickle is unavailable."""
    r = row.iloc[0].fillna(0.5)
    score = (
        0.28 * float(r["overall_win_rate"]) +
        0.24 * float(r["season_win_rate"]) +
        0.22 * float(r["form"]) +
        0.18 * float(r["h2h_win_pct"]) +
        0.08 * float(r["venue_win_pct"])
    )
    return max(0.05, min(0.95, score))


def predict_pair_probability(model, row1: pd.DataFrame, row2: pd.DataFrame):
    if model is not None and hasattr(model, "predict_proba"):
        p1 = float(model.predict_proba(row1)[0, 1])
        p2 = float(model.predict_proba(row2)[0, 1])
    else:
        p1 = fallback_match_probability(row1)
        p2 = fallback_match_probability(row2)

    total = p1 + p2
    return (p1 / total, p2 / total) if total > 0 else (0.5, 0.5)


def project_final_table(points: pd.DataFrame, strength: pd.DataFrame,
                        team_col_p: str, team_col_s: str, str_col: str,
                        matches_per_team: int = 14) -> pd.DataFrame:
    current = points.copy()
    strength_map = dict(zip(strength[team_col_s], strength[str_col]))
    avg_strength = float(strength[str_col].mean()) if len(strength) else 0.5

    rows = []
    for _, row in current.iterrows():
        team = row[team_col_p]
        played = int(row.get("matches", row.get("Matches", 0)))
        wins = float(row.get("wins", row.get("Wins", 0)))
        losses = float(row.get("losses", row.get("Losses", max(0, played - wins))))
        remaining = max(0, matches_per_team - played)
        strength_score = float(strength_map.get(team, avg_strength))
        win_prob = strength_score / (strength_score + avg_strength) if avg_strength > 0 else 0.5
        projected_wins = wins + remaining * win_prob
        projected_losses = losses + remaining * (1 - win_prob)
        rows.append({
            "team": team,
            "matches": matches_per_team,
            "projected_wins": round(projected_wins, 1),
            "projected_losses": round(projected_losses, 1),
            "projected_points": int(round(projected_wins * 2)),
        })

    return (pd.DataFrame(rows)
            .sort_values(["projected_points", "projected_wins"], ascending=False)
            .reset_index(drop=True))


# ============================================================================
# PAGE 1 — HOME
# ============================================================================

def page_home(matches, deliveries):
    hero("System Online", "IPL Analytics",
         "2008 – 2026 &nbsp;·&nbsp; 288K deliveries &nbsp;·&nbsp; ML-powered predictions")

    total_matches = len(matches)
    total_seasons = matches["season"].nunique()
    total_runs = int(deliveries["batter_runs"].sum())
    total_sixes = int((deliveries["batter_runs"] == 6).sum())
    runs_2026 = int(deliveries[deliveries["season_id"] == 2026]["batter_runs"].sum())
    matches_2026 = int((matches["season"] == 2026).sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Matches", fmt_int(total_matches),
                 f"+{matches_2026} in 2026", "cyan", CYAN)
    with c2: kpi("Seasons", fmt_int(total_seasons),
                 "2008 → 2026", "", TEXT_SECONDARY)
    with c3: kpi("Total Runs", fmt_compact(total_runs),
                 f"+{fmt_compact(runs_2026)} in 2026", "cyan", CYAN)
    with c4: kpi("Total Sixes", fmt_int(total_sixes),
                 "T20 era peak", "amber", AMBER)

    st.markdown("###  ")
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown(
            f'<div class="surface-card">'
            f'<h3 style="margin-top:0;">About this project</h3>'
            f'<p style="color:{TEXT_SECONDARY}; line-height:1.65; margin-bottom:0.8rem;">'
            f'End-to-end IPL analytics platform covering every ball bowled from '
            f'2008 to 2026. The pipeline includes raw data cleaning across 288K '
            f'deliveries, feature engineering with strict no-leakage rolling '
            f'statistics, a Random Forest classifier trained on 15 engineered '
            f'features with a time-based train/test split, and a Monte Carlo '
            f'playoff simulator.'
            f'</p>'
            f'<p style="color:{TEXT_SECONDARY}; line-height:1.65; margin:0;">'
            f'Use the sidebar to explore historical EDA, drill into team-specific '
            f'stats, run match predictions, simulate the 2026 playoffs, or read '
            f'the methodology behind the model.'
            f'</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="surface-card-deep">'
            f'<h4 style="margin-top:0; font-size:0.95rem;">Tech Stack</h4>'
            f'<span class="tech-badge">Python</span>'
            f'<span class="tech-badge">pandas</span>'
            f'<span class="tech-badge">scikit-learn</span>'
            f'<span class="tech-badge">XGBoost</span>'
            f'<span class="tech-badge">Plotly</span>'
            f'<span class="tech-badge">Streamlit</span>'
            f'<span class="tech-badge">Monte Carlo</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    section("Active teams · 2026")
    logos_html = (
        '<div class="surface-card-deep" '
        'style="display:grid; grid-template-columns:repeat(auto-fit, minmax(120px, 1fr)); '
        'gap:22px 18px; align-items:start;">'
    )
    for t in ACTIVE_TEAMS_2026:
        logos_html += (
            f'<div style="text-align:center; min-height:96px; display:flex; '
            f'flex-direction:column; align-items:center; justify-content:flex-start;">'
            f'<div style="width:68px; height:68px; display:flex; align-items:center; '
            f'justify-content:center;">'
            f'{logo_html(t, size=60, with_ring=True)}'
            f'</div>'
            f'<div style="color:{TEXT_SECONDARY}; font-size:0.72rem; '
            f'margin-top:8px; font-weight:600; letter-spacing:0.05em;">'
            f'{TEAM_ABBR[t]}</div>'
            f'</div>'
        )
    logos_html += "</div>"
    st.markdown(logos_html, unsafe_allow_html=True)

    section("Most successful teams · all-time wins")
    wins = (matches.dropna(subset=["match_winner"])
            .groupby("match_winner").size()
            .sort_values(ascending=False).head(5)
            .reset_index(name="wins"))
    fig = go.Figure(go.Bar(
        x=wins["match_winner"], y=wins["wins"],
        marker=dict(color=[team_color(t) for t in wins["match_winner"]],
                    line=dict(color=BG_PAGE, width=2)),
        text=wins["wins"], textposition="outside",
        textfont=dict(color=TEXT_PRIMARY, size=13, family="JetBrains Mono"),
    ))
    fig.update_layout(yaxis_title="Wins", xaxis_title=None)
    st.plotly_chart(plotly_cc(fig, 420), use_container_width=True)


# ============================================================================
# PAGE 2 — EDA
# ============================================================================

def page_eda(matches, deliveries):
    hero("Analytics Active", "EDA Dashboard",
         "Eight visualizations across team performance, scoring patterns, and player dominance")

    # 1
    section("1 · Team win % (2008–2025)")
    m = matches[(matches["match_winner"].notna()) & (matches["season"] != 2026)]
    played = m["team1"].value_counts().add(m["team2"].value_counts(), fill_value=0)
    won = m["match_winner"].value_counts()
    played = played.reindex(won.index).fillna(won)
    win_pct = (won / played * 100).round(2).sort_values(ascending=True)

    fig1 = go.Figure(go.Bar(
        x=win_pct.values, y=win_pct.index, orientation="h",
        marker=dict(color=[team_color(t) for t in win_pct.index],
                    line=dict(color=BG_PAGE, width=1)),
        text=[f"{v:.1f}%  ({int(played.get(t, 0))} M)" for t, v in win_pct.items()],
        textposition="outside",
        textfont=dict(color=TEXT_PRIMARY, family="JetBrains Mono", size=11),
    ))
    fig1.add_vline(x=50, line_dash="dash", line_color=CYAN, line_width=1.5)
    fig1.update_layout(xaxis_title="Win %", yaxis_title=None,
                       xaxis=dict(range=[0, max(win_pct.values) + 22]))
    st.plotly_chart(plotly_cc(fig1, 550), use_container_width=True)

    # 2
    section("2 · Toss impact")
    toss_m = matches[matches["match_winner"].notna()]
    won_after = int((toss_m["toss_winner"] == toss_m["match_winner"]).sum())
    lost_after = int((toss_m["toss_winner"] != toss_m["match_winner"]).sum())
    decision = matches["toss_decision"].value_counts()

    c1, c2, c3 = st.columns(3)
    with c1:
        fig_p = go.Figure(go.Pie(
            labels=["Toss winner won", "Toss winner lost"],
            values=[won_after, lost_after],
            marker_colors=[CYAN, RED], hole=0.6,
            textfont=dict(color=TEXT_PRIMARY, family="Inter"),
        ))
        fig_p.update_layout(title="Did toss winner win?",
                            legend=dict(orientation="h", y=-0.05))
        st.plotly_chart(plotly_cc(fig_p, 320), use_container_width=True)
    with c2:
        fig_d = go.Figure(go.Bar(
            x=decision.index, y=decision.values,
            marker_color=[CYAN, AMBER][:len(decision)],
            text=decision.values, textposition="outside",
            textfont=dict(color=TEXT_PRIMARY, family="JetBrains Mono"),
        ))
        fig_d.update_layout(title="Bat vs Field preference",
                            yaxis_title="Matches", xaxis_title=None)
        st.plotly_chart(plotly_cc(fig_d, 320), use_container_width=True)
    with c3:
        won_by_dec = toss_m[toss_m["toss_winner"] == toss_m["match_winner"]]["toss_decision"].value_counts()
        total_by_dec = toss_m["toss_decision"].value_counts()
        pct_by_dec = (won_by_dec / total_by_dec * 100).round(1)
        fig_dec = go.Figure(go.Bar(
            x=pct_by_dec.index, y=pct_by_dec.values,
            marker_color=[CYAN, AMBER][:len(pct_by_dec)],
            text=[f"{v}%" for v in pct_by_dec.values],
            textposition="outside",
            textfont=dict(color=TEXT_PRIMARY, family="JetBrains Mono"),
        ))
        fig_dec.update_layout(title="Win % by toss decision",
                              yaxis_title="Win %", xaxis_title=None,
                              yaxis=dict(range=[0, 100]))
        st.plotly_chart(plotly_cc(fig_dec, 320), use_container_width=True)

    # 3
    section("3 · Top players")
    top_b = (deliveries[~deliveries["is_wide_ball"]]
             .groupby("batter")["batter_runs"].sum()
             .sort_values(ascending=False).head(10).sort_values(ascending=True))
    top_w = (deliveries[(deliveries["is_wicket"])
                        & (deliveries["wicket_kind"].fillna("").str.lower() != "run out")]
             .groupby("bowler").size()
             .sort_values(ascending=False).head(10).sort_values(ascending=True))

    cb1, cb2 = st.columns(2)
    with cb1:
        fig_b = go.Figure(go.Bar(
            x=top_b.values, y=top_b.index, orientation="h",
            marker_color=CYAN,
            text=[fmt_int(v) for v in top_b.values], textposition="outside",
            textfont=dict(color=TEXT_PRIMARY, family="JetBrains Mono", size=11),
        ))
        fig_b.update_layout(title="Top 10 run scorers", xaxis_title="Runs",
                            yaxis_title=None)
        st.plotly_chart(plotly_cc(fig_b, 480), use_container_width=True)
    with cb2:
        fig_w = go.Figure(go.Bar(
            x=top_w.values, y=top_w.index, orientation="h",
            marker_color=AMBER,
            text=top_w.values, textposition="outside",
            textfont=dict(color=TEXT_PRIMARY, family="JetBrains Mono", size=11),
        ))
        fig_w.update_layout(title="Top 10 wicket takers (excl. run-outs)",
                            xaxis_title="Wickets", yaxis_title=None)
        st.plotly_chart(plotly_cc(fig_w, 480), use_container_width=True)

    # 4 — era
    section("4 · How IPL scoring has evolved")
    def era(s):
        return "Era 1 · 2008–2013" if s <= 2013 else (
            "Era 2 · 2014–2019" if s <= 2019 else "Era 3 · 2021–2025")
    d_legal = deliveries[(~deliveries["is_wide_ball"])
                         & (~deliveries["is_no_ball"])
                         & (deliveries["season_id"] != 2026)].copy()
    d_legal["era"] = d_legal["season_id"].apply(era)
    era_over = (d_legal.groupby(["era", "over_number"])["total_runs"]
                .mean().reset_index())
    era_over["runs_per_over"] = era_over["total_runs"] * 6
    fig_era = go.Figure()
    era_colors = {"Era 1 · 2008–2013": "#6B7A99",
                  "Era 2 · 2014–2019": CYAN,
                  "Era 3 · 2021–2025": AMBER}
    for e, grp in era_over.groupby("era"):
        fig_era.add_trace(go.Scatter(
            x=grp["over_number"] + 1, y=grp["runs_per_over"],
            mode="lines+markers", name=e,
            line=dict(color=era_colors[e], width=2.5),
            marker=dict(size=7, color=era_colors[e]),
        ))
    fig_era.add_vrect(x0=1, x1=6, fillcolor=CYAN, opacity=0.05, line_width=0)
    fig_era.add_vrect(x0=16, x1=20, fillcolor=AMBER, opacity=0.05, line_width=0)
    fig_era.update_layout(title="Average runs per over · powerplay & death shaded",
                          xaxis_title="Over", yaxis_title="Runs / Over")
    st.plotly_chart(plotly_cc(fig_era, 440), use_container_width=True)

    # 5
    section("5 · Boundaries by season")
    fs = (deliveries[~deliveries["is_wide_ball"]]
          .groupby("season_id")
          .agg(fours=("batter_runs", lambda x: (x == 4).sum()),
               sixes=("batter_runs", lambda x: (x == 6).sum()))
          .reset_index().query("season_id != 2026"))
    fig_fs = go.Figure()
    fig_fs.add_trace(go.Scatter(x=fs["season_id"], y=fs["fours"],
                                mode="lines+markers", name="Fours",
                                line=dict(color=CYAN, width=2.5),
                                marker=dict(size=8)))
    fig_fs.add_trace(go.Scatter(x=fs["season_id"], y=fs["sixes"],
                                mode="lines+markers", name="Sixes",
                                line=dict(color=AMBER, width=2.5),
                                marker=dict(size=8)))
    fig_fs.update_layout(
        title=f"Fours: {fmt_int(fs['fours'].sum())}  ·  "
              f"Sixes: {fmt_int(fs['sixes'].sum())}",
        xaxis_title="Season", yaxis_title="Count")
    st.plotly_chart(plotly_cc(fig_fs, 400), use_container_width=True)

    # 6
    section("6 · Dismissal types")
    dismissals = (deliveries[deliveries["is_wicket"]]
                  .groupby("wicket_kind").size()
                  .sort_values(ascending=True).reset_index(name="count"))
    dismissals["pct"] = (dismissals["count"] / dismissals["count"].sum() * 100).round(1)
    palette = [CYAN, BLUE, AMBER, GREEN, "#FF8B6B", RED, "#9F7AEA", "#7986CB", "#A0AEC0"]
    fig_dis = go.Figure(go.Bar(
        x=dismissals["pct"], y=dismissals["wicket_kind"], orientation="h",
        marker_color=palette[:len(dismissals)],
        text=[f"{p}%  ({fmt_int(c)})" for p, c in zip(dismissals["pct"], dismissals["count"])],
        textposition="outside",
        textfont=dict(color=TEXT_PRIMARY, family="JetBrains Mono", size=11),
    ))
    fig_dis.update_layout(title=f"Dismissals · {fmt_int(dismissals['count'].sum())} total",
                          xaxis_title="% of dismissals", yaxis_title=None,
                          xaxis=dict(range=[0, dismissals["pct"].max() + 25]))
    st.plotly_chart(plotly_cc(fig_dis, 480), use_container_width=True)

    # 7
    section("7 · Dot ball trend")
    dots = (deliveries[(~deliveries["is_wide_ball"])
                       & (~deliveries["is_no_ball"])
                       & (deliveries["season_id"] != 2026)]
            .groupby("season_id")
            .agg(total=("batter_runs", "count"),
                 dots=("batter_runs", lambda x: (x == 0).sum()))
            .reset_index())
    dots["dot_pct"] = (dots["dots"] / dots["total"] * 100).round(2)
    fig_dot = go.Figure(go.Scatter(
        x=dots["season_id"], y=dots["dot_pct"], mode="lines+markers",
        line=dict(color=AMBER, width=3),
        marker=dict(size=9, color=AMBER, line=dict(color=TEXT_PRIMARY, width=1.5)),
        fill="tozeroy", fillcolor="rgba(255, 179, 0, 0.10)",
    ))
    max_row = dots.loc[dots["dot_pct"].idxmax()]
    min_row = dots.loc[dots["dot_pct"].idxmin()]
    fig_dot.add_annotation(x=max_row["season_id"], y=max_row["dot_pct"],
                           text=f"High {max_row['dot_pct']}%",
                           font=dict(color=RED, family="JetBrains Mono"),
                           yshift=15, showarrow=False)
    fig_dot.add_annotation(x=min_row["season_id"], y=min_row["dot_pct"],
                           text=f"Low {min_row['dot_pct']}%",
                           font=dict(color=GREEN, family="JetBrains Mono"),
                           yshift=-20, showarrow=False)
    fig_dot.update_layout(title="Dot ball % by season",
                          xaxis_title="Season", yaxis_title="Dot %")
    st.plotly_chart(plotly_cc(fig_dot, 400), use_container_width=True)

    # 8
    section("8 · First innings score inflation")
    legal = deliveries[(~deliveries["is_wide_ball"]) & (~deliveries["is_no_ball"])]
    first_inn = legal[legal["innings"] == 1]
    inn_totals = (first_inn.groupby(["season_id", "match_id"])["total_runs"]
                  .sum().reset_index())
    avg_first = inn_totals.groupby("season_id")["total_runs"].mean().reset_index()
    fig_fi = go.Figure(go.Scatter(
        x=avg_first["season_id"], y=avg_first["total_runs"],
        mode="lines+markers",
        line=dict(color=CYAN, width=3),
        marker=dict(size=9, color=CYAN, line=dict(color=TEXT_PRIMARY, width=1.5)),
        fill="tozeroy", fillcolor="rgba(0, 217, 255, 0.10)",
    ))
    fig_fi.update_layout(title="Average first innings score",
                         xaxis_title="Season", yaxis_title="Avg Score")
    st.plotly_chart(plotly_cc(fig_fi, 400), use_container_width=True)


# ============================================================================
# PAGE 3 — TEAM ANALYSIS
# ============================================================================

def page_team_analysis(matches, deliveries):
    hero("Team Scope Active", "Team Analysis",
         "Deep dive into any franchise — history, top players, venues, head-to-head")

    all_teams = sorted(set(matches["team1"]).union(matches["team2"]))
    default_idx = all_teams.index("Mumbai Indians") if "Mumbai Indians" in all_teams else 0
    team = st.selectbox("Select team", all_teams, index=default_idx,
                        label_visibility="collapsed")

    trophies = sum(1 for champion in IPL_CHAMPIONS.values() if champion == team)

    team_matches = matches[(matches["team1"] == team) | (matches["team2"] == team)]
    total_matches = len(team_matches)
    total_wins = int((matches["match_winner"] == team).sum())
    win_pct = (total_wins / total_matches * 100) if total_matches else 0
    tc = team_color(team)

    # Header card with logo
    st.markdown(
        f'<div class="surface-card" style="border-left: 3px solid {tc};">'
        f'<div style="display:flex; align-items:center; gap:1.3rem;">'
        f'{logo_html(team, size=80, with_ring=True)}'
        f'<div>'
        f'<div style="color:{CYAN}; font-size:0.7rem; font-weight:600; '
        f'letter-spacing:0.1em; text-transform:uppercase; margin-bottom:4px;">'
        f'Selected Franchise</div>'
        f'<h2 style="margin:0; color:{TEXT_PRIMARY}; font-size:1.6rem;">{team}</h2>'
        f'<p style="color:{TEXT_SECONDARY}; margin:0.2rem 0 0; font-size:0.88rem; '
        f'font-family:JetBrains Mono;">'
        f'{TEAM_ABBR.get(team, "")} &nbsp;·&nbsp; '
        f'{total_matches} matches &nbsp;·&nbsp; {win_pct:.1f}% wins '
        f'&nbsp;·&nbsp; {trophies} {"trophies" if trophies != 1 else "trophy"}'
        f'</p>'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Matches", fmt_int(total_matches))
    with c2: kpi("Wins", fmt_int(total_wins), "", "cyan")
    with c3: kpi("Win %", f"{win_pct:.1f}%",
                 "Above 50%" if win_pct >= 50 else "Below 50%",
                 "", GREEN if win_pct >= 50 else RED)
    with c4: kpi("Trophies", str(trophies),
                 "★ Champion" if trophies > 0 else "—",
                 "amber" if trophies > 0 else "",
                 AMBER if trophies > 0 else TEXT_MUTED)

    section("Win % by season")
    season_played = team_matches.groupby("season").size()
    season_won = matches[matches["match_winner"] == team].groupby("season").size()
    season_pct = (season_won / season_played * 100).fillna(0).reset_index()
    season_pct.columns = ["season", "win_pct"]

    rgb = tuple(int(tc[i:i+2], 16) for i in (1, 3, 5))
    fill = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.18)"
    fig_season = go.Figure(go.Scatter(
        x=season_pct["season"], y=season_pct["win_pct"],
        mode="lines+markers",
        line=dict(color=tc, width=3),
        marker=dict(size=9, color=tc, line=dict(color=TEXT_PRIMARY, width=1.5)),
        fill="tozeroy", fillcolor=fill,
    ))
    fig_season.add_hline(y=50, line_dash="dash", line_color=CYAN, line_width=1.5)
    fig_season.update_layout(xaxis_title="Season", yaxis_title="Win %",
                             yaxis=dict(range=[0, 100]))
    st.plotly_chart(plotly_cc(fig_season, 380), use_container_width=True)

    section("Top players")
    td = deliveries[deliveries["team_batting"] == team]
    top_b = (td.groupby("batter")["batter_runs"].sum()
             .sort_values(ascending=False).head(5).reset_index())
    bp = deliveries[(deliveries["team_bowling"] == team)
                    & (deliveries["is_wicket"])
                    & (deliveries["wicket_kind"].fillna("").str.lower() != "run out")]
    top_w = (bp.groupby("bowler").size().sort_values(ascending=False).head(5)
             .reset_index(name="wickets"))

    bcol1, bcol2 = st.columns(2)
    with bcol1:
        fig_b = go.Figure(go.Bar(
            x=top_b["batter_runs"], y=top_b["batter"], orientation="h",
            marker_color=tc, text=top_b["batter_runs"], textposition="outside",
            textfont=dict(color=TEXT_PRIMARY, family="JetBrains Mono", size=11),
        ))
        fig_b.update_layout(title="Top 5 batters · runs",
                            xaxis_title="Runs", yaxis_title=None,
                            yaxis=dict(autorange="reversed"))
        st.plotly_chart(plotly_cc(fig_b, 380), use_container_width=True)
    with bcol2:
        fig_w = go.Figure(go.Bar(
            x=top_w["wickets"], y=top_w["bowler"], orientation="h",
            marker_color=tc, text=top_w["wickets"], textposition="outside",
            textfont=dict(color=TEXT_PRIMARY, family="JetBrains Mono", size=11),
        ))
        fig_w.update_layout(title="Top 5 bowlers · wickets (excl. run-outs)",
                            xaxis_title="Wickets", yaxis_title=None,
                            yaxis=dict(autorange="reversed"))
        st.plotly_chart(plotly_cc(fig_w, 380), use_container_width=True)

    # Venue
    section("Best & worst venues")
    venue_played = team_matches.groupby("venue").size()
    venue_won = matches[matches["match_winner"] == team].groupby("venue").size()
    venue_df = pd.DataFrame({"played": venue_played, "won": venue_won}).fillna(0)
    venue_df["win_pct"] = (venue_df["won"] / venue_df["played"] * 100).round(1)
    venue_df = venue_df[venue_df["played"] >= 5].sort_values("win_pct", ascending=True)

    if len(venue_df) > 0:
        fig_v = go.Figure(go.Bar(
            x=venue_df["win_pct"], y=venue_df.index, orientation="h",
            marker_color=tc,
            text=[f"{p}%  ({int(w)}/{int(pl)})" for p, w, pl in
                  zip(venue_df["win_pct"], venue_df["won"], venue_df["played"])],
            textposition="outside",
            textfont=dict(color=TEXT_PRIMARY, family="JetBrains Mono", size=10),
        ))
        fig_v.add_vline(x=50, line_dash="dash", line_color=CYAN)
        fig_v.update_layout(title=f"Win % by venue · min 5 matches",
                            xaxis_title="Win %", yaxis_title=None,
                            xaxis=dict(range=[0, 120]))
        st.plotly_chart(plotly_cc(fig_v, max(380, 26 * len(venue_df))),
                        use_container_width=True)

    # H2H with logos
    section("Head-to-head record")
    h2h_rows = []
    for opp in all_teams:
        if opp == team: continue
        h2h = matches[((matches["team1"] == team) & (matches["team2"] == opp))
                      | ((matches["team1"] == opp) & (matches["team2"] == team))]
        if len(h2h) == 0: continue
        wins_h = int((h2h["match_winner"] == team).sum())
        h2h_rows.append({"opponent": opp, "wins": wins_h, "played": len(h2h),
                         "win_pct": wins_h / len(h2h) * 100})

    h2h_df = pd.DataFrame(h2h_rows).sort_values("win_pct", ascending=False)
    h2h_html = '<div class="surface-card">'
    for _, row in h2h_df.iterrows():
        opp = row["opponent"]
        w, p = int(row["wins"]), int(row["played"])
        pct = row["win_pct"]
        bar_color = GREEN if pct >= 50 else RED
        h2h_html += (
            f'<div style="display:flex; align-items:center; gap:14px; '
            f'padding:9px 0; border-bottom:1px solid {BORDER};">'
            f'<div style="flex-shrink:0;">{logo_html(opp, size=34)}</div>'
            f'<div style="flex:1; min-width:0;">'
            f'<div style="display:flex; justify-content:space-between; '
            f'margin-bottom:4px;">'
            f'<span style="color:{TEXT_PRIMARY}; font-weight:500; font-size:0.9rem;">{opp}</span>'
            f'<span style="font-family:JetBrains Mono; color:{bar_color}; '
            f'font-weight:700; font-size:0.85rem;">'
            f'{w}/{p} &nbsp;·&nbsp; {pct:.1f}%</span>'
            f'</div>'
            f'<div style="background:rgba(255,255,255,0.05); border-radius:3px; '
            f'height:5px; overflow:hidden;">'
            f'<div style="background:{bar_color}; width:{pct}%; height:100%;"></div>'
            f'</div>'
            f'</div>'
            f'</div>'
        )
    h2h_html += "</div>"
    st.markdown(h2h_html, unsafe_allow_html=True)


# ============================================================================
# PAGE 4 — MATCH PREDICTOR
# ============================================================================

def page_predictor(matches, deliveries):
    hero("Predictor Ready", "Match Predictor",
         "Random Forest classifier &nbsp;·&nbsp; 15 engineered features &nbsp;·&nbsp; ~51% test accuracy")

    st.markdown(
        '<div class="warn-card">⚠️ <b>Disclaimer:</b> Test accuracy is ~51% '
        'on the 2024–25 hold-out. T20 cricket has high irreducible variance — '
        'this is indicative, not a betting tool. The model is useful because '
        'it identifies <i>relative</i> strengths between teams.</div>',
        unsafe_allow_html=True,
    )

    model = load_model()
    history = build_team_history(matches)
    h2h_stats = build_h2h_phase_stats(matches, deliveries)

    c1, c2, c3 = st.columns(3)
    with c1:
        team1 = st.selectbox("Team 1", ACTIVE_TEAMS_2026, index=0, key="t1")
    with c2:
        opps = [t for t in ACTIVE_TEAMS_2026 if t != team1]
        team2 = st.selectbox("Team 2", opps, index=0, key="t2")
    with c3:
        venues = sorted(matches["venue"].dropna().unique())
        venue = st.selectbox("Venue", venues, key="venue")

    # Team pick cards
    st.markdown("###  ")
    pcol1, pcol2, pcol3 = st.columns([2, 0.7, 2])
    with pcol1:
        st.markdown(
            f'<div class="team-pick">'
            f'<div class="team-pick-label">TEAM 1</div>'
            f'<div style="display:flex; justify-content:center; margin:0.5rem 0 0.8rem;">'
            f'{logo_html(team1, size=92, with_ring=True)}'
            f'</div>'
            f'<div style="color:{TEXT_PRIMARY}; font-weight:700; '
            f'font-family:Space Grotesk; font-size:1.05rem;">{team1}</div>'
            f'<div style="color:{TEXT_MUTED}; font-size:0.78rem; margin-top:4px; '
            f'font-family:JetBrains Mono;">{TEAM_ABBR.get(team1, "")} · {venue.split(",")[0]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with pcol2:
        st.markdown(
            f'<div style="text-align:center; padding-top:46px;">'
            f'<div class="vs-orb">VS</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with pcol3:
        st.markdown(
            f'<div class="team-pick">'
            f'<div class="team-pick-label">TEAM 2</div>'
            f'<div style="display:flex; justify-content:center; margin:0.5rem 0 0.8rem;">'
            f'{logo_html(team2, size=92, with_ring=True)}'
            f'</div>'
            f'<div style="color:{TEXT_PRIMARY}; font-weight:700; '
            f'font-family:Space Grotesk; font-size:1.05rem;">{team2}</div>'
            f'<div style="color:{TEXT_MUTED}; font-size:0.78rem; margin-top:4px; '
            f'font-family:JetBrains Mono;">{TEAM_ABBR.get(team2, "")} · away</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("###  ")
    toss = st.radio("Toss Winner", [team1, team2], horizontal=True)

    if st.button("Run Prediction"):
        row1 = build_feature_row(history, h2h_stats, team1, team2, venue, toss).fillna(0.5)
        row2 = build_feature_row(history, h2h_stats, team2, team1, venue, toss).fillna(0.5)
        try:
            p1n, p2n = predict_pair_probability(model, row1, row2)
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            st.dataframe(row1)
            return

        winner = team1 if p1n > p2n else team2
        winner_color = team_color(winner)

        # Result card
        st.markdown(
            f'<div class="surface-card" style="border-left:3px solid {AMBER};">'
            f'<div style="text-align:center;">'
            f'<div style="color:{CYAN}; font-size:0.7rem; font-weight:600; '
            f'letter-spacing:0.12em; text-transform:uppercase;">Predicted Winner</div>'
            f'<div style="display:flex; align-items:center; justify-content:center; '
            f'gap:14px; margin-top:10px;">'
            f'{logo_html(winner, size=56, with_ring=True)}'
            f'<h2 style="margin:0; color:{TEXT_PRIMARY};">{winner}</h2>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Probability bars with logos
        for tn, prob in [(team1, p1n), (team2, p2n)]:
            tc = team_color(tn)
            st.markdown(
                f'<div style="margin-bottom:14px;">'
                f'<div style="display:flex; align-items:center; gap:12px; margin-bottom:6px;">'
                f'<div style="flex-shrink:0;">{logo_html(tn, size=30)}</div>'
                f'<div style="flex:1; display:flex; justify-content:space-between;">'
                f'<span style="color:{TEXT_PRIMARY}; font-weight:500;">{tn}</span>'
                f'<span style="font-family:JetBrains Mono; color:{tc}; '
                f'font-weight:700; font-size:1.1rem;">{prob*100:.1f}%</span>'
                f'</div>'
                f'</div>'
                f'<div style="background:rgba(255,255,255,0.06); border-radius:5px; '
                f'height:10px; overflow:hidden; margin-left:42px;">'
                f'<div style="background:{tc}; width:{prob*100}%; height:100%; '
                f'border-radius:5px;"></div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with st.expander("See the 15 features sent to the model"):
            display = pd.DataFrame({
                "Feature": ML_FEATURES,
                f"{team1}": row1.iloc[0].round(3).astype(str).values,
                f"{team2}": row2.iloc[0].round(3).astype(str).values,
            })
            st.dataframe(display, use_container_width=True, hide_index=True)
            st.caption("Features computed live from matches_clean.csv and "
                       "deliveries_clean.csv. Rolling stats use all completed "
                       "matches through 2025.")


# ============================================================================
# PAGE 5 — IPL 2026 SIMULATOR
# ============================================================================

def simulate_match(a, b, smap, rng):
    sa, sb = smap.get(a, 50), smap.get(b, 50)
    return a if rng.random() < sa / (sa + sb) else b


def simulate_playoff_once(top4, smap, rng):
    t1, t2, t3, t4 = top4
    q1w = simulate_match(t1, t2, smap, rng)
    q1l = t2 if q1w == t1 else t1
    ew = simulate_match(t3, t4, smap, rng)
    q2w = simulate_match(q1l, ew, smap, rng)
    return simulate_match(q1w, q2w, smap, rng)


def page_simulator(points, strength, champ_prob):
    hero("Simulator Ready", "IPL 2026 Simulator",
         "Points table &nbsp;·&nbsp; Strength scores &nbsp;·&nbsp; Playoff bracket &nbsp;·&nbsp; Monte Carlo championship")

    points.columns = [c.strip() for c in points.columns]
    strength.columns = [c.strip() for c in strength.columns]
    champ_prob.columns = [c.strip() for c in champ_prob.columns]

    team_col_p = detect_col(points, ["team", "team_name"])
    points_col = detect_col(points, ["points", "pts"])
    position_col = next((c for c in points.columns if c.lower() == "position"), None)
    team_col_s = detect_col(strength, ["team", "team_name"])
    str_col = detect_col(strength, ["strength", "score"])
    recent_col = next((c for c in strength.columns if "recent" in c.lower()), None)
    hist_col = next((c for c in strength.columns if "hist" in c.lower()), None)
    team_col_c = detect_col(champ_prob, ["team", "team_name"])
    prob_col = detect_col(champ_prob, ["champ", "prob", "pct"])

    # ── Section 1: Points table ────────────────────────────────────────
    if position_col:
        points_sorted = points.sort_values(position_col, ascending=True).reset_index(drop=True)
    else:
        points_sorted = points.sort_values(points_col, ascending=False).reset_index(drop=True)

    # ── Section 2: Strength ────────────────────────────────────────────
    section("Projected final table after 14 matches")
    projected = project_final_table(
        points_sorted,
        strength,
        team_col_p=team_col_p,
        team_col_s=team_col_s,
        str_col=str_col,
        matches_per_team=14,
    )
    st.markdown(
        '<div class="info-card">'
        'This projection completes each team to <b>14 league matches</b>. '
        'Remaining wins are estimated from the blended strength score, '
        'then added to the current 2026 table.'
        '</div>',
        unsafe_allow_html=True,
    )
    render_points_table(
        projected,
        team_col="team",
        points_col="projected_points",
        title_cols=[
            "matches", "projected_wins", "projected_losses",
            "projected_points",
        ],
    )

    section("Team strength score")
    st.markdown(
        '<div class="info-card"><b>Formula:</b> '
        '<code>strength = 0.6 × recent_win_pct + 0.4 × historical_win_pct</code><br>'
        'Recent = 2023–2025. Historical = 2008–2022. Favors current form (60%) '
        'but anchors on long-run quality (40%).'
        '</div>',
        unsafe_allow_html=True,
    )

    if recent_col and hist_col:
        show_df = strength[[team_col_s, recent_col, hist_col, str_col]].copy()
        show_df = show_df.sort_values(str_col, ascending=False).reset_index(drop=True)
        st.dataframe(show_df, use_container_width=True, hide_index=True)

    strength_sorted = strength.sort_values(str_col, ascending=True)
    fig_str = go.Figure(go.Bar(
        x=strength_sorted[str_col], y=strength_sorted[team_col_s],
        orientation="h",
        marker_color=[team_color(t) for t in strength_sorted[team_col_s]],
        text=[f"{v:.2f}" for v in strength_sorted[str_col]],
        textposition="outside",
        textfont=dict(color=TEXT_PRIMARY, family="JetBrains Mono", size=11),
    ))
    fig_str.update_layout(title="Blended strength score",
                          xaxis_title="Strength", yaxis_title=None)
    st.plotly_chart(plotly_cc(fig_str, 450), use_container_width=True)

    # ── Section 3: Playoff bracket ────────────────────────────────────
    section("Projected playoff bracket")
    strength_map = dict(zip(strength[team_col_s], strength[str_col]))
    top4 = (strength.sort_values(str_col, ascending=False).head(4)
            [team_col_s].tolist())

    def win_prob(a, b):
        sa, sb = strength_map[a], strength_map[b]
        return sa / (sa + sb), sb / (sa + sb)

    def render_matchup(stage, a, b, is_final=False):
        pa, pb = win_prob(a, b)
        winner = a if pa > pb else b
        tca, tcb = team_color(a), team_color(b)
        winner_color = team_color(winner)
        card_class = "bracket-card is-final" if is_final else "bracket-card"
        label_class = "bracket-stage-label is-final" if is_final else "bracket-stage-label"
        return (
            f'<div class="{card_class}">'
            f'<div style="display:flex; justify-content:space-between; '
            f'align-items:center; margin-bottom:10px;">'
            f'<span class="{label_class}">{stage}</span>'
            f'<span style="color:{TEXT_SECONDARY}; font-size:0.78rem;">'
            f'{"★ " if is_final else ""}<b style="color:{winner_color};">{winner}</b>'
            f'</span>'
            f'</div>'
            # team A
            f'<div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">'
            f'{logo_html(a, size=26)}'
            f'<div style="flex:1;">'
            f'<div style="display:flex; justify-content:space-between; '
            f'color:{TEXT_PRIMARY}; font-size:0.82rem; margin-bottom:3px;">'
            f'<span>{a}</span>'
            f'<span style="font-family:JetBrains Mono; color:{tca};">{pa*100:.1f}%</span>'
            f'</div>'
            f'<div style="background:rgba(255,255,255,0.05); border-radius:3px; '
            f'height:5px; overflow:hidden;">'
            f'<div style="background:{tca}; width:{pa*100}%; height:100%;"></div>'
            f'</div>'
            f'</div>'
            f'</div>'
            # team B
            f'<div style="display:flex; align-items:center; gap:10px;">'
            f'{logo_html(b, size=26)}'
            f'<div style="flex:1;">'
            f'<div style="display:flex; justify-content:space-between; '
            f'color:{TEXT_PRIMARY}; font-size:0.82rem; margin-bottom:3px;">'
            f'<span>{b}</span>'
            f'<span style="font-family:JetBrains Mono; color:{tcb};">{pb*100:.1f}%</span>'
            f'</div>'
            f'<div style="background:rgba(255,255,255,0.05); border-radius:3px; '
            f'height:5px; overflow:hidden;">'
            f'<div style="background:{tcb}; width:{pb*100}%; height:100%;"></div>'
            f'</div>'
            f'</div>'
            f'</div>'
            f'</div>'
        )

    t1, t2, t3, t4 = top4
    pa_q1, _ = win_prob(t1, t2)
    q1w, q1l = (t1, t2) if pa_q1 >= 0.5 else (t2, t1)
    pa_e, _ = win_prob(t3, t4)
    ew = t3 if pa_e >= 0.5 else t4
    pa_q2, _ = win_prob(q1l, ew)
    q2w = q1l if pa_q2 >= 0.5 else ew

    cA, cB = st.columns(2)
    with cA:
        st.markdown(render_matchup("QUALIFIER 1", t1, t2), unsafe_allow_html=True)
        st.markdown(render_matchup("QUALIFIER 2", q1l, ew), unsafe_allow_html=True)
    with cB:
        st.markdown(render_matchup("ELIMINATOR", t3, t4), unsafe_allow_html=True)
        st.markdown(render_matchup("GRAND FINAL", q1w, q2w, is_final=True),
                    unsafe_allow_html=True)

    # ── Section 4: Monte Carlo ────────────────────────────────────────
    section("Monte Carlo championship")
    cp_sorted = champ_prob.sort_values(prob_col, ascending=False).reset_index(drop=True)
    pre_champ = cp_sorted.iloc[0][team_col_c]

    st.markdown(
        f'<div class="info-card">'
        f'<b>Preloaded (10,000 simulations):</b> '
        f'★ <b style="color:{team_color(pre_champ)};">{pre_champ}</b> '
        f'wins with <code>{cp_sorted.iloc[0][prob_col]:.1f}%</code> championship probability.'
        f'</div>',
        unsafe_allow_html=True,
    )

    colors_pre = [team_color(t) if t == pre_champ else
                  f"rgba({int(team_color(t)[1:3],16)},"
                  f"{int(team_color(t)[3:5],16)},"
                  f"{int(team_color(t)[5:7],16)},0.45)"
                  for t in cp_sorted[team_col_c]]
    fig_pre = go.Figure(go.Bar(
        x=cp_sorted[team_col_c], y=cp_sorted[prob_col],
        marker_color=colors_pre,
        text=[f"{v:.1f}" for v in cp_sorted[prob_col]],
        textposition="outside",
        textfont=dict(color=TEXT_PRIMARY, family="JetBrains Mono", size=11),
    ))
    fig_pre.update_layout(title="Championship probability · preloaded run",
                          yaxis_title="Probability", xaxis_title=None)
    st.plotly_chart(plotly_cc(fig_pre, 420), use_container_width=True)

    st.markdown("##### Re-run simulation")
    n_sims = st.slider("Number of simulations", 1000, 20000, 10000, step=1000)
    if st.button("Run New Simulation"):
        rng = np.random.default_rng()
        with st.spinner(f"Running {n_sims:,} brackets..."):
            wins = {t: 0 for t in top4}
            for _ in range(n_sims):
                champ = simulate_playoff_once(top4, strength_map, rng)
                wins[champ] = wins.get(champ, 0) + 1
        sim_df = (pd.DataFrame({
            "team": list(wins.keys()),
            "championship_pct": [v / n_sims * 100 for v in wins.values()],
        }).sort_values("championship_pct", ascending=False).reset_index(drop=True))
        champ = sim_df.iloc[0]["team"]
        st.success(f"★ {champ} · {sim_df.iloc[0]['championship_pct']:.1f}%")

        colors_new = [team_color(t) if t == champ else
                      f"rgba({int(team_color(t)[1:3],16)},"
                      f"{int(team_color(t)[3:5],16)},"
                      f"{int(team_color(t)[5:7],16)},0.45)"
                      for t in sim_df["team"]]
        fig_new = go.Figure(go.Bar(
            x=sim_df["team"], y=sim_df["championship_pct"],
            marker_color=colors_new,
            text=[f"{v:.1f}%" for v in sim_df["championship_pct"]],
            textposition="outside",
            textfont=dict(color=TEXT_PRIMARY, family="JetBrains Mono", size=11),
        ))
        fig_new.update_layout(title=f"Fresh simulation · {n_sims:,} runs",
                              yaxis_title="Championship %", xaxis_title=None)
        st.plotly_chart(plotly_cc(fig_new, 420), use_container_width=True)


# ============================================================================
# PAGE 6 — METHODOLOGY
# ============================================================================

def page_methodology(matches):
    hero("Documentation", "Model & Methodology",
         "How the ML pipeline was designed, trained, and evaluated")

    # Pipeline as a horizontal flow of stages
    section("Pipeline architecture")
    stages = [
        ("01", "Ingestion", "5 CSVs · team_ids mapped through lookup dict"),
        ("02", "Cleaning", "Fix duplicate teams · season strings · bool flags"),
        ("03", "Feature Eng.", "15 features · shift(1) prevents leakage"),
        ("04", "Train / Test", "Time-based · train 2008–2023 · test 2024–2025"),
        ("05", "Models", "Logistic Regression · Random Forest · XGBoost"),
        ("06", "Deployment", "Random Forest serialized to rf_model.pkl"),
    ]
    stages_html = '<div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; margin-bottom:1rem;">'
    for num, name, desc in stages:
        stages_html += (
            f'<div class="surface-card" style="margin-bottom:0; padding:1rem;">'
            f'<div style="color:{CYAN}; font-family:JetBrains Mono; '
            f'font-size:0.8rem; font-weight:700; letter-spacing:0.1em;">STAGE {num}</div>'
            f'<div style="color:{TEXT_PRIMARY}; font-family:Space Grotesk; '
            f'font-weight:600; font-size:1.05rem; margin:6px 0 4px;">{name}</div>'
            f'<div style="color:{TEXT_SECONDARY}; font-size:0.8rem; '
            f'line-height:1.4;">{desc}</div>'
            f'</div>'
        )
    stages_html += "</div>"
    st.markdown(stages_html, unsafe_allow_html=True)

    section("The 15 features")
    feature_df = pd.DataFrame([
        ("team_enc", "Label-encoded team ID (0–14)", "Categorical"),
        ("opponent_enc", "Label-encoded opponent ID", "Categorical"),
        ("toss_won", "1 if team won the toss, else 0", "Binary"),
        ("form", "Win % over last 5 matches", "Rolling (5, shifted)"),
        ("h2h_win_pct", "Lifetime win % vs this opponent", "Expanding (shifted)"),
        ("venue_win_pct", "Lifetime win % at this venue", "Expanding (shifted)"),
        ("season", "Year (2008–2025)", "Numeric"),
        ("overall_win_rate", "Lifetime win % up to this match", "Expanding (shifted)"),
        ("season_win_rate", "Season-to-date win %", "Expanding (shifted)"),
        ("h2h_runs_powerplay", "Avg runs vs opponent in overs 1–6", "Aggregate"),
        ("h2h_runs_middle", "Avg runs vs opponent in overs 7–15", "Aggregate"),
        ("h2h_runs_death", "Avg runs vs opponent in overs 16–20", "Aggregate"),
        ("h2h_wkts_powerplay", "Avg wickets vs opponent in overs 1–6", "Aggregate"),
        ("h2h_wkts_middle", "Avg wickets vs opponent in overs 7–15", "Aggregate"),
        ("h2h_wkts_death", "Avg wickets vs opponent in overs 16–20", "Aggregate"),
    ], columns=["Feature", "Definition", "Type"])
    st.dataframe(feature_df, use_container_width=True, hide_index=True)

    section("The shift(1) trick · no-leakage protocol")
    st.markdown(
        '<div class="info-card">'
        'A naive rolling win rate would include the <i>current</i> match in '
        'the calculation, leaking the label into the features. Every rolling '
        'and expanding stat uses <code>.shift(1)</code> first — the value at '
        'row <i>i</i> is computed using only rows 1 to <i>i</i>−1. Same '
        'principle as walk-forward validation in time-series forecasting.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.code("""
group['form'] = group['won'].shift(1).rolling(5, min_periods=1).mean()
group['overall_win_rate'] = group['won'].shift(1).expanding().mean()
""", language="python")

    section("Model comparison")
    results = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Accuracy": [0.4276, 0.5103, 0.4966],
        "ROC-AUC": [0.4516, 0.5038, 0.4905],
    })
    st.dataframe(results, use_container_width=True, hide_index=True)

    fig_m = go.Figure()
    fig_m.add_trace(go.Bar(name="Accuracy", x=results["Model"], y=results["Accuracy"],
                           marker_color=CYAN,
                           text=[f"{v:.3f}" for v in results["Accuracy"]],
                           textposition="outside",
                           textfont=dict(color=TEXT_PRIMARY, family="JetBrains Mono")))
    fig_m.add_trace(go.Bar(name="ROC-AUC", x=results["Model"], y=results["ROC-AUC"],
                           marker_color=AMBER,
                           text=[f"{v:.3f}" for v in results["ROC-AUC"]],
                           textposition="outside",
                           textfont=dict(color=TEXT_PRIMARY, family="JetBrains Mono")))
    fig_m.add_hline(y=0.5, line_dash="dash", line_color=RED, line_width=1.5)
    fig_m.update_layout(title="Performance on 2024–2025 test set",
                        barmode="group", yaxis=dict(range=[0, 0.7]),
                        yaxis_title="Score")
    st.plotly_chart(plotly_cc(fig_m, 420), use_container_width=True)

    section("Feature importance (Random Forest)")
    importance = pd.DataFrame({
        "feature": ["toss_won", "form", "team_enc", "opponent_enc",
                    "venue_win_pct", "season", "season_win_rate",
                    "h2h_win_pct", "overall_win_rate"],
        "importance": [0.032, 0.068, 0.079, 0.102, 0.115, 0.117, 0.140, 0.146, 0.202],
    }).sort_values("importance", ascending=True)
    fig_imp = go.Figure(go.Bar(
        x=importance["importance"], y=importance["feature"],
        orientation="h", marker_color=CYAN,
        text=[f"{v:.3f}" for v in importance["importance"]],
        textposition="outside",
        textfont=dict(color=TEXT_PRIMARY, family="JetBrains Mono", size=11),
    ))
    fig_imp.update_layout(title="Top features by Gini importance",
                          xaxis_title="Importance", yaxis_title=None)
    st.plotly_chart(plotly_cc(fig_imp, 420), use_container_width=True)

    st.markdown(
        '<div class="info-card">'
        '<b>Reading this:</b> <code>overall_win_rate</code> dominates — '
        'historical team quality is the strongest single signal. '
        '<code>h2h_win_pct</code> and <code>season_win_rate</code> follow. '
        '<code>toss_won</code> is the weakest — winning the toss alone '
        'barely moves the needle.'
        '</div>',
        unsafe_allow_html=True,
    )

    section("Why ~51% accuracy is actually fine")
    st.markdown(
        '<div class="info-card">'
        'T20 cricket has high <b>irreducible variance</b> — a single dropped '
        'catch, a rain interruption, a fluke dismissal can flip the outcome. '
        'Bookmakers\' implied probabilities for IPL matches typically range '
        'from ~45% to 55% for both sides. The practical use of this model '
        'is not predicting individual matches confidently, but identifying '
        '<i>relative</i> team strength — which directly drives the playoff simulation.'
        '</div>',
        unsafe_allow_html=True,
    )

    section("Manual data corrections")
    st.markdown(
        f'<div class="info-card">'
        f'The raw dataset had incorrect winners for several IPL finals (the '
        f'<code>match_winner</code> field stored the last match of the season, '
        f'not necessarily the final). <b>{len(FINAL_FIXES)} finals</b> were '
        f'manually corrected against verified sources before computing trophy '
        f'counts and feeding the model.'
        f'</div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():
    with st.sidebar:
        # Sidebar logo + title
        ipl_path = ASSETS_PATH / "ipl.png"
        if ipl_path.exists():
            with open(ipl_path, "rb") as f:
                ipl_b64 = base64.b64encode(f.read()).decode()
            st.markdown(
                f'<div style="padding:0.6rem 0 0.3rem;">'
                f'<div style="background:{BG_SURFACE}; border:1px solid {BORDER_GLOW}; '
                f'border-radius:12px; padding:16px 18px; width:100%; '
                f'box-shadow:0 0 28px rgba(0,217,255,0.08); text-align:center;">'
                f'<img src="data:image/png;base64,{ipl_b64}" '
                f'style="width:min(100%, 220px); height:auto; max-height:120px; '
                f'object-fit:contain; display:block; margin:0 auto;"/>'
                f'</div>'
                f'<div style="color:{TEXT_PRIMARY}; font-family:Space Grotesk; '
                f'font-weight:700; font-size:1.15rem; letter-spacing:0.02em; '
                f'line-height:1.1; margin-top:0.9rem;">IPL ANALYTICS</div>'
                f'<div style="color:{TEXT_MUTED}; font-size:0.7rem; '
                f'font-family:JetBrains Mono; margin-top:4px;">Series 2026</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div style="height:1px; background:linear-gradient(90deg,{BORDER_GLOW},transparent); '
            f'margin:1rem 0;"></div>',
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigate",
            ["🏠  Home", "📊  EDA Dashboard", "👥  Team Analysis",
             "🎯  Match Predictor", "🚀  IPL 2026 Simulator",
             "🧠  Model & Methodology"],
            label_visibility="collapsed",
        )

        # Footer card
        st.markdown("###  ")
        st.markdown(
            f'<div style="margin-top:2rem; padding:0.7rem 1rem; '
            f'background:linear-gradient(135deg, {CYAN} 0%, {BLUE} 100%); '
            f'border-radius:8px; text-align:center; color:#000000; '
            f'font-family:Inter; font-weight:700; font-size:0.85rem; '
            f'letter-spacing:0.02em;">'
            f'● &nbsp;Live Analytics'
            f'</div>'
            f'<div style="text-align:center; color:{TEXT_MUTED}; font-size:0.7rem; '
            f'margin-top:0.7rem; font-family:JetBrains Mono;">'
            f'Powered by ML Intelligence'
            f'</div>'
            f'<div style="text-align:center; color:{TEXT_MUTED}; font-size:0.65rem; '
            f'margin-top:0.4rem; opacity:0.7;">'
            f'Built by Harsh · Goa University'
            f'</div>',
            unsafe_allow_html=True,
        )

    try:
        matches = load_matches()
        deliveries = load_deliveries()
    except FileNotFoundError as e:
        st.error(f"❌ Data file not found: {e}\n\nCheck BASE_PATH: `{BASE_PATH}`")
        st.stop()

    def safe_call(fn, *args):
        try:
            fn(*args)
        except Exception as e:
            st.error(f"Page failed: {type(e).__name__}: {e}")
            import traceback
            st.code(traceback.format_exc())

    if "Home" in page:
        safe_call(page_home, matches, deliveries)
    elif "EDA" in page:
        safe_call(page_eda, matches, deliveries)
    elif "Team Analysis" in page:
        safe_call(page_team_analysis, matches, deliveries)
    elif "Predictor" in page:
        safe_call(page_predictor, matches, deliveries)
    elif "Simulator" in page:
        try:
            points = load_points_table()
            strength = load_strength_table()
            champ_prob = load_championship_prob()
            safe_call(page_simulator, points, strength, champ_prob)
        except FileNotFoundError as e:
            st.error(f"❌ Simulator file not found: {e}")
    elif "Methodology" in page:
        safe_call(page_methodology, matches)


if __name__ == "__main__":
    main()
