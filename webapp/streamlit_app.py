"""
webapp/streamlit_app.py
========================
Morocco Healthcare Access Optimization Dashboard
Phase 4 — Production-ready Streamlit application.

Moroccan Ministry of Health–inspired design system.
Deployable to Streamlit Community Cloud and Hugging Face Spaces.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.access_metrics import (
    coverage_within_radius,
    population_weighted_distance,
    population_per_facility_ratio,
)
from src.scenario_simulator import run_scenario

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
FACILITIES_PATH = ROOT / "data/processed/facilities.geojson"
POPGRID_PATH    = ROOT / "data/processed/popgrid.geojson"
METRICS_PATH    = ROOT / "data/processed/access_metrics.csv"
SCENARIO_PATH   = ROOT / "data/processed/scenario_results.json"

# ── Design tokens — Ministry of Health palette ────────────────────────────────
COLOR = {
    "primary":    "#1F4E79",
    "teal":       "#2F8F9D",
    "light_bg":   "#EAF2F8",
    "dark_text":  "#1B2631",
    "muted":      "#5D6D7E",
    "success":    "#1E8449",
    "warning":    "#CA6F1E",
    "danger":     "#922B21",
    "white":      "#FFFFFF",
    "border":     "#D5E8F0",
}

FACILITY_COLORS = {
    "hospital": "#922B21",
    "clinic":   "#1F4E79",
    "doctor":   "#2F8F9D",
    "pharmacy": "#1E8449",
    "other":    "#5D6D7E",
}

FACILITY_SYMBOLS = {
    # Only use symbols confirmed valid in go.Scattermap across Plotly versions.
    # "cross" and "diamond" render hover hitboxes but NO visible glyph in many
    # Plotly/mapbox versions — replaced with star, square-stroked, triangle, circle-stroked.
    "hospital": "star",
    "clinic":   "square",
    "doctor":   "circle",
    "pharmacy": "triangle",
    "other":    "x",
}

MOROCCO_CENTER = {"lat": 31.7917, "lon": -7.0926}
MOROCCO_ZOOM   = 4.8


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def configure_page() -> None:
    """Set Streamlit page config — must be first Streamlit call."""
    st.set_page_config(
        page_title="Morocco Healthcare Access · Dashboard",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/marouane-ouaomar/morocco-healthcare-optimization",
            "Report a bug": "https://github.com/marouane-ouaomar/morocco-healthcare-optimization/issues",
            "About": "Morocco Healthcare Access Optimization — Portfolio Demo",
        },
    )


def inject_css() -> None:
    """Inject global CSS for Ministry of Health design system."""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@300;400;600;700&family=Source+Serif+4:wght@600;700&display=swap');

    /* ── Base ── */
    html, body, [class*="css"] {{
        font-family: 'Source Sans 3', sans-serif;
        color: {COLOR['dark_text']};
    }}

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer {{ visibility: hidden; }}
    .block-container {{ padding-top: 1rem; padding-bottom: 2rem; }}

    /* ── Ensure sidebar is always interactive ── */
    [data-testid="stSidebar"] * {{ pointer-events: auto !important; }}
    [data-testid="stSidebar"] {{ pointer-events: auto !important; }}

    /* ── Ministry Banner ── */
    .moh-banner {{
        background: linear-gradient(135deg, {COLOR['primary']} 0%, #163d61 100%);
        padding: 20px 32px;
        border-radius: 8px;
        margin-bottom: 6px;
        border-left: 6px solid {COLOR['teal']};
        box-shadow: 0 4px 16px rgba(31,78,121,0.18);
    }}
    .moh-banner h1 {{
        font-family: 'Source Serif 4', serif;
        color: white;
        font-size: 1.55rem;
        font-weight: 700;
        margin: 0 0 4px 0;
        letter-spacing: 0.5px;
        line-height: 1.2;
    }}
    .moh-banner p {{
        color: rgba(255,255,255,0.78);
        font-size: 0.85rem;
        margin: 0;
        font-weight: 300;
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }}
    .moh-flag {{
        display: flex;
        gap: 3px;
        margin-bottom: 10px;
    }}
    .moh-flag span {{
        display: inline-block;
        width: 28px; height: 5px;
        border-radius: 2px;
    }}

    /* ── Metric cards ── */
    .metric-card {{
        background: {COLOR['white']};
        border: 1px solid {COLOR['border']};
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 10px;
        border-left: 4px solid {COLOR['teal']};
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }}
    .metric-card.good  {{ border-left-color: {COLOR['success']}; }}
    .metric-card.warn  {{ border-left-color: {COLOR['warning']}; }}
    .metric-card.bad   {{ border-left-color: {COLOR['danger']}; }}

    .metric-label {{
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: {COLOR['muted']};
        margin-bottom: 4px;
    }}
    .metric-value {{
        font-family: 'Source Serif 4', serif;
        font-size: 2rem;
        font-weight: 700;
        color: {COLOR['dark_text']};
        line-height: 1;
    }}
    .metric-sub {{
        font-size: 0.78rem;
        color: {COLOR['muted']};
        margin-top: 4px;
    }}

    /* ── Before/After scenario cards ── */
    .scenario-card {{
        background: {COLOR['white']};
        border: 1px solid {COLOR['border']};
        border-radius: 8px;
        padding: 20px;
        height: 100%;
    }}
    .scenario-card.before {{ border-top: 4px solid {COLOR['danger']}; }}
    .scenario-card.after  {{ border-top: 4px solid {COLOR['success']}; }}
    .scenario-title {{
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 16px;
    }}
    .scenario-title.before {{ color: {COLOR['danger']}; }}
    .scenario-title.after  {{ color: {COLOR['success']}; }}

    .delta-positive {{ color: {COLOR['success']}; font-weight: 700; font-size: 0.85rem; }}
    .delta-negative {{ color: {COLOR['danger']};  font-weight: 700; font-size: 0.85rem; }}

    /* ── Section headers ── */
    .section-header {{
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: {COLOR['primary']};
        border-bottom: 2px solid {COLOR['teal']};
        padding-bottom: 6px;
        margin: 20px 0 14px 0;
    }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background: {COLOR['light_bg']};
        border-right: 1px solid {COLOR['border']};
    }}
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label {{
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: {COLOR['primary']};
    }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        border-bottom: 2px solid {COLOR['border']};
    }}
    .stTabs [data-baseweb="tab"] {{
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        padding: 8px 20px;
        border-radius: 4px 4px 0 0;
        color: {COLOR['muted']};
    }}
    .stTabs [aria-selected="true"] {{
        background: {COLOR['primary']} !important;
        color: white !important;
    }}

    /* ── Buttons ── */
    .stButton > button {{
        background: {COLOR['primary']};
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.78rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        padding: 10px 24px;
        width: 100%;
        transition: background 0.2s;
    }}
    .stButton > button:hover {{
        background: {COLOR['teal']};
        color: white;
    }}

    /* ── Download buttons ── */
    .stDownloadButton > button {{
        background: {COLOR['teal']};
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.78rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        width: 100%;
    }}

    /* ── Info boxes ── */
    .info-box {{
        background: {COLOR['light_bg']};
        border: 1px solid {COLOR['border']};
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 0.82rem;
        color: {COLOR['muted']};
        margin: 8px 0;
    }}

    /* ── Legend ── */
    .legend-item {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.78rem;
        color: {COLOR['dark_text']};
        margin-bottom: 4px;
    }}
    .legend-dot {{
        width: 12px; height: 12px;
        border-radius: 50%;
        flex-shrink: 0;
    }}
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_facilities() -> gpd.GeoDataFrame:
    """Load processed facility GeoDataFrame from disk."""
    if not FACILITIES_PATH.exists():
        st.error(f"Facilities file not found: {FACILITIES_PATH}\nRun the data pipeline first.")
        st.stop()
    gdf = gpd.read_file(FACILITIES_PATH)
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y
    return gdf


@st.cache_data(show_spinner=False)
def load_popgrid() -> gpd.GeoDataFrame:
    """Load processed population grid GeoDataFrame from disk."""
    if not POPGRID_PATH.exists():
        st.error(f"Population grid not found: {POPGRID_PATH}\nRun the data pipeline first.")
        st.stop()
    return gpd.read_file(POPGRID_PATH)


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    """Load access metrics CSV from disk."""
    if not METRICS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(METRICS_PATH)


@st.cache_data(show_spinner=False)
def load_scenario_results() -> dict:
    """Load saved scenario results JSON from disk."""
    if not SCENARIO_PATH.exists():
        return {}
    with open(SCENARIO_PATH, "r") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def compute_baseline_metrics(
    _pop_gdf: gpd.GeoDataFrame,
    _facilities_gdf: gpd.GeoDataFrame,
) -> dict:
    """
    Compute and cache baseline access metrics.
    Underscore prefix prevents Streamlit from hashing GeoDataFrames.
    """
    coverage = coverage_within_radius(_pop_gdf, _facilities_gdf)
    pwd = population_weighted_distance(_pop_gdf, _facilities_gdf)
    ratio_df = population_per_facility_ratio(_pop_gdf, _facilities_gdf)
    return {
        "coverage": coverage,
        "pop_weighted_distance_km": pwd,
        "ratio_df": ratio_df,
        "total_population": float(_pop_gdf["population"].sum()),
        "n_facilities": len(_facilities_gdf),
    }


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

def render_header() -> None:
    """Render the Ministry of Health–style banner."""
    st.markdown("""
    <div class="moh-banner">
        <div class="moh-flag">
            <span style="background:#c1272d;width:20px"></span>
            <span style="background:#006233;width:20px"></span>
        </div>
        <h1>🏥 MOROCCO HEALTHCARE ACCESS OPTIMIZATION</h1>
        <p>Decision-support system for facility planning &amp; telemedicine simulation — Ministry of Health</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar(facilities_gdf: gpd.GeoDataFrame) -> dict:
    """
    Render sidebar controls and return filter/scenario configuration.
    """
    with st.sidebar:
        st.markdown(f"""
        <div style='padding:12px 0 8px 0;'>
            <div style='font-size:0.65rem;font-weight:700;letter-spacing:2px;
                        text-transform:uppercase;color:{COLOR["primary"]};
                        margin-bottom:2px;'>Dashboard Controls</div>
            <div style='height:2px;background:{COLOR["teal"]};border-radius:1px;'></div>
        </div>
        """, unsafe_allow_html=True)

        # ── 1. Region filter ──────────────────────────────────────────────
        st.markdown("**🗺 Region**")
        regions = ["All Regions"]
        if "region" in facilities_gdf.columns:
            r = sorted(facilities_gdf["region"].dropna().unique().tolist())
            regions += [x for x in r if x != "Unknown"]
        selected_region = st.selectbox(
            "Select region", regions, label_visibility="collapsed"
        )

        # ── 2. Facility type ──────────────────────────────────────────────
        st.markdown("**🏥 Facility Type**")
        all_types = sorted(facilities_gdf["facility_type"].dropna().unique().tolist())
        selected_types = st.multiselect(
            "Select types", all_types, default=all_types,
            label_visibility="collapsed"
        )
        if not selected_types:
            selected_types = all_types

        # ── 3. Distance radius ────────────────────────────────────────────
        st.markdown("**📏 Coverage Radius**")
        radius_km = st.selectbox(
            "Coverage radius", [5, 10, 20],
            format_func=lambda x: f"{x} km",
            label_visibility="collapsed"
        )

        st.markdown("---")

        # ── 4. Scenario simulator ─────────────────────────────────────────
        st.markdown(f"""
        <div style='font-size:0.65rem;font-weight:700;letter-spacing:2px;
                    text-transform:uppercase;color:{COLOR["primary"]};
                    margin-bottom:8px;'>⚙️ Scenario Simulator</div>
        """, unsafe_allow_html=True)

        new_facilities = st.number_input(
            "🏗 New clinics / facilities", min_value=0, max_value=500, value=3, step=1
        )
        mobile_units = st.number_input(
            "🚐 Mobile health units", min_value=0, max_value=200, value=1, step=1
        )
        kiosks = st.number_input(
            "💻 Telemedicine kiosks", min_value=0, max_value=500, value=2, step=1
        )

        run_btn = st.button("▶ RUN SCENARIO", use_container_width=True)

        st.markdown("---")
        st.markdown(f"""
        <div class='info-box'>
            <b>Data source:</b> OpenStreetMap via Overpass API<br>
            <b>Population:</b> WorldPop Morocco synthetic grid<br>
            <b>CRS:</b> EPSG:4326 (WGS84)
        </div>
        """, unsafe_allow_html=True)

    return {
        "region":         selected_region,
        "facility_types": selected_types,
        "radius_km":      radius_km,
        "new_facilities": new_facilities,
        "mobile_units":   mobile_units,
        "kiosks":         kiosks,
        "run_scenario":   run_btn,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FILTERS
# ══════════════════════════════════════════════════════════════════════════════

def apply_filters(
    facilities_gdf: gpd.GeoDataFrame,
    region: str,
    facility_types: list[str],
) -> gpd.GeoDataFrame:
    """
    Filter facilities by region and type only.

    Spatial validation (polygon containment) is performed once at ingestion
    time by data_prep.validate_coordinates(). Re-running it here on every
    render silently drops valid southern facilities (Laayoune, Dakhla,
    Tan-Tan) whose centroids land near polygon edges due to floating-point
    precision. Trust the already-clean processed data.
    """
    gdf = facilities_gdf.copy()
    if region != "All Regions" and "region" in gdf.columns:
        gdf = gdf[gdf["region"] == region]
    if facility_types:
        gdf = gdf[gdf["facility_type"].isin(facility_types)]
    return gdf


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════

# Unicode glyphs that visually match each Plotly Scattermap symbol.
# Displayed in the custom HTML legend so the user sees the exact shape + colour.
FACILITY_LEGEND_GLYPHS = {
    "hospital": "★",   # star
    "clinic":   "■",   # square
    "doctor":   "●",   # circle
    "pharmacy": "▲",   # triangle
    "other":    "✕",   # x
}

# Canonical display order for the legend
FACILITY_TYPE_ORDER = ["hospital", "clinic", "doctor", "pharmacy", "other"]


def build_facility_map(
    facilities_gdf: gpd.GeoDataFrame,
    pop_gdf: gpd.GeoDataFrame,
    radius_km: float = 10.0,
    show_radius: bool = True,
) -> go.Figure:
    """Build the main facility map with population density heatmap and markers."""
    fig = go.Figure()

    # ── Population density heatmap ────────────────────────────────────────
    if len(pop_gdf) > 0:
        pop_sample = pop_gdf.copy()
        pop_sample["lon"] = pop_sample.geometry.x
        pop_sample["lat"] = pop_sample.geometry.y
        pop_sample = pop_sample.nlargest(2000, "population")

        fig.add_trace(go.Densitymap(
            lat=pop_sample["lat"],
            lon=pop_sample["lon"],
            z=pop_sample["population"],
            radius=14,
            colorscale=[
                [0.0, "rgba(234,242,248,0)"],
                [0.3, "rgba(47,143,157,0.3)"],
                [0.7, "rgba(31,78,121,0.5)"],
                [1.0, "rgba(27,38,49,0.7)"],
            ],
            showscale=False,
            name="Population density",
            hoverinfo="skip",
            showlegend=False,
        ))

    # ── Facility markers by type (in fixed display order) ─────────────────
    present_types = facilities_gdf["facility_type"].unique().tolist()
    ordered_types = [t for t in FACILITY_TYPE_ORDER if t in present_types]
    ordered_types += [t for t in present_types if t not in FACILITY_TYPE_ORDER]

    for ftype in ordered_types:
        subset = facilities_gdf[facilities_gdf["facility_type"] == ftype].copy()
        subset["lon"] = subset.geometry.x
        subset["lat"] = subset.geometry.y
        subset = subset[subset["lat"].notna() & subset["lon"].notna()]
        subset = subset[
            subset["lat"].between(-90, 90) & subset["lon"].between(-180, 180)
        ]
        if len(subset) == 0:
            continue

        color  = FACILITY_COLORS.get(ftype, FACILITY_COLORS["other"])
        symbol = FACILITY_SYMBOLS.get(ftype, "circle")
        glyph  = FACILITY_LEGEND_GLYPHS.get(ftype, "●")
        count  = len(subset)

        names  = subset["name_clean"].fillna("Unnamed").tolist() if "name_clean" in subset.columns else [""] * len(subset)
        region = subset["region"].fillna("Unknown").tolist()     if "region"     in subset.columns else ["Unknown"] * len(subset)
        cd     = [[ftype, rg] for rg in region]

        fig.add_trace(go.Scattermap(
            lat=subset["lat"].tolist(),
            lon=subset["lon"].tolist(),
            mode="markers",
            marker=dict(size=9, color=color, symbol=symbol, opacity=0.9, allowoverlap=True),
            # Hide Plotly's built-in legend — we draw our own HTML legend below
            showlegend=False,
            name=ftype.capitalize(),
            text=names,
            customdata=cd,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Type: %{customdata[0]}<br>"
                "Region: %{customdata[1]}<br>"
                "Lat: %{lat:.4f}, Lon: %{lon:.4f}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        map=dict(style="carto-positron", center={"lat": 28.5, "lon": -9.0}, zoom=4.5),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=520,
        showlegend=False,
        paper_bgcolor=COLOR["white"],
    )
    return fig


def build_facility_legend_html(facilities_gdf: gpd.GeoDataFrame) -> str:
    """
    Return a full HTML document for the facility-type legend.
    Rendered via st.components.v1.html() (iframe) so the markup is never
    sanitised or escaped by Streamlit's markdown processor.
    """
    present_types = facilities_gdf["facility_type"].dropna().unique().tolist()
    ordered = [t for t in FACILITY_TYPE_ORDER if t in present_types]
    ordered += [t for t in present_types if t not in FACILITY_TYPE_ORDER]

    rows = ""
    for ftype in ordered:
        color = FACILITY_COLORS.get(ftype, FACILITY_COLORS["other"])
        glyph = FACILITY_LEGEND_GLYPHS.get(ftype, "●")
        count = int((facilities_gdf["facility_type"] == ftype).sum())
        rows += f"""
        <div class="row">
            <span class="glyph" style="color:{color};">{glyph}</span>
            <span class="label">{ftype.capitalize()}</span>
            <span class="count">{count:,}</span>
        </div>"""

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: 'Segoe UI', sans-serif;
    background: transparent;
    padding: 0;
  }}
  .card {{
    display: inline-block;
    background: rgba(255,255,255,0.97);
    border: 1px solid #D5E8F0;
    border-radius: 7px;
    padding: 10px 14px;
    min-width: 190px;
    box-shadow: 0 1px 5px rgba(0,0,0,0.09);
  }}
  .title {{
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    color: #1F4E79;
    border-bottom: 1px solid #D5E8F0;
    padding-bottom: 6px;
    margin-bottom: 8px;
  }}
  .row {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 5px;
  }}
  .glyph {{
    font-size: 14px;
    width: 18px;
    text-align: center;
    flex-shrink: 0;
    line-height: 1;
  }}
  .label {{
    font-size: 12px;
    color: #1B2631;
    flex: 1;
  }}
  .count {{
    font-size: 11px;
    color: #5D6D7E;
    font-weight: 600;
  }}
</style>
</head>
<body>
  <div class="card">
    <div class="title">Facility Types</div>
    {rows}
  </div>
</body>
</html>"""


SCENARIO_STYLES = {
    "new_facilities":      dict(color="#922B21", symbol="star",     size=14, label="New facility",        glyph="★"),
    "mobile_units":        dict(color="#CA6F1E", symbol="triangle", size=12, label="Mobile unit",         glyph="▲"),
    "telemedicine_kiosks": dict(color="#2F8F9D", symbol="circle",   size=10, label="Telemedicine kiosk",  glyph="●"),
}


def build_scenario_map(
    existing_gdf: gpd.GeoDataFrame,
    scenario_results: dict,
) -> go.Figure:
    """Build scenario map showing existing + proposed facility locations."""
    fig = go.Figure()

    # ── Existing facilities — sampled for performance ─────────────────────
    if "lon" not in existing_gdf.columns:
        existing_gdf = existing_gdf.copy()
        existing_gdf["lon"] = existing_gdf.geometry.x
        existing_gdf["lat"] = existing_gdf.geometry.y

    display_existing = existing_gdf
    if len(existing_gdf) > 3000:
        display_existing = existing_gdf.sample(3000, random_state=42)

    ftype_col = (
        display_existing["facility_type"]
        if "facility_type" in display_existing.columns
        else pd.Series(["facility"] * len(display_existing))
    )

    fig.add_trace(go.Scattermap(
        lat=display_existing["lat"],
        lon=display_existing["lon"],
        mode="markers",
        marker=dict(size=6, color="#888888", opacity=0.65),
        showlegend=False,
        name=f"Existing facilities ({len(existing_gdf):,})",
        customdata=ftype_col,
        hovertemplate=(
            "<b>Existing facility</b><br>"
            "Type: %{customdata}<br>"
            "Lat: %{lat:.4f}, Lon: %{lon:.4f}"
            "<extra></extra>"
        ),
    ))

    # ── Proposed sites by intervention type ───────────────────────────────
    interventions = scenario_results.get("interventions", {})
    for key, style in SCENARIO_STYLES.items():
        sites = interventions.get(key, {}).get("sites", [])
        if not sites:
            continue
        lats = [s["lat"] for s in sites]
        lons = [s["lon"] for s in sites]
        pops = [s.get("cluster_population", s.get("population_served", 0)) for s in sites]

        fig.add_trace(go.Scattermap(
            lat=lats,
            lon=lons,
            mode="markers",
            marker=dict(size=style["size"], color=style["color"],
                        symbol=style["symbol"], opacity=0.95),
            showlegend=False,
            name=style["label"],
            customdata=[[f"{p:,.0f}"] for p in pops],
            hovertemplate=(
                f"<b>{style['label']}</b><br>"
                "Population served: %{customdata[0]}<br>"
                "Lat: %{lat:.4f}, Lon: %{lon:.4f}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        map=dict(
            style="carto-positron",
            center={"lat": 29.5, "lon": -9.0},
            zoom=4.2,
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=520,
        showlegend=False,
        paper_bgcolor=COLOR["white"],
    )
    return fig


def build_scenario_legend_html(
    existing_gdf: gpd.GeoDataFrame,
    scenario_results: dict,
) -> str:
    """
    Full HTML document for the scenario map legend.
    Shows existing facilities + only the intervention types actually present
    in the current scenario results.
    Rendered via st.components.v1.html() to avoid Streamlit markdown escaping.
    """
    interventions = scenario_results.get("interventions", {})

    rows = f"""
    <div class="row">
        <span class="glyph" style="color:#888888;">●</span>
        <span class="label">Existing facilities</span>
        <span class="count">{len(existing_gdf):,}</span>
    </div>"""

    for key, style in SCENARIO_STYLES.items():
        sites = interventions.get(key, {}).get("sites", [])
        if not sites:
            continue
        rows += f"""
    <div class="row">
        <span class="glyph" style="color:{style['color']};">{style['glyph']}</span>
        <span class="label">{style['label']}</span>
        <span class="count">{len(sites)}</span>
    </div>"""

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: transparent; }}
  .card {{
    display: inline-block;
    background: rgba(255,255,255,0.97);
    border: 1px solid #D5E8F0;
    border-radius: 7px;
    padding: 10px 14px;
    min-width: 210px;
    box-shadow: 0 1px 5px rgba(0,0,0,0.09);
  }}
  .title {{
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    color: #1F4E79;
    border-bottom: 1px solid #D5E8F0;
    padding-bottom: 6px;
    margin-bottom: 8px;
  }}
  .row {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 5px;
  }}
  .glyph {{
    font-size: 14px;
    width: 18px;
    text-align: center;
    flex-shrink: 0;
    line-height: 1;
  }}
  .label {{ font-size: 12px; color: #1B2631; flex: 1; }}
  .count {{ font-size: 11px; color: #5D6D7E; font-weight: 600; }}
</style>
</head>
<body>
  <div class="card">
    <div class="title">Proposed Facility Locations</div>
    {rows}
  </div>
</body>
</html>"""


def build_region_bar_chart(ratio_df: pd.DataFrame) -> go.Figure:
    """Bar chart of population per facility by region."""
    df = ratio_df[
        (ratio_df["region"] != "National") & (ratio_df["facility_type"] == "all")
    ].copy()
    if df.empty:
        return go.Figure()

    df = df.sort_values("pop_per_facility", ascending=True).head(15)
    df["color"] = df["underserved"].map({True: COLOR["danger"], False: COLOR["success"]})

    fig = go.Figure(go.Bar(
        x=df["pop_per_facility"],
        y=df["region"],
        orientation="h",
        marker_color=df["color"],
        text=df["pop_per_facility"].apply(lambda x: f"{x:,.0f}"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Pop/facility: %{x:,.0f}<br><extra></extra>",
    ))

    fig.add_vline(
        x=10000, line_dash="dash", line_color=COLOR["warning"],
        annotation_text="WHO threshold (10,000)",
        annotation_font_color=COLOR["warning"],
        annotation_font_size=11,
    )

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=60, t=10, b=10),
        paper_bgcolor=COLOR["white"],
        plot_bgcolor=COLOR["light_bg"],
        xaxis=dict(title="Population per facility", gridcolor=COLOR["border"], title_font=dict(size=11)),
        yaxis=dict(title="", tickfont=dict(size=10)),
        showlegend=False,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# METRIC CARDS (HTML helpers)
# ══════════════════════════════════════════════════════════════════════════════

def metric_card(label: str, value: str, sub: str = "", status: str = "default") -> str:
    cls = {"good": "good", "warn": "warn", "bad": "bad"}.get(status, "")
    return f"""
    <div class="metric-card {cls}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-sub">{sub}</div>' if sub else ''}
    </div>
    """


def coverage_status(pct: float, radius: int) -> str:
    if radius <= 5:
        return "good" if pct >= 60 else ("warn" if pct >= 40 else "bad")
    elif radius <= 10:
        return "good" if pct >= 75 else ("warn" if pct >= 55 else "bad")
    else:
        return "good" if pct >= 85 else ("warn" if pct >= 65 else "bad")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ACCESS OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def render_overview_tab(
    filtered_facilities: gpd.GeoDataFrame,
    pop_gdf: gpd.GeoDataFrame,
    baseline: dict,
    radius_km: int,
) -> None:
    col_map, col_metrics = st.columns([3, 1.1], gap="medium")

    with col_map:
        st.markdown('<div class="section-header">🗺 Facility Map & Population Density</div>',
                    unsafe_allow_html=True)
        with st.spinner("Rendering map..."):
            fig = build_facility_map(filtered_facilities, pop_gdf, radius_km)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            components.html(build_facility_legend_html(filtered_facilities), height=160, scrolling=False)

        st.markdown('<div class="section-header">📊 Population per Facility by Region</div>',
                    unsafe_allow_html=True)
        ratio_df = baseline.get("ratio_df", pd.DataFrame())
        if not ratio_df.empty:
            fig2 = build_region_bar_chart(ratio_df)
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    with col_metrics:
        st.markdown('<div class="section-header">📈 Key Metrics</div>', unsafe_allow_html=True)

        total_pop    = baseline.get("total_population", 0)
        n_facilities = len(filtered_facilities)
        pwd          = baseline.get("pop_weighted_distance_km", 0)
        coverage     = baseline.get("coverage", {})

        cov5    = coverage.get("coverage_5km",  0)
        cov10   = coverage.get("coverage_10km", 0)
        cov20   = coverage.get("coverage_20km", 0)
        pct_far = max(0, 100 - cov20)

        st.markdown(metric_card("Total Population", f"{total_pop/1_000_000:.1f}M",
                                sub="Morocco estimate (WorldPop)"), unsafe_allow_html=True)
        st.markdown(metric_card("Facilities Mapped", f"{n_facilities:,}",
                                sub="OSM-sourced, Morocco"), unsafe_allow_html=True)
        st.markdown(metric_card("Avg Distance to Facility", f"{pwd:.1f} km",
                                sub="Population-weighted mean",
                                status="good" if pwd < 5 else ("warn" if pwd < 12 else "bad")),
                    unsafe_allow_html=True)
        st.markdown(metric_card(f"Within {radius_km} km",
                                f"{coverage.get(f'coverage_{radius_km}km', 0):.1f}%",
                                sub="of population near a facility",
                                status=coverage_status(coverage.get(f"coverage_{radius_km}km", 0), radius_km)),
                    unsafe_allow_html=True)
        st.markdown(metric_card("Within 5 km",  f"{cov5:.1f}%",  status=coverage_status(cov5, 5)),
                    unsafe_allow_html=True)
        st.markdown(metric_card("Within 10 km", f"{cov10:.1f}%", status=coverage_status(cov10, 10)),
                    unsafe_allow_html=True)
        st.markdown(metric_card("Beyond 20 km", f"{pct_far:.1f}%",
                                sub="Underserved population",
                                status="bad" if pct_far > 20 else ("warn" if pct_far > 10 else "good")),
                    unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:20px;">🏥 By Facility Type</div>',
                    unsafe_allow_html=True)
        if "facility_type" in filtered_facilities.columns:
            type_counts = filtered_facilities["facility_type"].value_counts()
            for ftype, count in type_counts.items():
                color = FACILITY_COLORS.get(ftype, FACILITY_COLORS["other"])
                st.markdown(f"""
                <div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>
                    <div style='width:10px;height:10px;border-radius:50%;
                                background:{color};flex-shrink:0;'></div>
                    <div style='flex:1;font-size:0.8rem;'>{ftype.capitalize()}</div>
                    <div style='font-size:0.8rem;font-weight:600;
                                color:{COLOR["dark_text"]};'>{count:,}</div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SCENARIO SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def render_scenario_tab(
    facilities_gdf: gpd.GeoDataFrame,
    pop_gdf: gpd.GeoDataFrame,
    baseline: dict,
    scenario_config: dict,
) -> None:
    # ── Run scenario if button pressed ───────────────────────────────────
    if scenario_config["run_scenario"]:
        total = (scenario_config["new_facilities"]
                 + scenario_config["mobile_units"]
                 + scenario_config["kiosks"])
        if total == 0:
            st.warning("Set at least one intervention in the sidebar before running.")
        else:
            with st.spinner("🔄 Running optimization scenario..."):
                try:
                    results = run_scenario(
                        pop_gdf=pop_gdf,
                        facilities_gdf=facilities_gdf,
                        new_facilities=scenario_config["new_facilities"],
                        mobile_units=scenario_config["mobile_units"],
                        telemedicine_kiosks=scenario_config["kiosks"],
                    )
                    st.session_state["scenario_results"] = results
                    st.session_state["scenario_config_used"] = {
                        "new_facilities": scenario_config["new_facilities"],
                        "mobile_units":   scenario_config["mobile_units"],
                        "kiosks":         scenario_config["kiosks"],
                    }
                except Exception as e:
                    st.error(f"Scenario failed: {e}")
                    logger.exception("Scenario error")

    # ── Display results ───────────────────────────────────────────────────
    results = st.session_state.get("scenario_results", load_scenario_results())

    if not results:
        st.markdown(f"""
        <div class='info-box' style='text-align:center;padding:40px;'>
            <div style='font-size:2rem;margin-bottom:12px;'>⚙️</div>
            <div style='font-weight:600;font-size:1rem;color:{COLOR["primary"]};
                        margin-bottom:8px;'>No scenario run yet</div>
            <div style='color:{COLOR["muted"]};font-size:0.85rem;'>
                Set the values in the sidebar and click <b>▶ RUN SCENARIO</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    before   = results.get("baseline", {})
    after    = results.get("after", {})
    delta    = results.get("delta", {})
    combined = results.get("combined_results", {})
    costs    = results.get("cost_analysis", {})

    col_metrics, col_map = st.columns([1.1, 3], gap="medium")

    with col_metrics:
        st.markdown('<div class="section-header">📊 Before / After</div>', unsafe_allow_html=True)

        st.markdown(f"<div class='scenario-card before'><div class='scenario-title before'>◼ BEFORE</div>",
                    unsafe_allow_html=True)
        st.markdown(metric_card("Avg Distance",  f"{before.get('avg_distance_km', 0):.2f} km"),
                    unsafe_allow_html=True)
        st.markdown(metric_card("Coverage 5km",  f"{before.get('coverage_5km', 0):.1f}%"),
                    unsafe_allow_html=True)
        st.markdown(metric_card("Coverage 10km", f"{before.get('coverage_10km', 0):.1f}%"),
                    unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        st.markdown(f"<div class='scenario-card after'><div class='scenario-title after'>▲ AFTER</div>",
                    unsafe_allow_html=True)
        st.markdown(metric_card("Avg Distance", f"{after.get('avg_distance_km', 0):.2f} km",
                                sub=f"▼ {abs(delta.get('avg_distance_km', 0)):.2f} km improvement",
                                status="good"), unsafe_allow_html=True)
        st.markdown(metric_card("Coverage 5km",  f"{after.get('coverage_5km', 0):.1f}%",
                                sub=f"▲ +{delta.get('coverage_5km', 0):.1f}%",
                                status="good"), unsafe_allow_html=True)
        st.markdown(metric_card("Coverage 10km", f"{after.get('coverage_10km', 0):.1f}%",
                                sub=f"▲ +{delta.get('coverage_10km', 0):.1f}%",
                                status="good"), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">💰 Cost Analysis</div>', unsafe_allow_html=True)

        st.markdown(metric_card("Total Investment",
                                f"{costs.get('total_cost_mad', 0)/1_000_000:.1f}M MAD",
                                sub=f"≈ ${costs.get('total_cost_usd_approx', 0)/1_000:.0f}K USD"),
                    unsafe_allow_html=True)

        newly_covered = combined.get("population_newly_covered_10km", 0)
        st.markdown(metric_card("Newly Covered (10km)", f"{newly_covered/1000:.0f}K",
                                sub="people gained access",
                                status="good" if newly_covered > 0 else "warn"),
                    unsafe_allow_html=True)

        cpp = costs.get("cost_per_person_reached_mad")
        if cpp:
            st.markdown(metric_card("Cost per Person", f"{cpp:,.0f} MAD",
                                    sub="reached within 10km"), unsafe_allow_html=True)

    with col_map:
        st.markdown('<div class="section-header">🗺 Proposed Facility Locations</div>',
                    unsafe_allow_html=True)

        if "lon" not in facilities_gdf.columns:
            facilities_gdf = facilities_gdf.copy()
            facilities_gdf["lon"] = facilities_gdf.geometry.x
            facilities_gdf["lat"] = facilities_gdf.geometry.y

        with st.spinner("Rendering scenario map..."):
            fig = build_scenario_map(facilities_gdf, results)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            components.html(
                build_scenario_legend_html(facilities_gdf, results),
                height=130, scrolling=False,
            )

        interventions = results.get("interventions", {})
        if interventions:
            st.markdown('<div class="section-header">📋 Proposed Sites</div>', unsafe_allow_html=True)
            rows = []
            type_labels = {
                "new_facilities":      "🏗 New Facility",
                "mobile_units":        "🚐 Mobile Unit",
                "telemedicine_kiosks": "💻 Telemedicine",
            }
            for key, data in interventions.items():
                for site in data.get("sites", []):
                    rows.append({
                        "Type":             type_labels.get(key, key),
                        "Site ID":          site["site_id"],
                        "Latitude":         site["lat"],
                        "Longitude":        site["lon"],
                        "Pop. Served":      f"{site.get('cluster_population', site.get('population_served', 0)):,}",
                        "Nearest Existing": f"{site.get('nearest_existing_km', 0):.1f} km",
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATA EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def render_export_tab(
    filtered_facilities: gpd.GeoDataFrame,
    metrics_df: pd.DataFrame,
) -> None:
    st.markdown('<div class="section-header">📥 Export Data</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown(f"""
        <div class='scenario-card' style='border-top:4px solid {COLOR["primary"]};'>
            <div class='scenario-title' style='color:{COLOR["primary"]};'>📊 ACCESS METRICS</div>
            <div style='font-size:0.82rem;color:{COLOR["muted"]};margin-bottom:16px;'>
                Full access metrics CSV including nearest facility distances,
                coverage flags, and population weights for all grid cells.
            </div>
        """, unsafe_allow_html=True)
        if not metrics_df.empty:
            st.download_button(label="⬇ Download access_metrics.csv",
                               data=metrics_df.to_csv(index=False).encode("utf-8"),
                               file_name="morocco_access_metrics.csv",
                               mime="text/csv", use_container_width=True)
        else:
            st.info("Run the access metrics pipeline first.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='scenario-card' style='border-top:4px solid {COLOR["teal"]};'>
            <div class='scenario-title' style='color:{COLOR["teal"]};'>⚙️ SCENARIO RESULTS</div>
            <div style='font-size:0.82rem;color:{COLOR["muted"]};margin-bottom:16px;'>
                Scenario simulation output including before/after metrics,
                proposed site coordinates, cost analysis, and coverage gains.
            </div>
        """, unsafe_allow_html=True)
        scenario_data = st.session_state.get("scenario_results", load_scenario_results())
        if scenario_data:
            st.download_button(label="⬇ Download scenario_results.json",
                               data=json.dumps(scenario_data, indent=2, ensure_ascii=False).encode("utf-8"),
                               file_name="morocco_scenario_results.json",
                               mime="application/json", use_container_width=True)
        else:
            st.info("Run a scenario first.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='scenario-card' style='border-top:4px solid {COLOR["success"]};'>
            <div class='scenario-title' style='color:{COLOR["success"]};'>🏥 FILTERED FACILITIES</div>
            <div style='font-size:0.82rem;color:{COLOR["muted"]};margin-bottom:16px;'>
                Currently filtered facility dataset. Reflects your active region and
                facility type selections. <b>{len(filtered_facilities):,} facilities</b> selected.
            </div>
        """, unsafe_allow_html=True)
        export_df = filtered_facilities.drop(columns=["geometry"], errors="ignore")
        st.download_button(label="⬇ Download filtered_facilities.csv",
                           data=export_df.to_csv(index=False).encode("utf-8"),
                           file_name="morocco_facilities_filtered.csv",
                           mime="text/csv", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📖 Data Dictionary</div>', unsafe_allow_html=True)

    dict_data = {
        "Field": ["osm_id","facility_type","name_clean","sector","region","lat","lon",
                  "nearest_facility_km","within_5km","within_10km","within_20km"],
        "Description": [
            "OpenStreetMap element ID",
            "Standardised type: hospital / clinic / doctor / pharmacy",
            "Facility name (French preferred, Arabic fallback)",
            "Operator sector: public / private / ngo / unknown",
            "Moroccan administrative region (12 regions, 2015 reform)",
            "Latitude (WGS84 / EPSG:4326)", "Longitude (WGS84 / EPSG:4326)",
            "Straight-line distance to nearest facility (km)",
            "Population cell within 5 km of a facility",
            "Population cell within 10 km of a facility",
            "Population cell within 20 km of a facility",
        ],
        "Source": ["OSM","OSM","OSM","OSM","Computed","OSM","OSM",
                   "KD-tree","KD-tree","KD-tree","KD-tree"],
    }
    st.dataframe(pd.DataFrame(dict_data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    configure_page()
    inject_css()

    with st.spinner("Loading Morocco healthcare data..."):
        facilities_gdf = load_facilities()
        pop_gdf        = load_popgrid()
        metrics_df     = load_metrics()

    render_header()
    config = render_sidebar(facilities_gdf)

    filtered = apply_filters(
        facilities_gdf,
        region=config["region"],
        facility_types=config["facility_types"],
    )

    if len(filtered) == 0:
        st.warning("No facilities match the current filters. Adjust the sidebar selections.")
        return

    with st.spinner("Computing access metrics..."):
        baseline = compute_baseline_metrics(pop_gdf, filtered)

    region_label = config["region"]
    st.markdown(f"""
    <div style='background:{COLOR["light_bg"]};border:1px solid {COLOR["border"]};
                border-radius:6px;padding:8px 16px;font-size:0.8rem;
                color:{COLOR["muted"]};margin-bottom:12px;display:flex;gap:24px;'>
        <span>📍 <b>Region:</b> {region_label}</span>
        <span>🏥 <b>Types:</b> {", ".join(config["facility_types"])}</span>
        <span>📏 <b>Radius:</b> {config["radius_km"]} km</span>
        <span>🔢 <b>Facilities shown:</b> {len(filtered):,}</span>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "  🗺  ACCESS OVERVIEW  ",
        "  ⚙️  SCENARIO SIMULATION  ",
        "  📥  DATA EXPORT  ",
    ])

    with tab1:
        render_overview_tab(filtered, pop_gdf, baseline, config["radius_km"])

    with tab2:
        render_scenario_tab(
            facilities_gdf=facilities_gdf,
            pop_gdf=pop_gdf,
            baseline=baseline,
            scenario_config=config,
        )

    with tab3:
        render_export_tab(filtered, metrics_df)


if __name__ == "__main__":
    main()
