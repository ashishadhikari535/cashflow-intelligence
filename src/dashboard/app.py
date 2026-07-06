"""
src/dashboard/app.py
─────────────────────
CFO Cash Flow Intelligence Dashboard — Streamlit entry point.

Layout:
  Sidebar     — filters and controls
  Tab 1       — Cash Flow Forecast (headline chart + forecast table)
  Tab 2       — AR Risk Monitor   (flagged invoice table + waterfall on click)
  Tab 3       — Portfolio Health  (aging heatmap + segment analysis)
  Tab 4       — Audit Export      (PDF report generation)

Design philosophy:
  Every number on this dashboard is explainable.
  The CFO can click any flagged invoice and see exactly
  why it was flagged — in plain English, backed by SHAP.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import (
    FEATURE_STORE_PATH,
    AR_AGING_PATH,
    CLASSIFIER_PATH,
    DASHBOARD_CFG,
    CLASSIFIER_CFG,
)
from src.utils.logger import get_logger
from src.models.classifier import load_model, predict as classify
from src.models.forecaster import (
    build_forecast_dataset, generate_forecast
)
from src.models.explainer import (
    load_shap_explainer, compute_shap_values,
    generate_narrative,
    SHAP_EXPLAINER_PATH
)
from src.dashboard.charts import (
    forecast_chart,
    aging_heatmap, shap_waterfall_chart,
    risk_distribution_chart, segment_bar_chart
)
from src.dashboard.export import generate_pdf_report

log = get_logger(__name__)


def inject_dashboard_styles():
    st.markdown(
        """
        <style>
        :root {
            --cf-bg-1: #f3f7ff;
            --cf-bg-2: #eef7f4;
            --cf-card-1: #f8fbff;
            --cf-card-2: #f3f8ff;
            --cf-border: #d8e5fb;
            --cf-shadow: rgba(20, 64, 140, 0.10);
            --cf-title: #0b2f66;
            --cf-body: #5f6d80;
            --cf-accent: #0b5cab;
            --cf-accent-soft: #e7f1ff;
        }
        .stApp {
            background:
                radial-gradient(1200px 420px at 12% -10%, #dcecff 0%, rgba(220, 236, 255, 0) 62%),
                radial-gradient(900px 380px at 94% 4%, #daf4eb 0%, rgba(218, 244, 235, 0) 58%),
                linear-gradient(180deg, var(--cf-bg-1) 0%, var(--cf-bg-2) 100%);
        }
        [data-testid="stHeader"],
        [data-testid="stToolbar"] {
            display: none;
        }
        [data-testid="stAppViewContainer"] > .main {
            padding-top: 0rem;
        }
        .main .block-container {
            padding-top: 0.45rem;
            padding-bottom: 1.2rem;
        }
        @media (max-width: 768px) {
            .main .block-container {
                padding-top: 0.3rem;
            }
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f7fbff 0%, #f3f8ff 100%);
            border-right: 1px solid #d9e6fb;
        }
        [data-testid="stSidebar"] > div:first-child {
            overflow-y: hidden;
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 0.65rem;
            padding-left: 0.75rem;
            padding-right: 0.75rem;
        }
        [data-testid="stSidebar"] hr {
            border-top: 1px solid #d6e4fb;
            margin-top: 0.45rem;
            margin-bottom: 0.45rem;
        }
        .sidebar-hero {
            background: linear-gradient(125deg, #0e4b96 0%, #0b3d7d 100%);
            border: 1px solid #2c5f9f;
            border-radius: 14px;
            padding: 10px 10px 8px 10px;
            margin-bottom: 0.45rem;
            box-shadow: 0 8px 20px rgba(13, 50, 103, 0.22);
        }
        .sidebar-hero-title {
            font-size: 0.96rem;
            line-height: 1.2;
            font-weight: 800;
            color: #ffffff;
            margin-bottom: 4px;
        }
        .sidebar-hero-sub {
            font-size: 0.76rem;
            color: #d8e9ff;
            margin-bottom: 0;
        }
        .sidebar-hero-chip {
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 700;
            color: #ecf4ff;
            background: rgba(255, 255, 255, 0.15);
            border: 1px solid rgba(255, 255, 255, 0.28);
            border-radius: 999px;
            padding: 4px 9px;
        }
        .sidebar-section-title {
            font-size: 0.73rem;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            font-weight: 750;
            color: #40638f;
            margin-top: 0.05rem;
            margin-bottom: 0.25rem;
        }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown p {
            color: #2d4f77;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="base-input"] > div {
            border: 1px solid #cadcf8;
            border-radius: 10px;
            background: #fbfdff;
        }
        [data-testid="stSidebar"] .stSlider > div[data-baseweb="slider"] {
            padding-top: 0.25rem;
            padding-bottom: 0.2rem;
        }
        [data-testid="stSidebar"] .stSelectbox,
        [data-testid="stSidebar"] .stSlider {
            margin-bottom: 0.2rem;
        }
        [data-testid="stSidebar"] [data-testid="stMetric"] {
            background: #f9fcff;
            border: 1px solid #d5e3f8;
            border-radius: 10px;
            padding: 8px 10px;
            margin-bottom: 0.45rem;
        }
        .dashboard-hero {
            background: linear-gradient(120deg, #0f4a95 0%, #0b3f80 54%, #0a346c 100%);
            border: 1px solid #275a98;
            border-radius: 18px;
            padding: 18px 20px;
            margin-bottom: 0.75rem;
            box-shadow: 0 14px 28px rgba(10, 44, 96, 0.24);
        }
        .hero-kicker {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #b8d4ff;
            margin-bottom: 5px;
            font-weight: 700;
        }
        .hero-title {
            font-size: 1.9rem;
            line-height: 1.1;
            font-weight: 800;
            color: #ffffff;
            margin-bottom: 5px;
        }
        .hero-subtitle {
            font-size: 0.93rem;
            color: #dfeeff;
            margin-bottom: 10px;
        }
        .hero-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .hero-chip {
            display: inline-block;
            font-size: 0.76rem;
            font-weight: 700;
            color: #eaf3ff;
            background: rgba(255, 255, 255, 0.14);
            border: 1px solid rgba(255, 255, 255, 0.26);
            border-radius: 999px;
            padding: 5px 10px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            padding-bottom: 3px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            border: 1px solid #d7e5fb;
            background: #f6faff;
            color: #355176;
            font-weight: 600;
            padding: 7px 12px;
        }
        .stTabs [aria-selected="true"] {
            background: var(--cf-accent-soft);
            border-color: #b7d2f7;
            color: #0c3b73;
        }
        .stButton > button {
            border-radius: 10px;
            border: 1px solid #195ba4;
        }
        .stMarkdown hr {
            border-top: 1px solid #d8e5fb;
            margin-top: 0.7rem;
            margin-bottom: 0.7rem;
        }
        .kpi-card {
            background: linear-gradient(140deg, var(--cf-card-1) 0%, var(--cf-card-2) 100%);
            border: 1px solid var(--cf-border);
            border-radius: 14px;
            padding: 14px 14px 12px 14px;
            box-shadow: 0 6px 16px var(--cf-shadow);
            min-height: 116px;
            transition: transform 0.14s ease, box-shadow 0.14s ease;
        }
        .kpi-card:hover {
            transform: translateY(-1px);
            box-shadow: 0 9px 20px rgba(20, 64, 140, 0.14);
        }
        .kpi-title {
            font-size: 0.79rem;
            font-weight: 650;
            color: #4f6a8f;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            margin-bottom: 4px;
        }
        .kpi-value {
            font-size: 1.65rem;
            line-height: 1.18;
            font-weight: 800;
            color: var(--cf-title);
            margin-bottom: 8px;
            white-space: nowrap;
        }
        .kpi-sub {
            font-size: 0.80rem;
            color: var(--cf-body);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(title: str, value: str, subtitle: str = ""):
    subtitle_html = subtitle if subtitle else "&nbsp;"
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{subtitle_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard_header(filters: dict, meta: dict):
    st.markdown(
        f"""
        <div class="dashboard-hero">
            <div class="hero-kicker">Finance Intelligence Workspace</div>
            <div class="hero-title">{DASHBOARD_CFG.app_icon} {DASHBOARD_CFG.app_title}</div>
            <div class="hero-subtitle">
                Explainable AR risk scoring and probabilistic cash flow forecasting on SAP FI-structured data.
            </div>
            <div class="hero-meta">
                <span class="hero-chip">Risk threshold: {filters['threshold']:.2f}</span>
                <span class="hero-chip">Forecast horizon: {filters['horizon']} days</span>
                <span class="hero-chip">CV AUC: {meta['cv_auc']:.3f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title=DASHBOARD_CFG.app_title,
    page_icon=DASHBOARD_CFG.app_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# CACHED DATA LOADERS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_model()


@st.cache_resource(show_spinner="Loading explainer...")
def get_explainer():
    if SHAP_EXPLAINER_PATH.exists():
        return load_shap_explainer()
    return None


@st.cache_data(show_spinner="Loading data...")
def get_feature_store():
    return pd.read_csv(FEATURE_STORE_PATH, parse_dates=["BLDAT", "FAEDT"])


@st.cache_data(show_spinner="Loading AR aging...")
def get_aging():
    return pd.read_csv(AR_AGING_PATH)


@st.cache_data(show_spinner="Building forecast dataset...")
def get_forecast_dataset(
    region: str = "All",
    industry: str = "All",
    cache_tag: str = "forecast_v3",
):
    region_value = None if region == "All" else region
    industry_value = None if industry == "All" else industry
    _ = cache_tag
    return build_forecast_dataset(region=region_value, industry=industry_value)


@st.cache_data(show_spinner="Scoring invoices...")
def get_scored(_model, _meta, _fs):
    return classify(_model, _fs, _meta["threshold"])


@st.cache_data(show_spinner="Generating forecast...")
def get_forecast(
    _df,
    region: str = "All",
    industry: str = "All",
    cache_tag: str = "forecast_v3",
):
    region_value = None if region == "All" else region
    industry_value = None if industry == "All" else industry
    _ = cache_tag
    return generate_forecast(_df, region=region_value, industry=industry_value)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar(scored: pd.DataFrame, meta: dict) -> dict:
    st.sidebar.markdown(
        f"""
        <div class="sidebar-hero">
            <div class="sidebar-hero-title">{DASHBOARD_CFG.app_icon} CFO Intelligence</div>
            <div class="sidebar-hero-sub">Portfolio risk controls and forecast settings</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown('<div class="sidebar-section-title">Filters</div>', unsafe_allow_html=True)

    # Industry filter
    industry_values = (
        scored["INDUSTRY"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    industries = ["All"] + sorted(industry_values[industry_values != ""].unique().tolist())
    selected_industry = st.sidebar.selectbox("Industry", industries)

    # Region filter
    region_values = (
        scored["REGION"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    regions = ["All"] + sorted(region_values[region_values != ""].unique().tolist())
    selected_region = st.sidebar.selectbox("Region", regions)

    # Risk threshold slider
    threshold = st.sidebar.slider(
        "Risk threshold",
        min_value=0.20, max_value=0.80,
        value=float(meta["threshold"]),
        step=0.05,
        help="Invoices above this probability are flagged as high risk"
    )

    # Forecast horizon
    horizon = st.sidebar.selectbox(
        "Forecast horizon",
        options=[30, 60, 90],
        index=0,
        format_func=lambda x: f"{x} days"
    )

    return {
        "industry":  selected_industry,
        "region":    selected_region,
        "threshold": threshold,
        "horizon":   horizon,
    }


# ─────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────
def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    filtered = df.copy()
    if filters["industry"] != "All":
        filtered = filtered[
            filtered["INDUSTRY"].astype(str).str.strip() == filters["industry"]
        ]
    if filters["region"] != "All":
        filtered = filtered[
            filtered["REGION"].astype(str).str.strip() == filters["region"]
        ]
    return filtered


# ─────────────────────────────────────────────
# KPI HEADER
# ─────────────────────────────────────────────
def render_kpis(scored: pd.DataFrame, forecast_df: pd.DataFrame, filters: dict):
    filtered   = apply_filters(scored, filters)
    flagged    = filtered[filtered["LATE_PROB"] >= filters["threshold"]]
    total_ar   = filtered["DMBTR"].sum()
    at_risk_ar = flagged["DMBTR"].sum()
    risk_share = (at_risk_ar / total_ar) if total_ar > 0 else 0.0

    # Get median forecast for selected horizon
    horizon_match = forecast_df[forecast_df["Horizon_Days"] == filters["horizon"]]
    horizon_row = horizon_match.iloc[0] if not horizon_match.empty else None

    avg_risk = float(filtered["LATE_PROB"].mean()) if len(filtered) else 0.0
    forecast_value = "N/A"
    forecast_subtitle = "No forecast for selected horizon"
    if horizon_row is not None:
        interval = (horizon_row["Upper_90pct"] - horizon_row["Lower_10pct"]) / 2 / 1e6
        forecast_value = f"${horizon_row['Median_50pct']/1e6:.2f}M"
        forecast_subtitle = f"±${interval:.2f}M uncertainty range"

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_kpi_card("Total AR", f"${total_ar/1e6:.2f}M", "Outstanding receivables")
    with col2:
        render_kpi_card("At-Risk AR", f"${at_risk_ar/1e6:.2f}M", f"{risk_share:.1%} of portfolio")
    with col3:
        render_kpi_card("Flagged Invoices", f"{len(flagged):,}", f"of {len(filtered):,} total invoices")
    with col4:
        render_kpi_card("Avg Risk Score", f"{avg_risk:.1%}", "Mean late-payment probability")
    with col5:
        render_kpi_card(f"{filters['horizon']}-Day Forecast", forecast_value, forecast_subtitle)


# ─────────────────────────────────────────────
# TAB 1 — CASH FLOW FORECAST
# ─────────────────────────────────────────────
def render_forecast_tab(
    forecast_dataset: pd.DataFrame,
    forecast_df: pd.DataFrame,
    filters: dict,
    fallback_reason: str | None = None,
):
    st.subheader("Cash Flow Forecast")
    st.caption("Historical collections + 30/60/90-day probabilistic forecast with confidence bands")
    if fallback_reason:
        st.warning(
            "Using portfolio-level forecast data for this view due to insufficient segment history. "
            f"Reason: {fallback_reason}"
        )
    if "Model_Source" in forecast_df.columns:
        source = str(forecast_df["Model_Source"].iloc[0]).title()
        segment = str(forecast_df["Model_Segment"].iloc[0])
        st.caption(
            f"Forecast source: {source} model"
            f"{'' if segment in ('All', '') else f' ({segment})'} "
            f"| Filters -> Region: {filters['region']} | Industry: {filters['industry']}"
        )
    st.caption(
        f"History points: {len(forecast_dataset)} | "
        f"Historical total: ${forecast_dataset['ACTUAL_COLLECTIONS'].sum()/1e6:.2f}M"
    )

    # Headline chart
    fig = forecast_chart(forecast_dataset, forecast_df, filters["horizon"])
    chart_key = (
        f"forecast_chart_{filters['region']}_{filters['industry']}_{filters['horizon']}"
    )
    st.plotly_chart(fig, use_container_width=True, key=chart_key)

    # Forecast table
    st.markdown("#### Forecast Detail")
    display_cols = ["Horizon", "Lower_10pct", "Median_50pct", "Upper_90pct", "Uncertainty_Pct"]
    fmt = {
        "Lower_10pct":    "${:,.0f}",
        "Median_50pct":   "${:,.0f}",
        "Upper_90pct":    "${:,.0f}",
        "Uncertainty_Pct": "{:.1f}%",
    }
    st.dataframe(
        forecast_df[display_cols].style.format(fmt),
        use_container_width=True, hide_index=True
    )

    st.info(
        "**Reading this chart:** The shaded band represents the 80% confidence interval "
        "(10th–90th percentile). On a good month, collections land near the upper band. "
        "In a stress scenario, near the lower band. Plan cash needs against the lower band."
    )


# ─────────────────────────────────────────────
# TAB 2 — AR RISK MONITOR
# ─────────────────────────────────────────────
def render_risk_tab(scored: pd.DataFrame, explainer, meta: dict, filters: dict):
    st.subheader("AR Risk Monitor")
    st.caption("Invoices ranked by late payment probability — click any row to see the full audit explanation")

    filtered = apply_filters(scored, filters)
    flagged  = (
        filtered[filtered["LATE_PROB"] >= filters["threshold"]]
        .sort_values("LATE_PROB", ascending=False)
        .head(DASHBOARD_CFG.top_n_risks)
        .reset_index(drop=True)
    )

    if len(flagged) == 0:
        st.success("No invoices flagged at the current threshold. Try lowering the threshold in the sidebar.")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        # Risk table
        display = flagged[[
            "BELNR", "KUNNR", "INDUSTRY", "REGION",
            "DMBTR", "FAEDT", "LATE_PROB", "RISK_LABEL"
        ]].copy()
        display["DMBTR"]     = display["DMBTR"].apply(lambda x: f"${x:,.0f}")
        display["LATE_PROB"] = display["LATE_PROB"].apply(lambda x: f"{x:.1%}")

        st.dataframe(
            display.rename(columns={
                "BELNR":      "Invoice",
                "KUNNR":      "Customer",
                "INDUSTRY":   "Industry",
                "REGION":     "Region",
                "DMBTR":      "Amount",
                "FAEDT":      "Due Date",
                "LATE_PROB":  "Risk Score",
                "RISK_LABEL": "Risk Level",
            }),
            use_container_width=True, hide_index=True
        )

        # Risk distribution chart
        fig_dist = risk_distribution_chart(filtered, filters["threshold"])
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        # Invoice deep-dive
        st.markdown("#### Invoice Deep-Dive")
        selected_idx = st.selectbox(
            "Select invoice",
            options=range(len(flagged)),
            format_func=lambda i: f"{flagged.iloc[i]['BELNR']} — {flagged.iloc[i]['LATE_PROB']:.1%}"
        )

        if explainer is not None:
            selected = flagged.iloc[selected_idx]
            X_sel    = selected[CLASSIFIER_CFG.feature_cols]

            # Waterfall chart
            shap_vals = compute_shap_values(explainer, X_sel.to_frame().T)
            fig_wf    = shap_waterfall_chart(
                shap_vals[0],
                CLASSIFIER_CFG.feature_cols,
                X_sel,
                selected["LATE_PROB"],
                str(selected.get("BELNR", ""))
            )
            st.plotly_chart(fig_wf, use_container_width=True)

            # Audit narrative
            st.markdown("#### Audit Narrative")
            narrative = generate_narrative(
                selected, shap_vals[0],
                selected["LATE_PROB"],
                filters["threshold"]
            )
            st.text(narrative)
        else:
            st.warning("SHAP explainer not found. Run `make explain` first.")


# ─────────────────────────────────────────────
# TAB 3 — PORTFOLIO HEALTH
# ─────────────────────────────────────────────
def render_portfolio_tab(scored: pd.DataFrame, aging: pd.DataFrame, filters: dict):
    st.subheader("Portfolio Health")

    filtered = apply_filters(scored, filters)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### AR Aging Heatmap")
        fig_aging = aging_heatmap(aging)
        st.plotly_chart(fig_aging, use_container_width=True)

    with col2:
        st.markdown("#### Risk by Industry")
        fig_seg = segment_bar_chart(filtered, "INDUSTRY")
        st.plotly_chart(fig_seg, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Risk by Region")
        fig_reg = segment_bar_chart(filtered, "REGION")
        st.plotly_chart(fig_reg, use_container_width=True)

    with col4:
        st.markdown("#### Late Rate Trend")
        filtered_sorted = filtered.copy()
        filtered_sorted["PERIOD"] = filtered_sorted["BLDAT"].dt.to_period("Q").astype(str)
        trend = (
            filtered_sorted.groupby("PERIOD")["LATE_FLAG"]
            .mean().mul(100).reset_index()
        )
        import plotly.express as px
        fig_trend = px.line(
            trend, x="PERIOD", y="LATE_FLAG",
            markers=True,
            labels={"LATE_FLAG": "% paid late", "PERIOD": "Quarter"},
            title="Portfolio late rate by quarter"
        )
        fig_trend.update_traces(line_color="#E24B4A")
        fig_trend.update_layout(height=300, margin=dict(t=40, b=20))
        st.plotly_chart(fig_trend, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 4 — AUDIT EXPORT
# ─────────────────────────────────────────────
def render_export_tab(scored: pd.DataFrame, forecast_df: pd.DataFrame, meta: dict, filters: dict):
    st.subheader("Audit Export")
    st.caption("Generate a PDF risk report for the CFO or audit team")

    filtered = apply_filters(scored, filters)
    flagged  = (
        filtered[filtered["LATE_PROB"] >= filters["threshold"]]
        .sort_values("LATE_PROB", ascending=False)
        .head(DASHBOARD_CFG.top_n_risks)
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Report Parameters")
        report_title  = st.text_input("Report title", "AR Risk & Cash Flow Report")
        include_narr  = st.checkbox("Include audit narratives", value=True)
        top_n         = st.slider("Top N flagged invoices", 5, 50, 20)

    with col2:
        st.markdown("#### Summary Preview")
        avg_flagged_risk = float(flagged["LATE_PROB"].mean()) if len(flagged) else 0.0
        forecast_30 = forecast_df[forecast_df["Horizon_Days"] == 30]
        forecast_30_text = (
            f"${forecast_30.iloc[0]['Median_50pct']/1e6:.2f}M"
            if not forecast_30.empty else "N/A"
        )
        st.metric("Invoices to export",  len(flagged))
        st.metric("Total at-risk AR",    f"${flagged['DMBTR'].sum()/1e6:.2f}M")
        st.metric("Avg risk score",      f"{avg_flagged_risk:.1%}")
        st.metric("Forecast (30d)", forecast_30_text)

    st.markdown("---")

    if st.button("Generate PDF Report", type="primary"):
        with st.spinner("Generating report..."):
            pdf_path = generate_pdf_report(
                flagged, forecast_df, meta,
                title=report_title,
                include_narratives=include_narr,
                top_n=top_n
            )
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="Download PDF Report",
                    data=f.read(),
                    file_name=pdf_path.name,
                    mime="application/pdf"
                )
            st.success(f"Report generated: {pdf_path.name}")

    # CSV export
    st.markdown("#### Quick CSV Export")
    csv_data = flagged[[
        "BELNR", "KUNNR", "INDUSTRY", "REGION",
        "DMBTR", "FAEDT", "LATE_PROB", "RISK_LABEL"
    ]].to_csv(index=False)

    st.download_button(
        label="Download Flagged Invoices (CSV)",
        data=csv_data,
        file_name="flagged_invoices.csv",
        mime="text/csv"
    )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    inject_dashboard_styles()
    forecast_cache_tag = "forecast_v4"

    # Validate artifacts exist
    if not CLASSIFIER_PATH.exists():
        st.error("Model not found. Run `make train` before launching the dashboard.")
        st.stop()

    # Load everything
    model, meta      = get_model()
    explainer        = get_explainer()
    fs               = get_feature_store()
    aging            = get_aging()
    scored           = get_scored(model, meta, fs)

    # Sidebar filters
    filters = render_sidebar(scored, meta)

    # Forecast dataset and predictions for selected portfolio slice.
    forecast_fallback_reason = None
    try:
        forecast_dataset = get_forecast_dataset(
            filters["region"],
            filters["industry"],
            cache_tag=forecast_cache_tag,
        )
    except Exception as exc:
        forecast_fallback_reason = str(exc)
        log.warning(
            "Forecast dataset fallback to portfolio-level "
            f"[region={filters['region']}, industry={filters['industry']}]: {exc}"
        )
        st.info(
            "Forecast reverted to portfolio-level view because selected filters "
            "do not have enough history."
        )
        forecast_dataset = get_forecast_dataset("All", "All", cache_tag=forecast_cache_tag)

    forecast_df = get_forecast(
        forecast_dataset,
        filters["region"],
        filters["industry"],
        cache_tag=forecast_cache_tag,
    )

    # Header
    render_dashboard_header(filters, meta)

    # KPI row
    render_kpis(scored, forecast_df, filters)
    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Cash Flow Forecast",
        "🚨 AR Risk Monitor",
        "🏥 Portfolio Health",
        "📄 Audit Export",
    ])

    with tab1:
        render_forecast_tab(
            forecast_dataset,
            forecast_df,
            filters,
            fallback_reason=forecast_fallback_reason,
        )

    with tab2:
        render_risk_tab(scored, explainer, meta, filters)

    with tab3:
        render_portfolio_tab(scored, aging, filters)

    with tab4:
        render_export_tab(scored, forecast_df, meta, filters)


if __name__ == "__main__":
    main()
