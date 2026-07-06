"""
src/dashboard/charts.py
────────────────────────
Plotly chart builders for the CFO dashboard.
Each function returns a go.Figure — ready to drop into st.plotly_chart().

Consistent color palette across all charts:
  Green  (#1D9E75) — positive / on-time / lower risk
  Amber  (#EF9F27) — medium risk / warning
  Red    (#E24B4A) — high risk / late / alert
  Blue   (#378ADD) — neutral / historical / informational
  Purple (#7F77DD) — forecast / model output
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import FORECASTER_CFG, CLASSIFIER_CFG

# ─────────────────────────────────────────────
# THEME DEFAULTS
# ─────────────────────────────────────────────
LAYOUT_DEFAULTS = dict(
    font_family="Arial, sans-serif",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    hoverlabel=dict(bgcolor="white", font_size=12),
)


# ─────────────────────────────────────────────
# 1. CASH FLOW FORECAST CHART
# ─────────────────────────────────────────────
def forecast_chart(
    df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    selected_horizon: int = 30
) -> go.Figure:
    """
    Historical collections + forward forecast with confidence bands.
    Headline chart on the dashboard — should be immediately readable.
    """
    hist = df.tail(18).copy()
    hist["COLLECT_MONTH"] = pd.to_datetime(hist["COLLECT_MONTH"], errors="coerce")
    hist = hist.dropna(subset=["COLLECT_MONTH"]).sort_values("COLLECT_MONTH")

    fig = go.Figure()

    # Historical confidence-style band based on rolling variability (visual context).
    hist_m = hist["ACTUAL_COLLECTIONS"] / 1e6
    hist_std = hist_m.rolling(window=3, min_periods=2).std()
    fallback_std = max(float(hist_m.std(skipna=True) * 0.25), 0.15)
    hist_std = hist_std.fillna(fallback_std)
    hist_upper = hist_m + hist_std
    hist_lower = (hist_m - hist_std).clip(lower=0)

    fig.add_trace(go.Scatter(
        x=list(hist["COLLECT_MONTH"]) + list(hist["COLLECT_MONTH"])[::-1],
        y=list(hist_upper) + list(hist_lower)[::-1],
        fill="toself",
        fillcolor="rgba(55, 138, 221, 0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Historical confidence band",
        hoverinfo="skip"
    ))

    # Historical actuals
    fig.add_trace(go.Scatter(
        x=hist["COLLECT_MONTH"],
        y=hist["ACTUAL_COLLECTIONS"] / 1e6,
        mode="lines+markers",
        name="Historical collections",
        line=dict(color="#378ADD", width=2.5),
        marker=dict(size=5),
        hovertemplate="<b>%{x}</b><br>$%{y:.2f}M<extra></extra>"
    ))

    # Forecast bands
    last_month = hist["COLLECT_MONTH"].iloc[-1]
    x_forecast = [
        last_month + pd.Timedelta(days=int(r["Horizon_Days"]))
        for _, r in forecast_df.iterrows()
    ]
    medians    = forecast_df["Median_50pct"].values / 1e6
    lowers     = forecast_df["Lower_10pct"].values / 1e6
    uppers     = forecast_df["Upper_90pct"].values / 1e6

    # Confidence band (filled)
    fig.add_trace(go.Scatter(
        x=x_forecast + x_forecast[::-1],
        y=list(uppers) + list(lowers[::-1]),
        fill="toself",
        fillcolor="rgba(29, 158, 117, 0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="80% confidence band",
        hoverinfo="skip"
    ))

    # Upper / lower dashed lines
    fig.add_trace(go.Scatter(
        x=x_forecast, y=uppers,
        mode="lines", name="Upper (90th pct)",
        line=dict(color="#1D9E75", width=1, dash="dot"),
        hovertemplate="<b>%{x}</b><br>Upper: $%{y:.2f}M<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=x_forecast, y=lowers,
        mode="lines", name="Lower (10th pct)",
        line=dict(color="#1D9E75", width=1, dash="dot"),
        hovertemplate="<b>%{x}</b><br>Lower: $%{y:.2f}M<extra></extra>"
    ))

    # Forecast median
    fig.add_trace(go.Scatter(
        x=x_forecast, y=medians,
        mode="lines+markers", name="Forecast (median)",
        line=dict(color="#1D9E75", width=2.5),
        marker=dict(size=8, symbol="square"),
        hovertemplate="<b>%{x}</b><br>Median: $%{y:.2f}M<extra></extra>"
    ))

    # Highlight selected horizon
    sel_row = forecast_df[forecast_df["Horizon_Days"] == selected_horizon]
    if len(sel_row):
        r = sel_row.iloc[0]
        selected_x = last_month + pd.Timedelta(days=int(selected_horizon))
        fig.add_annotation(
            x=selected_x,
            y=r["Median_50pct"] / 1e6,
            text=f"${r['Median_50pct']/1e6:.2f}M<br>±${(r['Upper_90pct']-r['Lower_10pct'])/2/1e6:.2f}M",
            showarrow=True, arrowhead=2,
            bgcolor="white", bordercolor="#1D9E75", borderwidth=1,
            font=dict(size=11, color="#1D9E75")
        )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="Cash Collections — Historical & Forecast",
        yaxis_title="Collections ($M)",
        yaxis_tickprefix="$", yaxis_ticksuffix="M",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
        margin=dict(t=50, b=40, l=40, r=20),
    )
    return fig


# ─────────────────────────────────────────────
# 2. RISK DISTRIBUTION CHART
# ─────────────────────────────────────────────
def risk_distribution_chart(
    scored: pd.DataFrame,
    threshold: float
) -> go.Figure:
    """
    Histogram of late payment probabilities across portfolio.
    Vertical line at threshold — shows how many invoices are flagged.
    """
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=scored["LATE_PROB"],
        nbinsx=40,
        marker_color="#378ADD",
        opacity=0.75,
        name="Invoice count",
        hovertemplate="Risk: %{x:.1%}<br>Count: %{y}<extra></extra>"
    ))

    fig.add_vline(
        x=threshold,
        line_dash="dash", line_color="#E24B4A", line_width=2,
        annotation_text=f"Threshold ({threshold:.2f})",
        annotation_position="top right",
        annotation_font_color="#E24B4A"
    )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="Risk Score Distribution",
        xaxis_title="Late payment probability",
        yaxis_title="Invoice count",
        xaxis_tickformat=".0%",
        height=280,
        showlegend=False,
        margin=dict(t=40, b=30, l=40, r=20),
    )
    return fig


# ─────────────────────────────────────────────
# 3. AR AGING HEATMAP
# ─────────────────────────────────────────────
def aging_heatmap(aging: pd.DataFrame) -> go.Figure:
    """
    Heatmap of AR aging buckets — each customer is a row,
    each bucket is a column. Color = outstanding balance.
    Top 20 customers by total outstanding.
    """
    aging_buckets = [
        "Current", "1-30 days", "31-60 days",
        "61-90 days", "91-180 days", "180+ days"
    ]
    buckets_present = [b for b in aging_buckets if b in aging.columns]

    top20 = aging.nlargest(20, "TOTAL_OUTSTANDING").reset_index(drop=True)
    z     = top20[buckets_present].values / 1e3   # thousands

    fig = go.Figure(go.Heatmap(
        z=z,
        x=buckets_present,
        y=top20["KUNNR"].astype(str),
        colorscale=[[0, "#E1F5EE"], [0.5, "#EF9F27"], [1, "#E24B4A"]],
        text=np.round(z, 1),
        texttemplate="$%{text}K",
        textfont=dict(size=9),
        hovertemplate="Customer: %{y}<br>Bucket: %{x}<br>Balance: $%{z:.1f}K<extra></extra>",
        colorbar=dict(title="$K", tickprefix="$", ticksuffix="K")
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="AR Aging Heatmap — Top 20 Customers ($000s)",
        xaxis_title="Aging bucket",
        yaxis_title="Customer",
        height=420,
        margin=dict(t=50, b=40, l=90, r=20),
    )
    return fig


# ─────────────────────────────────────────────
# 4. SHAP WATERFALL CHART
# ─────────────────────────────────────────────
def shap_waterfall_chart(
    shap_values: np.ndarray,
    feature_names: list,
    feature_values: pd.Series,
    late_prob: float,
    invoice_id: str
) -> go.Figure:
    """
    Horizontal waterfall showing each feature's contribution
    to the invoice's risk score. Red = increases risk, Blue = reduces risk.
    """
    df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP":    shap_values,
        "Value":   feature_values.values
    }).sort_values("SHAP", key=abs, ascending=True).tail(8)

    colors = ["#E24B4A" if v > 0 else "#378ADD" for v in df["SHAP"]]
    labels = [f"{v:.3g}" for v in df["Value"]]

    fig = go.Figure(go.Bar(
        x=df["SHAP"],
        y=df["Feature"],
        orientation="h",
        marker_color=colors,
        text=labels,
        textposition="outside",
        textfont=dict(size=9, color="gray"),
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.4f}<br>Value: %{text}<extra></extra>"
    ))

    fig.add_vline(x=0, line_color="black", line_width=0.8)

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=f"Invoice {invoice_id} — Risk: {late_prob:.1%}",
        xaxis_title="SHAP value",
        height=320,
        margin=dict(t=45, b=30, l=160, r=60),
    )
    return fig


# ─────────────────────────────────────────────
# 5. SEGMENT BAR CHART
# ─────────────────────────────────────────────
def segment_bar_chart(scored: pd.DataFrame, segment_col: str) -> go.Figure:
    """
    Average late probability by segment (industry or region).
    Sorted descending — highest-risk segments at top.
    """
    seg = (
        scored.groupby(segment_col)["LATE_PROB"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("mean", ascending=True)
    )

    colors = [
        "#E24B4A" if v > 0.5 else
        "#EF9F27" if v > 0.3 else
        "#1D9E75"
        for v in seg["mean"]
    ]

    fig = go.Figure(go.Bar(
        x=seg["mean"],
        y=seg[segment_col],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1%}  (n={n})" for v, n in zip(seg["mean"], seg["count"])],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Avg risk: %{x:.1%}<extra></extra>"
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=f"Average Risk Score by {segment_col.title()}",
        xaxis_title="Avg late probability",
        xaxis_tickformat=".0%",
        height=300,
        showlegend=False,
        margin=dict(t=45, b=30, l=110, r=80),
    )
    return fig
