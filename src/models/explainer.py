"""
src/models/explainer.py
────────────────────────
SHAP explainability layer + plain English audit narrative generator.

Two outputs per flagged invoice:
  1. SHAP waterfall — which features drove the risk score and by how much
  2. Audit narrative — plain English explanation a CFO can read and act on

Design philosophy:
  A black-box risk score is useless in finance.
  "Invoice C000042 has 73% late probability" means nothing without:
    → WHY is it 73%?
    → WHAT has changed for this customer?
    → WHAT action should the collections team take?

  This module answers all three — using the same logic an auditor
  would use manually, now automated and consistent at scale.

Audit intuition encoded in narrative templates:
  - Customer payment deterioration trend
  - Invoice size anomaly vs customer baseline
  - Quarter-end timing risk
  - New customer with limited history
  - Industry/region payment culture signals
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import warnings
from pathlib import Path
from typing import Optional
import sys

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import (
    FEATURE_STORE_PATH,
    SHAP_EXPLAINER_PATH,
    EXPORTS_DIR,
    CLASSIFIER_CFG,
    DASHBOARD_CFG,
)
from src.utils.logger import format_path, get_logger
from src.models.classifier import load_model, predict as classify, get_Xy

log = get_logger(__name__)


# ─────────────────────────────────────────────
# BUILD SHAP EXPLAINER
# ─────────────────────────────────────────────
def build_shap_explainer(model, X_background: pd.DataFrame):
    """
    Builds a SHAP TreeExplainer using a background sample.
    TreeExplainer is exact for tree models — no approximation needed.
    Background sample sets the baseline (E[f(x)]) for SHAP values.
    """
    log.info("Building SHAP TreeExplainer...")

    # Background = random 200-row sample — sets the baseline expectation
    bg = X_background.sample(
        n=min(200, len(X_background)),
        random_state=42
    )
    explainer = shap.TreeExplainer(model, bg)

    with open(SHAP_EXPLAINER_PATH, "wb") as f:
        pickle.dump(explainer, f)

    log.success(f"SHAP explainer saved → {format_path(SHAP_EXPLAINER_PATH)}")
    return explainer


def load_shap_explainer():
    with open(SHAP_EXPLAINER_PATH, "rb") as f:
        explainer = pickle.load(f)
    log.info(f"SHAP explainer loaded from {format_path(SHAP_EXPLAINER_PATH)}")
    return explainer


# ─────────────────────────────────────────────
# COMPUTE SHAP VALUES
# ─────────────────────────────────────────────
def compute_shap_values(
    explainer,
    X: pd.DataFrame
) -> np.ndarray:
    """
    Returns SHAP values array — shape (n_samples, n_features).
    Positive SHAP = pushes toward late prediction.
    Negative SHAP = pushes toward on-time prediction.
    """
    log.info(f"Computing SHAP values for {len(X):,} rows...")
    shap_values = explainer.shap_values(X)

    # XGBoost binary classification returns list [neg_class, pos_class]
    # We want the positive class (late payment) SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    log.info("SHAP values computed")
    return shap_values


# ─────────────────────────────────────────────
# PORTFOLIO-LEVEL PLOTS
# ─────────────────────────────────────────────
def plot_shap_summary(shap_values: np.ndarray, X: pd.DataFrame) -> None:
    """
    Summary beeswarm plot — shows impact of each feature across
    the entire portfolio. Most important feature at the top.
    Color = feature value (red=high, blue=low).
    """
    log.info("Generating SHAP summary plot...")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X,
        feature_names=CLASSIFIER_CFG.feature_cols,
        max_display=12,
        show=False,
        plot_size=None
    )
    plt.title("SHAP Feature Importance — Late Payment Driver Analysis", pad=20)
    plt.tight_layout()

    path = EXPORTS_DIR / "shap_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    log.success(f"SHAP summary plot saved → {format_path(path)}")
    plt.show()


def plot_shap_bar(shap_values: np.ndarray, X: pd.DataFrame) -> None:
    """
    Mean absolute SHAP bar chart — cleaner version of feature importance.
    Shows which features matter most across the whole portfolio.
    """
    mean_abs = pd.DataFrame({
        "Feature":    CLASSIFIER_CFG.feature_cols,
        "Mean |SHAP|": np.abs(shap_values).mean(axis=0)
    }).sort_values("Mean |SHAP|", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        mean_abs["Feature"],
        mean_abs["Mean |SHAP|"],
        color="#378ADD", edgecolor="white"
    )
    ax.set_xlabel("Mean |SHAP value|  (average impact on late probability)")
    ax.set_title("Feature Importance — SHAP Analysis")

    for bar, val in zip(bars, mean_abs["Mean |SHAP|"]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    path = EXPORTS_DIR / "shap_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    log.success(f"SHAP bar chart saved → {format_path(path)}")
    plt.show()


# ─────────────────────────────────────────────
# INVOICE-LEVEL WATERFALL
# ─────────────────────────────────────────────
def plot_waterfall(
    explainer,
    X_single: pd.Series,
    invoice_id: str,
    late_prob: float,
    save: bool = True
) -> None:
    """
    Waterfall chart for a single invoice.
    Shows exactly which features pushed the risk score up or down
    from the baseline expected value.

    Red bars = factors increasing late risk
    Blue bars = factors reducing late risk
    """
    log.info(f"Generating waterfall for invoice {invoice_id}...")

    shap_vals = explainer.shap_values(X_single.to_frame().T)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    expected_val = explainer.expected_value
    if isinstance(expected_val, (list, np.ndarray)):
        expected_val = expected_val[1]

    # Build waterfall data
    feature_shap = pd.DataFrame({
        "Feature": CLASSIFIER_CFG.feature_cols,
        "SHAP":    shap_vals[0],
        "Value":   X_single.values
    }).sort_values("SHAP", key=abs, ascending=False).head(8)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#E24B4A" if v > 0 else "#378ADD" for v in feature_shap["SHAP"]]

    bars = ax.barh(
        feature_shap["Feature"],
        feature_shap["SHAP"],
        color=colors, edgecolor="white", height=0.6
    )

    # Value annotations
    for bar, (_, row) in zip(bars, feature_shap.iterrows()):
        x = row["SHAP"]
        label = f"{row['Value']:.3f}"
        ax.text(
            x + (0.003 if x >= 0 else -0.003),
            bar.get_y() + bar.get_height() / 2,
            label, va="center",
            ha="left" if x >= 0 else "right",
            fontsize=8, color="gray"
        )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on late probability)")
    ax.set_title(
        f"Invoice {invoice_id} — Late Probability: {late_prob:.1%}\n"
        f"Baseline: {expected_val:.1%}  |  Red = increases risk  |  Blue = reduces risk"
    )
    plt.tight_layout()

    if save:
        path = EXPORTS_DIR / f"waterfall_{invoice_id}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        log.success(f"Waterfall saved → {format_path(path)}")
    plt.show()


# ─────────────────────────────────────────────
# AUDIT NARRATIVE GENERATOR
# ─────────────────────────────────────────────
def generate_narrative(
    row: pd.Series,
    shap_row: np.ndarray,
    late_prob: float,
    threshold: float
) -> str:
    """
    Generates a plain English audit narrative for a single invoice.

    Logic mirrors what an experienced AR auditor would note:
      - What is the overall risk level?
      - What is the primary driver of that risk?
      - Are there aggravating factors?
      - What action should be taken?

    This is your audit background encoded as automated logic.
    """
    feature_names = CLASSIFIER_CFG.feature_cols
    shap_dict     = dict(zip(feature_names, shap_row))

    # ── Risk level ───────────────────────────────
    risk_pct = late_prob * 100
    if late_prob >= 0.7:
        risk_level  = "HIGH RISK"
        risk_action = "Immediate escalation to senior collections"
    elif late_prob >= threshold:
        risk_level  = "ELEVATED RISK"
        risk_action = "Priority follow-up within 5 business days"
    else:
        risk_level  = "MONITORED"
        risk_action = "Standard collections workflow"

    lines = [
        f"[{risk_level}]  Late probability: {risk_pct:.0f}%",
        f"Recommended action: {risk_action}",
        "",
        "Key risk drivers:",
    ]

    # ── Driver narratives ────────────────────────
    # Sort features by absolute SHAP, take top 3 positive (risk-increasing)
    top_risk_features = sorted(
        [(k, v) for k, v in shap_dict.items() if v > 0],
        key=lambda x: x[1], reverse=True
    )[:3]

    driver_templates = {
        "CUST_AVG_DAYS_LATE": lambda v, rv:
            f"  • Customer payment history: averaging {row.get('CUST_AVG_DAYS_LATE', 0):.0f} days late "
            f"({'deteriorating' if row.get('CUST_LATE_TREND', 0) > 2 else 'stable'} trend)",

        "CUST_LATE_RATE": lambda v, rv:
            f"  • Customer late rate: {row.get('CUST_LATE_RATE', 0):.0%} of historical invoices paid late",

        "CUST_INVOICE_COUNT": lambda v, rv:
            f"  • Limited payment history: only {int(row.get('CUST_INVOICE_COUNT', 0))} prior invoices "
            f"({'new relationship — insufficient data' if row.get('CUST_INVOICE_COUNT', 0) < 5 else 'early stage'})",

        "LOG_AMOUNT": lambda v, rv:
            f"  • Invoice size: ${row.get('DMBTR', 0):,.0f} — "
            f"{'significantly above' if row.get('CUST_AMOUNT_CONCENTRATION', 1) > 1.5 else 'above'} "
            f"customer average (requires additional approval steps)",

        "IS_QUARTER_END": lambda v, rv:
            f"  • Quarter-end timing: invoice issued in {_month_name(int(row.get('INVOICE_MONTH', 1)))} "
            f"— customer likely prioritising own close activities",

        "CUST_AVG_AMOUNT": lambda v, rv:
            f"  • Invoice materially {'larger' if rv > 0 else 'smaller'} than this customer's "
            f"historical average — atypical transaction pattern",

        "PAYMENT_DAYS_NUM": lambda v, rv:
            f"  • Extended payment terms ({int(row.get('PAYMENT_DAYS_NUM', 30))} days) "
            f"increase collection window and exposure duration",

        "INDUSTRY_CODE": lambda v, rv:
            f"  • Industry payment culture: sector shows {'above' if rv > 0 else 'below'}-average "
            f"late payment rates in historical data",

        "REGION_CODE": lambda v, rv:
            f"  • Regional factor: region shows {'above' if rv > 0 else 'below'}-average "
            f"payment punctuality historically",

        "INVOICE_MONTH": lambda v, rv:
            f"  • Seasonal timing: month {int(row.get('INVOICE_MONTH', 1))} historically "
            f"shows {'lower' if rv > 0 else 'higher'} collection rates",
    }

    for feat, shap_val in top_risk_features:
        if feat in driver_templates:
            try:
                lines.append(driver_templates[feat](shap_val, shap_val))
            except Exception:
                lines.append(f"  • {feat}: SHAP impact {shap_val:+.4f}")

    # ── Mitigating factors ───────────────────────
    top_mitigants = sorted(
        [(k, v) for k, v in shap_dict.items() if v < 0],
        key=lambda x: x[1]
    )[:2]

    if top_mitigants:
        lines.append("")
        lines.append("Mitigating factors:")
        for feat, shap_val in top_mitigants:
            lines.append(f"  ✓ {feat}: reduces risk by {abs(shap_val):.4f}")

    # ── Suggested actions ────────────────────────
    lines.extend([
        "",
        "Suggested actions:",
        f"  1. {risk_action}",
    ])

    if row.get("CUST_LATE_RATE", 0) > 0.5:
        lines.append("  2. Review credit limit — customer late on >50% of invoices")
    if row.get("CUST_INVOICE_COUNT", 99) < 5:
        lines.append("  2. Request advance payment or shorter terms — limited history")
    if late_prob >= 0.7:
        lines.append("  3. Flag for CFO review if not resolved within 10 days")

    return "\n".join(lines)


def _month_name(month: int) -> str:
    months = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
    return months[max(0, min(month - 1, 11))]


# ─────────────────────────────────────────────
# BATCH NARRATIVE GENERATION
# ─────────────────────────────────────────────
def generate_risk_report(
    df: pd.DataFrame,
    top_n: int = None
) -> pd.DataFrame:
    """
    Generates SHAP-backed narratives for all high-risk invoices.
    Returns a dataframe with risk scores + narratives — feeds the dashboard.
    """
    if top_n is None:
        top_n = DASHBOARD_CFG.top_n_risks

    log.info("=" * 55)
    log.info("Generating risk report with SHAP narratives...")
    log.info("=" * 55)

    # Load model and explainer
    model, meta = load_model()
    threshold   = meta["threshold"]

    # Score all invoices
    scored = classify(model, df, threshold)

    # Focus on flagged invoices sorted by risk
    flagged = (
        scored[scored["LATE_PRED"] == 1]
        .sort_values("LATE_PROB", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    if len(flagged) == 0:
        log.warning("No invoices flagged at current threshold")
        return pd.DataFrame()

    log.info(f"Generating narratives for {len(flagged)} flagged invoices...")

    # Build SHAP explainer if not already saved
    if not SHAP_EXPLAINER_PATH.exists():
        X_all, _ = get_Xy(df)
        explainer = build_shap_explainer(model, X_all)
    else:
        explainer = load_shap_explainer()

    # Compute SHAP values for flagged invoices
    X_flagged = flagged[CLASSIFIER_CFG.feature_cols]
    shap_vals = compute_shap_values(explainer, X_flagged)

    # Generate narrative per invoice
    narratives = []
    for i, (_, row) in enumerate(flagged.iterrows()):
        narrative = generate_narrative(
            row, shap_vals[i], row["LATE_PROB"], threshold
        )
        narratives.append(narrative)
        log.info(f"[{i+1}/{len(flagged)}] Invoice {row.get('BELNR', i)} — {row['LATE_PROB']:.1%} risk")

    flagged["NARRATIVE"] = narratives
    flagged["SHAP_VALUES"] = [shap_vals[i].tolist() for i in range(len(flagged))]

    log.success(f"Risk report complete — {len(flagged)} invoices with narratives")
    return flagged


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 55)
    log.info("SHAP Explainability Pipeline")
    log.info("=" * 55)

    # Load data and model
    fs          = pd.read_csv(FEATURE_STORE_PATH, parse_dates=["BLDAT"])
    model, meta = load_model()
    threshold   = meta["threshold"]

    # Build explainer on full cleared dataset
    X_all, y_all = get_Xy(fs)
    explainer    = build_shap_explainer(model, X_all)
    shap_vals    = compute_shap_values(explainer, X_all)

    # Portfolio-level plots
    plot_shap_summary(shap_vals, X_all)
    plot_shap_bar(shap_vals, X_all)

    # Single invoice waterfall — highest risk invoice
    scored     = classify(model, fs, threshold)
    top_risk   = scored.nlargest(1, "LATE_PROB").iloc[0]
    X_top      = top_risk[CLASSIFIER_CFG.feature_cols]
    invoice_id = str(top_risk.get("BELNR", "SAMPLE"))
    late_prob  = top_risk["LATE_PROB"]

    plot_waterfall(explainer, X_top, invoice_id, late_prob)

    # Print sample narrative
    shap_top = compute_shap_values(explainer, X_top.to_frame().T)
    narrative = generate_narrative(top_risk, shap_top[0], late_prob, threshold)
    print("\n" + "=" * 55)
    print("SAMPLE AUDIT NARRATIVE")
    print("=" * 55)
    print(narrative)

    # Full risk report
    risk_report = generate_risk_report(fs, top_n=20)
    report_path = EXPORTS_DIR / "risk_report.csv"
    risk_report.drop(columns=["SHAP_VALUES"], errors="ignore").to_csv(
        report_path, index=False
    )
    log.success(f"Risk report saved → {format_path(report_path)}")
    log.info("\nNext step: make dashboard")
