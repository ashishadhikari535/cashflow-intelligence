"""
src/models/evaluator.py
────────────────────────
Evaluation pipeline for both classifier and forecaster.

Produces:
  - Classification report (AUC-ROC, precision, recall, F1)
  - Confusion matrix
  - Calibration curve (are probabilities trustworthy?)
  - Pinball loss for forecaster
  - Clean summary table printed to console

Audit mindset: a model that says 70% probability should be right
~70% of the time. Calibration matters as much as accuracy here
because the CFO will make decisions based on these probabilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import TRAIN_PATH, TEST_PATH, CLASSIFIER_CFG, EXPORTS_DIR
from src.utils.logger import format_path, get_logger
from src.models.classifier import load_model, get_Xy, predict

log = get_logger(__name__)


# ─────────────────────────────────────────────
# CLASSIFICATION METRICS
# ─────────────────────────────────────────────
def evaluate_classifier(save_plots: bool = True) -> dict:
    log.info("=" * 55)
    log.info("Evaluating classifier on held-out test set...")
    log.info("=" * 55)

    # Load model and test data
    model, meta   = load_model()
    test          = pd.read_csv(TEST_PATH)
    X_test, y_test = get_Xy(test)
    threshold      = meta["threshold"]

    # Predictions
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    # ── Core metrics ──────────────────────────
    auc_roc   = roc_auc_score(y_test, probs)
    auc_pr    = average_precision_score(y_test, probs)
    brier     = brier_score_loss(y_test, probs)
    report    = classification_report(y_test, preds, target_names=["On time", "Late"], output_dict=True)
    cm        = confusion_matrix(y_test, preds)

    # ── Print summary ─────────────────────────
    log.info("\n" + "─" * 45)
    log.info(f"  AUC-ROC              : {auc_roc:.4f}")
    log.info(f"  AUC-PR               : {auc_pr:.4f}")
    log.info(f"  Brier score          : {brier:.4f}  (lower = better calibrated)")
    log.info(f"  Threshold used       : {threshold:.2f}")
    log.info(f"  CV AUC (train)       : {meta['cv_auc']:.4f}")
    log.info("─" * 45)
    log.info(f"  Late — Precision     : {report['Late']['precision']:.4f}")
    log.info(f"  Late — Recall        : {report['Late']['recall']:.4f}")
    log.info(f"  Late — F1            : {report['Late']['f1-score']:.4f}")
    log.info(f"  Late — Support       : {int(report['Late']['support']):,}")
    log.info("─" * 45)

    tn, fp, fn, tp = cm.ravel()
    log.info(f"  True positives       : {tp:,}   (correctly flagged late)")
    log.info(f"  False positives      : {fp:,}   (wrongly flagged — collection noise)")
    log.info(f"  False negatives      : {fn:,}   (missed late payers — cash surprise)")
    log.info(f"  True negatives       : {tn:,}   (correctly cleared)")
    log.info("─" * 45)

    # ── Plots ─────────────────────────────────
    if save_plots:
        _plot_evaluation(y_test, probs, preds, cm, threshold)

    return {
        "auc_roc":   auc_roc,
        "auc_pr":    auc_pr,
        "brier":     brier,
        "threshold": threshold,
        "cv_auc":    meta["cv_auc"],
        "report":    report,
        "confusion": cm.tolist(),
    }


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────
def _plot_evaluation(y_test, probs, preds, cm, threshold) -> None:
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1. ROC Curve
    ax1 = fig.add_subplot(gs[0, 0])
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    ax1.plot(fpr, tpr, color="#378ADD", lw=2, label=f"AUC = {auc:.4f}")
    ax1.plot([0,1],[0,1], "k--", alpha=0.4, label="Random")
    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.set_title("ROC curve")
    ax1.legend()

    # 2. Precision-Recall Curve
    ax2 = fig.add_subplot(gs[0, 1])
    prec, rec, thresh = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)
    ax2.plot(rec, prec, color="#7F77DD", lw=2, label=f"AP = {ap:.4f}")
    ax2.axhline(y_test.mean(), color="gray", linestyle="--", alpha=0.5, label="Baseline")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall curve")
    ax2.legend()

    # 3. Confusion Matrix
    ax3 = fig.add_subplot(gs[0, 2])
    im = ax3.imshow(cm, interpolation="nearest", cmap="Blues")
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(["On time", "Late"])
    ax3.set_yticklabels(["On time", "Late"])
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    ax3.set_title(f"Confusion matrix (threshold={threshold:.2f})")
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                     color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=13)

    # 4. Calibration Curve
    ax4 = fig.add_subplot(gs[1, 0])
    frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
    ax4.plot(mean_pred, frac_pos, "s-", color="#1D9E75", label="Model")
    ax4.plot([0,1],[0,1], "k--", alpha=0.4, label="Perfect calibration")
    ax4.set_xlabel("Mean predicted probability")
    ax4.set_ylabel("Fraction of positives")
    ax4.set_title("Calibration curve")
    ax4.legend()

    # 5. Probability Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(probs[y_test==0], bins=40, alpha=0.6, color="#1D9E75",
             label="On time", density=True)
    ax5.hist(probs[y_test==1], bins=40, alpha=0.6, color="#E24B4A",
             label="Late", density=True)
    ax5.axvline(threshold, color="black", linestyle="--", label=f"Threshold={threshold:.2f}")
    ax5.set_xlabel("Predicted probability")
    ax5.set_ylabel("Density")
    ax5.set_title("Score distribution by class")
    ax5.legend()

    # 6. Threshold sensitivity
    ax6 = fig.add_subplot(gs[1, 2])
    from sklearn.metrics import f1_score, precision_score, recall_score
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1s   = [f1_score(y_test, (probs>=t).astype(int), zero_division=0) for t in thresholds]
    precs = [precision_score(y_test, (probs>=t).astype(int), zero_division=0) for t in thresholds]
    recs  = [recall_score(y_test, (probs>=t).astype(int), zero_division=0) for t in thresholds]
    ax6.plot(thresholds, f1s,   color="#7F77DD", label="F1")
    ax6.plot(thresholds, precs, color="#378ADD", label="Precision", alpha=0.7)
    ax6.plot(thresholds, recs,  color="#E24B4A", label="Recall", alpha=0.7)
    ax6.axvline(threshold, color="black", linestyle="--", alpha=0.6, label=f"Chosen={threshold:.2f}")
    ax6.set_xlabel("Threshold")
    ax6.set_ylabel("Score")
    ax6.set_title("Threshold sensitivity")
    ax6.legend(fontsize=8)

    plt.suptitle("Classifier Evaluation — Late Payment Prediction", fontsize=14, y=1.01)

    plot_path = EXPORTS_DIR / "classifier_evaluation.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    log.success(f"Evaluation plots saved → {format_path(plot_path)}")
    plt.show()


# ─────────────────────────────────────────────
# RISK SEGMENT ANALYSIS
# ─────────────────────────────────────────────
def risk_segment_analysis() -> pd.DataFrame:
    """
    Breaks down model performance by industry and region.
    Important for audit — does the model work equally well across segments?
    Bias in financial models is a real risk.
    """
    log.info("\nRisk segment analysis...")

    model, meta = load_model()
    test        = pd.read_csv(TEST_PATH)

    result = predict(model, test, meta["threshold"])

    segments = []
    for col in ["INDUSTRY", "REGION"]:
        if col not in result.columns:
            continue
        for val, grp in result.groupby(col):
            if len(grp) < 10:
                continue
            auc = roc_auc_score(grp["LATE_FLAG"], grp["LATE_PROB"]) if grp["LATE_FLAG"].nunique() > 1 else None
            segments.append({
                "Segment":   col,
                "Value":     val,
                "N":         len(grp),
                "Late %":    f"{grp['LATE_FLAG'].mean():.1%}",
                "AUC-ROC":   f"{auc:.4f}" if auc else "N/A",
                "Flagged %": f"{grp['LATE_PRED'].mean():.1%}",
            })

    seg_df = pd.DataFrame(segments)
    log.info(f"\n{seg_df.to_string(index=False)}")
    return seg_df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    metrics = evaluate_classifier(save_plots=True)
    seg_df  = risk_segment_analysis()

    log.info("\n=== Evaluation Complete ===")
    log.info(f"AUC-ROC : {metrics['auc_roc']:.4f}")
    log.info(f"AUC-PR  : {metrics['auc_pr']:.4f}")
    log.info(f"Brier   : {metrics['brier']:.4f}")
    log.info("\nNext step: python -m src.models.forecaster")
