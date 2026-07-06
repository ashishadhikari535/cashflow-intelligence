"""
src/models/classifier.py
─────────────────────────
XGBoost late payment classifier.

Predicts the probability that a given invoice will be paid late.
This is the core model — every other output (cash forecast, risk flags,
SHAP narratives) builds on top of these probability scores.

Design decisions:
  - Time-aware train/test split (already done in features.py)
  - Cross-validation on train set only — no test leakage
  - Threshold optimization for business use case (precision vs recall tradeoff)
  - Model serialized as JSON — human readable, version controllable
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import (
    TRAIN_PATH, TEST_PATH,
    CLASSIFIER_PATH, CLASSIFIER_CFG
)
from src.utils.logger import format_path, get_logger

log = get_logger(__name__)
CALIBRATOR_PATH = CLASSIFIER_PATH.parent / "classifier_calibrator.pkl"


def _predict_positive_proba(model, X: pd.DataFrame) -> np.ndarray:
    """
    Returns positive-class probabilities.
    Uses calibration layer if attached to the model.
    """
    calibrator = getattr(model, "_calibrator", None)
    predictor = calibrator if calibrator is not None else model
    return predictor.predict_proba(X)[:, 1]


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("Loading train/test splits...")
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    log.info(f"Train: {len(train):,}  |  Test: {len(test):,}")
    return train, test


def get_Xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[CLASSIFIER_CFG.feature_cols].copy()
    y = df[CLASSIFIER_CFG.target_col].astype(int)
    return X, y


# ─────────────────────────────────────────────
# CROSS VALIDATION
# ─────────────────────────────────────────────
def cross_validate(X_train: pd.DataFrame, y_train: pd.Series) -> float:
    """
    Stratified k-fold CV on train set.
    Stratified = preserves class imbalance ratio in each fold.
    Returns mean AUC-ROC across folds.
    """
    log.info(f"Running {CLASSIFIER_CFG.cv_folds}-fold stratified cross-validation...")

    model = xgb.XGBClassifier(
        **{k: v for k, v in CLASSIFIER_CFG.xgb_params.items()
           if k not in ["early_stopping_rounds", "eval_metric"]},
        use_label_encoder=False,
        eval_metric="auc",
    )

    cv = StratifiedKFold(
        n_splits=CLASSIFIER_CFG.cv_folds,
        shuffle=True,
        random_state=CLASSIFIER_CFG.random_seed
    )

    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

    log.info(f"CV AUC-ROC: {scores.mean():.4f} ± {scores.std():.4f}")
    log.info(f"Per-fold:   {[round(s, 4) for s in scores]}")

    return float(scores.mean())


# ─────────────────────────────────────────────
# TRAIN FINAL MODEL
# ─────────────────────────────────────────────
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> xgb.XGBClassifier:
    """
    Trains final XGBoost model with early stopping on validation set.
    Uses scale_pos_weight to handle class imbalance — avoids oversampling
    which can leak information in time-series financial data.
    """
    log.info("Training XGBoost classifier...")

    # Compute scale_pos_weight from training data
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw   = round(n_neg / n_pos, 2)
    log.info(f"Class balance — negative: {n_neg:,}  |  positive: {n_pos:,}  |  scale_pos_weight: {spw}")

    params = {**CLASSIFIER_CFG.xgb_params, "scale_pos_weight": spw}

    model = xgb.XGBClassifier(
        **{k: v for k, v in params.items() if k not in ["eval_metric"]},
        use_label_encoder=False,
        eval_metric="auc",
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    best_iter = model.best_iteration
    log.success(f"Training complete — best iteration: {best_iter}")
    return model


# ─────────────────────────────────────────────
# THRESHOLD OPTIMIZATION
# ─────────────────────────────────────────────
def calibrate_model(
    model: xgb.XGBClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series
):
    """
    Calibrates probability estimates on the validation split.
    """
    calibration_cfg = CLASSIFIER_CFG.calibration or {}
    if not calibration_cfg.get("enabled", True):
        log.info("Calibration disabled in config")
        return None

    method = calibration_cfg.get("method", "sigmoid")
    log.info(f"Calibrating probabilities (method={method})...")

    calibrator = CalibratedClassifierCV(model, method=method, cv="prefit")
    calibrator.fit(X_val, y_val)
    log.success("Calibration complete")
    return calibrator


def optimize_threshold(
    predictor,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> float:
    """
    Finds the probability threshold that maximizes F1 on validation set.

    Business context:
      - False negative (miss a late payer) = cash flow surprise — costly
      - False positive (flag a good payer) = unnecessary collection call — annoying
      - F1 balances both — good default for AR risk flagging
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    probs = predictor.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.2, 0.8, 0.01)
    base_rate = float(y_val.mean())

    rows = []
    for t in thresholds:
        pred = (probs >= t).astype(int)
        rows.append(
            {
                "threshold": float(t),
                "f1": float(f1_score(y_val, pred, zero_division=0)),
                "precision": float(precision_score(y_val, pred, zero_division=0)),
                "recall": float(recall_score(y_val, pred, zero_division=0)),
                "flag_rate": float(pred.mean()),
            }
        )

    metrics = pd.DataFrame(rows)

    # Guardrail: avoid operationally useless thresholds that flag almost everything.
    max_flag_rate = min(0.85, base_rate + 0.10)
    candidate = metrics[metrics["flag_rate"] <= max_flag_rate].copy()
    if candidate.empty:
        candidate = metrics[metrics["flag_rate"] <= 0.95].copy()
    if candidate.empty:
        candidate = metrics.copy()

    best_idx = candidate["f1"].idxmax()
    best_row = candidate.loc[best_idx]

    log.info(
        "Optimal threshold policy "
        f"(base_rate={base_rate:.3f}, max_flag_rate={max_flag_rate:.3f}) "
        f"-> t={best_row['threshold']:.2f} | "
        f"F1={best_row['f1']:.4f}, "
        f"precision={best_row['precision']:.4f}, "
        f"recall={best_row['recall']:.4f}, "
        f"flag_rate={best_row['flag_rate']:.4f}"
    )
    return float(best_row["threshold"])


# ─────────────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────────────
def save_model(
    model: xgb.XGBClassifier,
    threshold: float,
    cv_auc: float,
    calibrator=None
) -> None:
    """
    Save model artifact + metadata.
    JSON format — readable and diff-able in git.
    """
    model.save_model(str(CLASSIFIER_PATH))

    # Save threshold and metadata separately
    meta = {
        "threshold":        threshold,
        "cv_auc":           cv_auc,
        "feature_cols":     CLASSIFIER_CFG.feature_cols,
        "target_col":       CLASSIFIER_CFG.target_col,
        "xgb_params":       CLASSIFIER_CFG.xgb_params,
        "best_iteration":   int(model.best_iteration),
        "calibration":      CLASSIFIER_CFG.calibration,
        "is_calibrated":    calibrator is not None,
    }
    meta_path = CLASSIFIER_PATH.parent / "classifier_meta.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    if calibrator is not None:
        with open(CALIBRATOR_PATH, "wb") as f:
            pickle.dump(calibrator, f)

    log.success(f"Model saved  → {format_path(CLASSIFIER_PATH)}")
    log.success(f"Meta saved   → {format_path(meta_path)}")
    if calibrator is not None:
        log.success(f"Calibrator saved → {format_path(CALIBRATOR_PATH)}")


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
def load_model() -> tuple[xgb.XGBClassifier, dict]:
    """
    Load trained model and metadata.
    Used by dashboard, explainer, and evaluator.
    """
    model = xgb.XGBClassifier()
    model.load_model(str(CLASSIFIER_PATH))

    meta_path = CLASSIFIER_PATH.parent / "classifier_meta.pkl"
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    if CALIBRATOR_PATH.exists():
        try:
            with open(CALIBRATOR_PATH, "rb") as f:
                calibrator = pickle.load(f)
            model._calibrator = calibrator
            model.predict_proba = calibrator.predict_proba
            log.info(f"Calibrator loaded from {format_path(CALIBRATOR_PATH)}")
        except Exception as exc:
            log.warning(f"Failed to load calibrator: {exc}")

    log.info(f"Model loaded from {format_path(CLASSIFIER_PATH)}")
    return model, meta


# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
def predict(
    model: xgb.XGBClassifier,
    df: pd.DataFrame,
    threshold: float
) -> pd.DataFrame:
    """
    Returns input df with two new columns:
      LATE_PROB  — probability of late payment (0-1)
      LATE_PRED  — binary prediction at given threshold
      RISK_LABEL — human readable risk label for dashboard
    """
    X     = df[CLASSIFIER_CFG.feature_cols].copy()
    probs = _predict_positive_proba(model, X)
    preds = (probs >= threshold).astype(int)

    result = df.copy()
    result["LATE_PROB"]  = probs.round(4)
    result["LATE_PRED"]  = preds
    result["RISK_LABEL"] = pd.cut(
        probs,
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"]
    )
    return result


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 55)
    log.info("XGBoost Late Payment Classifier — Training Pipeline")
    log.info("=" * 55)

    # Load
    train, test = load_splits()
    X_train, y_train = get_Xy(train)
    X_test,  y_test  = get_Xy(test)

    # Use last 20% of train as validation for early stopping
    split   = int(len(X_train) * 0.8)
    X_tr, X_val = X_train.iloc[:split], X_train.iloc[split:]
    y_tr, y_val = y_train.iloc[:split], y_train.iloc[split:]

    # Cross validate
    cv_auc = cross_validate(X_tr, y_tr)

    # Train final model
    model = train_model(X_tr, y_tr, X_val, y_val)

    # Calibrate probabilities on validation split
    calibrator = calibrate_model(model, X_val, y_val)

    # Optimize threshold on calibrated probabilities (if available)
    threshold_model = calibrator if calibrator is not None else model
    threshold = optimize_threshold(threshold_model, X_val, y_val)

    # Save
    save_model(model, threshold, cv_auc, calibrator=calibrator)

    log.info("\nNext step: python -m src.models.evaluator")
