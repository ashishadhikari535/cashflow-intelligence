"""
tests/test_classifier.py
─────────────────────────
Unit tests for the late payment classifier.
Tests: model loads, predicts correct shape, AUC above baseline,
       threshold is sensible, output columns are correct.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.config import TEST_PATH, CLASSIFIER_CFG, CLASSIFIER_PATH
from src.models.classifier import load_model, get_Xy, predict


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────
@pytest.fixture(scope="module")
def model_and_meta():
    return load_model()


@pytest.fixture(scope="module")
def test_data():
    return pd.read_csv(TEST_PATH)


# ─────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────
def test_model_artifact_exists():
    assert CLASSIFIER_PATH.exists(), "Model artifact not found — run make train first"


def test_model_loads(model_and_meta):
    model, meta = model_and_meta
    assert model is not None
    assert "threshold" in meta
    assert "feature_cols" in meta
    assert "cv_auc" in meta


def test_feature_cols_match(model_and_meta, test_data):
    _, meta = model_and_meta
    for col in meta["feature_cols"]:
        assert col in test_data.columns, f"Feature '{col}' missing from test data"


def test_predict_shape(model_and_meta, test_data):
    model, meta = model_and_meta
    result = predict(model, test_data, meta["threshold"])
    assert len(result) == len(test_data), "Prediction length mismatch"
    assert "LATE_PROB" in result.columns
    assert "LATE_PRED" in result.columns
    assert "RISK_LABEL" in result.columns


def test_probabilities_in_range(model_and_meta, test_data):
    model, meta = model_and_meta
    result = predict(model, test_data, meta["threshold"])
    assert result["LATE_PROB"].between(0, 1).all(), "Probabilities outside [0, 1]"


def test_predictions_are_binary(model_and_meta, test_data):
    model, meta = model_and_meta
    result = predict(model, test_data, meta["threshold"])
    assert set(result["LATE_PRED"].unique()).issubset({0, 1})


def test_auc_above_baseline(model_and_meta, test_data):
    from sklearn.metrics import roc_auc_score
    model, meta = model_and_meta
    X_test, y_test = get_Xy(test_data)
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    assert auc > 0.65, f"AUC {auc:.4f} is too low — check features or training"


def test_threshold_is_sensible(model_and_meta):
    _, meta = model_and_meta
    assert 0.2 <= meta["threshold"] <= 0.8, \
        f"Threshold {meta['threshold']} is outside sensible range [0.2, 0.8]"


def test_cv_auc_not_overfit(model_and_meta, test_data):
    from sklearn.metrics import roc_auc_score
    model, meta = model_and_meta
    X_test, y_test = get_Xy(test_data)
    probs = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, probs)
    gap = abs(meta["cv_auc"] - test_auc)
    assert gap < 0.08, \
        f"Train/test AUC gap {gap:.4f} suggests overfitting — review features"


def test_risk_labels_are_valid(model_and_meta, test_data):
    model, meta = model_and_meta
    result = predict(model, test_data, meta["threshold"])
    valid_labels = {"Low", "Medium", "High"}
    actual = set(result["RISK_LABEL"].dropna().unique())
    assert actual.issubset(valid_labels), f"Unexpected risk labels: {actual - valid_labels}"
