"""
tests/test_generator.py

Explainer tests kept in this file per request.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.config import CLASSIFIER_CFG, FEATURE_STORE_PATH
from src.models.classifier import load_model, predict
from src.models.explainer import (
    build_shap_explainer,
    compute_shap_values,
    generate_narrative,
    generate_risk_report,
)


@pytest.fixture(scope="module")
def model_and_meta():
    return load_model()


@pytest.fixture(scope="module")
def feature_store():
    return pd.read_csv(FEATURE_STORE_PATH)


def test_explainer_shap_shape(model_and_meta, feature_store):
    model, _ = model_and_meta
    X = feature_store[CLASSIFIER_CFG.feature_cols].head(120).copy()
    explainer = build_shap_explainer(model, X)
    shap_vals = compute_shap_values(explainer, X.head(15))
    assert isinstance(shap_vals, np.ndarray)
    assert shap_vals.shape == (15, len(CLASSIFIER_CFG.feature_cols))


def test_generate_narrative_returns_text(model_and_meta, feature_store):
    model, meta = model_and_meta
    scored = predict(model, feature_store.head(50), meta["threshold"])
    row = scored.iloc[0]
    shap_row = np.zeros(len(CLASSIFIER_CFG.feature_cols), dtype=float)
    text = generate_narrative(row, shap_row, float(row["LATE_PROB"]), float(meta["threshold"]))
    assert isinstance(text, str)
    assert "Late probability" in text
    assert "Suggested actions" in text


def test_generate_risk_report_columns(feature_store):
    report = generate_risk_report(feature_store.head(300), top_n=5)
    assert isinstance(report, pd.DataFrame)
    if len(report) == 0:
        pytest.skip("No flagged rows at current threshold for the sampled subset.")
    for col in ["LATE_PROB", "LATE_PRED", "NARRATIVE", "SHAP_VALUES"]:
        assert col in report.columns


def test_shap_values_serializable_in_report(feature_store):
    report = generate_risk_report(feature_store.head(300), top_n=3)
    if len(report) == 0:
        pytest.skip("No flagged rows at current threshold for the sampled subset.")
    assert report["SHAP_VALUES"].apply(lambda v: isinstance(v, list)).all()
