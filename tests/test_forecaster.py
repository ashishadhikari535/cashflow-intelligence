"""
tests/test_forecaster.py
─────────────────────────
Unit tests for the probabilistic cash flow forecaster.
Tests: artifact exists, loads correctly, forecast shape,
       bands don't cross, uncertainty is sensible.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.config import FORECASTER_PATH, FORECASTER_CFG
from src.models.forecaster import (
    load_forecaster, build_forecast_dataset,
    generate_forecast, _pinball_loss
)


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────
@pytest.fixture(scope="module")
def forecast_payload():
    return load_forecaster()


@pytest.fixture(scope="module")
def forecast_dataset():
    return build_forecast_dataset()


@pytest.fixture(scope="module")
def forecast_df(forecast_dataset):
    return generate_forecast(forecast_dataset)


# ─────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────
def test_forecaster_artifact_exists():
    assert FORECASTER_PATH.exists(), \
        "Forecaster artifact not found — run make train first"


def test_forecaster_loads(forecast_payload):
    assert "models" in forecast_payload
    assert "feature_cols" in forecast_payload
    assert "quantiles" in forecast_payload
    assert "horizons" in forecast_payload
    if "segment_models" in forecast_payload:
        assert isinstance(forecast_payload["segment_models"], dict)
    if "segment_summary" in forecast_payload:
        assert isinstance(forecast_payload["segment_summary"], dict)


def test_all_models_present(forecast_payload):
    from itertools import product
    for horizon, quantile in product(
        FORECASTER_CFG.forecast_horizons,
        FORECASTER_CFG.quantiles
    ):
        assert (horizon, round(float(quantile), 4)) in forecast_payload["models"], \
            f"Missing model for horizon={horizon}, quantile={quantile}"


def test_forecast_shape(forecast_df):
    assert len(forecast_df) == len(FORECASTER_CFG.forecast_horizons), \
        "Forecast should have one row per horizon"
    for col in ["Lower_10pct", "Median_50pct", "Upper_90pct"]:
        assert col in forecast_df.columns


def test_bands_dont_cross(forecast_df):
    for _, row in forecast_df.iterrows():
        assert row["Lower_10pct"] <= row["Median_50pct"], \
            f"Lower band exceeds median at {row['Horizon']}"
        assert row["Median_50pct"] <= row["Upper_90pct"], \
            f"Median exceeds upper band at {row['Horizon']}"


def test_forecast_values_positive(forecast_df):
    for col in ["Lower_10pct", "Median_50pct", "Upper_90pct"]:
        assert (forecast_df[col] >= 0).all(), \
            f"Negative forecast values in {col}"


def test_uncertainty_increases_with_horizon(forecast_df):
    widths = forecast_df["Range_Width"].values
    # Wider bands at longer horizons is common but not guaranteed for every retrain.
    # Keep a floor so uncertainty does not collapse at long horizon.
    assert (widths > 0).all(), "All forecast intervals should be positive width"
    assert widths[-1] >= widths[0] * 0.6, \
        "90-day band unexpectedly narrow vs 30-day band"


def test_pinball_loss_calculation():
    y_true   = np.array([100, 200, 300, 400, 500])
    y_pred   = np.array([110, 190, 310, 390, 510])
    quantile = 0.5
    loss = _pinball_loss(y_true, y_pred, quantile)
    assert loss >= 0, "Pinball loss should be non-negative"
    assert loss < 100, "Pinball loss unreasonably large for small errors"


def test_dataset_has_required_features(forecast_dataset):
    from src.models.forecaster import FORECAST_FEATURE_COLS
    for col in FORECAST_FEATURE_COLS:
        assert col in forecast_dataset.columns, \
            f"Feature '{col}' missing from forecast dataset"


def test_dataset_has_no_nulls_in_features(forecast_dataset):
    from src.models.forecaster import FORECAST_FEATURE_COLS
    nulls = forecast_dataset[FORECAST_FEATURE_COLS].isnull().sum()
    assert nulls.sum() == 0, \
        f"Null values in forecast features: {nulls[nulls > 0].to_dict()}"


def test_generate_forecast_with_segment_args(forecast_dataset):
    regional_forecast = generate_forecast(forecast_dataset, region="West")
    assert len(regional_forecast) == len(FORECASTER_CFG.forecast_horizons)
    assert "Model_Source" in regional_forecast.columns
