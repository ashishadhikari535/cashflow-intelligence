"""
src/models/forecaster.py
Hybrid probabilistic cash flow forecaster with segment fallback.

Design:
- Global quantile forecaster trained on pooled segment-month observations.
- Optional segment-specific forecasters (for mature segments only).
- Inference fallback: segment model when eligible, otherwise global model.
"""

from __future__ import annotations

import pickle
import sys
from itertools import product
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.models.classifier import load_model, predict as classify
from src.utils.config import (
    CUSTOMER_MASTER_PATH,
    EXPORTS_DIR,
    FEATURE_STORE_PATH,
    FORECASTER_CFG,
    FORECASTER_PATH,
    GL_CASH_PATH,
    INVOICE_HISTORY_PATH,
)
from src.utils.logger import format_path, get_logger

log = get_logger(__name__)


FORECAST_FEATURE_COLS = [
    "MONTH",
    "QUARTER",
    "IS_Q_END",
    "MONTH_SIN",
    "MONTH_COS",
    "COLLECTIONS_LAG1",
    "COLLECTIONS_LAG2",
    "COLLECTIONS_LAG3",
    "LATE_RATE_LAG1",
    "LATE_RATE_LAG2",
    "AVG_LATE_PROB",
    "HIGH_RISK_SHARE",
    "N_INVOICES_CLEARED",
    "COLLECTIONS_ROLL3_MEAN",
    "COLLECTIONS_ROLL3_STD",
    "LOG_OPENING_CASH",
    "LOG_OPEN_AR",
]

INDUSTRY_MAP = {
    "EN": "Energy",
    "MF": "Manufacturing",
    "RT": "Retail",
    "HC": "Healthcare",
    "CN": "Construction",
    "LG": "Logistics",
}
REGION_MAP = {
    "TX": "South",
    "NY": "Northeast",
    "IL": "Midwest",
    "CA": "West",
    "INTL": "International",
}


def _normalize_customer_master(customers: pd.DataFrame) -> pd.DataFrame:
    out = customers.copy()
    if "REGIO" not in out.columns and "REGION" in out.columns:
        inv_region = {v: k for k, v in REGION_MAP.items()}
        out["REGIO"] = out["REGION"].map(inv_region).fillna("INTL")
    if "BRSCH" not in out.columns and "INDUSTRY" in out.columns:
        inv_industry = {v: k for k, v in INDUSTRY_MAP.items()}
        out["BRSCH"] = out["INDUSTRY"].map(inv_industry).fillna("MF")

    out["REGION"] = out["REGIO"].astype(str).map(REGION_MAP).fillna("International")
    out["INDUSTRY"] = out["BRSCH"].astype(str).map(INDUSTRY_MAP).fillna("Other")
    return out[["KUNNR", "INDUSTRY", "REGION"]].drop_duplicates(subset=["KUNNR"])


def _normalize_gl_cash(gl: pd.DataFrame) -> pd.DataFrame:
    out = gl.copy()
    out["HSL"] = pd.to_numeric(out["HSL"], errors="coerce")
    if "FISCPER" not in out.columns:
        if "GJAHR" in out.columns and "POPER" in out.columns:
            year = out["GJAHR"].astype(str).str.slice(0, 4)
            period = pd.to_numeric(out["POPER"], errors="coerce").fillna(1).astype(int).clip(1, 12)
            out["FISCPER"] = year + period.astype(str).str.zfill(2)
        else:
            raise KeyError("GL data must include FISCPER or (GJAHR, POPER).")
    return out


def _derive_invoice_risk_tier(inv: pd.DataFrame) -> pd.Series:
    cleared = inv[inv["STATUS"] == "CLEARED"].copy()
    if cleared.empty:
        return pd.Series("Medium", index=inv.index)

    rates = (
        cleared.groupby("KUNNR")["LATE_FLAG"]
        .mean()
        .fillna(0.0)
    )
    tiers = pd.cut(
        rates,
        bins=[-np.inf, 0.20, 0.50, np.inf],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).astype(str)
    return inv["KUNNR"].map(tiers).fillna("Medium")


def _normalize_invoice_history(inv: pd.DataFrame, customer_dim: pd.DataFrame) -> pd.DataFrame:
    out = inv.copy()
    for col in ["BLDAT", "FAEDT", "AUGDT"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    if "STATUS" not in out.columns:
        out["STATUS"] = np.where(out["AUGDT"].isna(), "OPEN", "CLEARED")
    if "DAYS_LATE" not in out.columns:
        days_late = (out["AUGDT"] - out["FAEDT"]).dt.days
        out["DAYS_LATE"] = np.where(out["AUGDT"].isna(), np.nan, days_late)
    if "LATE_FLAG" not in out.columns:
        out["LATE_FLAG"] = np.where(out["AUGDT"].isna(), np.nan, (out["DAYS_LATE"] > 0).astype(float))

    cust_idx = customer_dim.set_index("KUNNR")
    if "INDUSTRY" not in out.columns:
        out["INDUSTRY"] = out["KUNNR"].map(cust_idx["INDUSTRY"])
    else:
        out["INDUSTRY"] = out["INDUSTRY"].fillna(out["KUNNR"].map(cust_idx["INDUSTRY"]))

    if "REGION" not in out.columns:
        out["REGION"] = out["KUNNR"].map(cust_idx["REGION"])
    else:
        out["REGION"] = out["REGION"].fillna(out["KUNNR"].map(cust_idx["REGION"]))

    if "RISK_TIER" not in out.columns:
        out["RISK_TIER"] = _derive_invoice_risk_tier(out)

    return out


def _normalize_selector(value: str | None) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "all":
        return None
    return s


def _build_horizon_map(horizons: list[int]) -> dict[int, int]:
    return {h: max(1, int(round(h / 30.0))) for h in horizons}


def _load_forecast_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fs = pd.read_csv(FEATURE_STORE_PATH, parse_dates=["BLDAT", "FAEDT"])
    customers = pd.read_csv(CUSTOMER_MASTER_PATH)
    customer_dim = _normalize_customer_master(customers)

    inv = pd.read_csv(INVOICE_HISTORY_PATH)
    inv = _normalize_invoice_history(inv, customer_dim)

    gl = pd.read_csv(GL_CASH_PATH)
    gl = _normalize_gl_cash(gl)

    clf_model, clf_meta = load_model()
    scored = classify(clf_model, fs, clf_meta["threshold"])
    return inv, gl, scored


def _apply_segment_filters(
    df: pd.DataFrame,
    region: str | None = None,
    industry: str | None = None,
) -> pd.DataFrame:
    filtered = df.copy()
    region_norm = _normalize_selector(region)
    industry_norm = _normalize_selector(industry)

    if region_norm is not None and "REGION" in filtered.columns:
        filtered = filtered[filtered["REGION"].astype(str).str.strip() == region_norm]
    if industry_norm is not None and "INDUSTRY" in filtered.columns:
        filtered = filtered[filtered["INDUSTRY"].astype(str).str.strip() == industry_norm]
    return filtered


def _safe_monthly_open_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "COLLECT_MONTH": pd.Series(dtype="string"),
            "OPEN_AR_TOTAL": pd.Series(dtype="float64"),
            "N_OPEN_INVOICES": pd.Series(dtype="float64"),
        }
    )


def _safe_monthly_risk_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "COLLECT_MONTH": pd.Series(dtype="string"),
            "AVG_LATE_PROB": pd.Series(dtype="float64"),
        }
    )


def _build_monthly_dataset_from_sources(
    inv: pd.DataFrame,
    gl: pd.DataFrame,
    scored: pd.DataFrame,
    *,
    region: str | None = None,
    industry: str | None = None,
    min_rows: int = 12,
) -> pd.DataFrame:
    inv_filtered = _apply_segment_filters(inv, region=region, industry=industry)
    scored_filtered = _apply_segment_filters(scored, region=region, industry=industry)

    cleared = inv_filtered[inv_filtered["STATUS"] == "CLEARED"].copy()
    if cleared.empty:
        return pd.DataFrame()

    cleared["COLLECT_MONTH"] = pd.to_datetime(cleared["AUGDT"]).dt.to_period("M").astype(str)
    monthly_collections = (
        cleared.groupby("COLLECT_MONTH")
        .agg(
            ACTUAL_COLLECTIONS=("DMBTR", "sum"),
            N_INVOICES_CLEARED=("BELNR", "count"),
            AVG_DAYS_LATE=("DAYS_LATE", "mean"),
            LATE_RATE=("LATE_FLAG", "mean"),
            HIGH_RISK_SHARE=("RISK_TIER", lambda x: (x == "High").mean()),
        )
        .reset_index()
    )
    if monthly_collections.empty:
        return pd.DataFrame()

    open_inv = inv_filtered[inv_filtered["STATUS"] == "OPEN"].copy()
    if open_inv.empty:
        monthly_open = _safe_monthly_open_frame()
    else:
        open_inv["INVOICE_MONTH"] = pd.to_datetime(open_inv["BLDAT"]).dt.to_period("M").astype(str)
        monthly_open = (
            open_inv.groupby("INVOICE_MONTH")
            .agg(OPEN_AR_TOTAL=("DMBTR", "sum"), N_OPEN_INVOICES=("BELNR", "count"))
            .reset_index()
            .rename(columns={"INVOICE_MONTH": "COLLECT_MONTH"})
        )

    gl_agg = (
        gl.groupby("FISCPER")["HSL"]
        .sum()
        .reset_index()
        .rename(columns={"HSL": "OPENING_CASH", "FISCPER": "COLLECT_MONTH"})
    )
    gl_agg["COLLECT_MONTH"] = (
        pd.to_datetime(gl_agg["COLLECT_MONTH"], format="%Y%m")
        .dt.to_period("M")
        .astype(str)
    )

    if scored_filtered.empty:
        monthly_risk = _safe_monthly_risk_frame()
    else:
        scored_filtered = scored_filtered.copy()
        scored_filtered["INVOICE_MONTH"] = pd.to_datetime(scored_filtered["BLDAT"]).dt.to_period("M").astype(str)
        monthly_risk = (
            scored_filtered.groupby("INVOICE_MONTH")
            .agg(AVG_LATE_PROB=("LATE_PROB", "mean"))
            .reset_index()
            .rename(columns={"INVOICE_MONTH": "COLLECT_MONTH"})
        )

    df = (
        monthly_collections.merge(monthly_open, on="COLLECT_MONTH", how="left")
        .merge(gl_agg, on="COLLECT_MONTH", how="left")
        .merge(monthly_risk, on="COLLECT_MONTH", how="left")
        .sort_values("COLLECT_MONTH")
        .reset_index(drop=True)
    )

    df["OPEN_AR_TOTAL"] = df["OPEN_AR_TOTAL"].fillna(0.0)
    df["N_OPEN_INVOICES"] = df["N_OPEN_INVOICES"].fillna(0.0)
    df["OPENING_CASH"] = df["OPENING_CASH"].ffill().fillna(0.0)
    df["AVG_LATE_PROB"] = df["AVG_LATE_PROB"].fillna(df["LATE_RATE"])

    df["PERIOD_DT"] = pd.to_datetime(df["COLLECT_MONTH"])
    df["MONTH"] = df["PERIOD_DT"].dt.month
    df["QUARTER"] = df["PERIOD_DT"].dt.quarter
    df["IS_Q_END"] = df["MONTH"].isin([3, 6, 9, 12]).astype(int)
    df["MONTH_SIN"] = np.sin(2 * np.pi * df["MONTH"] / 12)
    df["MONTH_COS"] = np.cos(2 * np.pi * df["MONTH"] / 12)

    for lag in [1, 2, 3]:
        df[f"COLLECTIONS_LAG{lag}"] = df["ACTUAL_COLLECTIONS"].shift(lag)
        df[f"LATE_RATE_LAG{lag}"] = df["LATE_RATE"].shift(lag)

    df["COLLECTIONS_ROLL3_MEAN"] = df["ACTUAL_COLLECTIONS"].shift(1).rolling(3).mean()
    df["COLLECTIONS_ROLL3_STD"] = df["ACTUAL_COLLECTIONS"].shift(1).rolling(3).std()
    df["LOG_ACTUAL_COLLECTIONS"] = np.log1p(df["ACTUAL_COLLECTIONS"])
    df["LOG_OPENING_CASH"] = np.log1p(df["OPENING_CASH"].fillna(0))
    df["LOG_OPEN_AR"] = np.log1p(df["OPEN_AR_TOTAL"].fillna(0))

    df = df.dropna().reset_index(drop=True)

    if min_rows > 0 and len(df) < min_rows:
        raise ValueError(f"Forecast dataset too small after feature engineering ({len(df)} rows).")
    return df


def build_forecast_dataset(
    region: str | None = None,
    industry: str | None = None,
) -> pd.DataFrame:
    log.info(
        "Building forecast dataset"
        f" [region={_normalize_selector(region) or 'All'},"
        f" industry={_normalize_selector(industry) or 'All'}]..."
    )
    inv, gl, scored = _load_forecast_sources()
    df = _build_monthly_dataset_from_sources(
        inv,
        gl,
        scored,
        region=region,
        industry=industry,
        min_rows=FORECASTER_CFG.min_rows_for_forecast_dataset,
    )
    log.info(f"Forecast dataset: {len(df)} monthly observations")
    return df


def build_horizon_targets(df: pd.DataFrame, horizon_months: int) -> tuple[pd.DataFrame, np.ndarray]:
    target = df["ACTUAL_COLLECTIONS"].shift(-horizon_months)
    valid = ~target.isna()
    X = df.loc[valid, FORECAST_FEATURE_COLS].copy()
    y = target[valid].values
    return X, y


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    errors = y_true - y_pred
    loss = np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)
    return float(loss.mean())


def _train_quantile_models(df: pd.DataFrame, label: str) -> tuple[dict[tuple[int, float], Any], pd.DataFrame]:
    horizon_map = _build_horizon_map(FORECASTER_CFG.forecast_horizons)
    split_idx = int(len(df) * 0.8)

    models: dict[tuple[int, float], Any] = {}
    results: list[dict[str, Any]] = []

    for horizon_days, quantile in product(FORECASTER_CFG.forecast_horizons, FORECASTER_CFG.quantiles):
        horizon_months = horizon_map[horizon_days]
        X, y = build_horizon_targets(df, horizon_months)

        train_mask = np.arange(len(X)) < split_idx
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[~train_mask], y[~train_mask]
        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError(
                f"Insufficient data for {label} horizon={horizon_days}d after split: "
                f"train={len(X_train)}, test={len(X_test)}"
            )

        params = {
            **FORECASTER_CFG.lgbm_params,
            "objective": "quantile",
            "alpha": quantile,
            "metric": "quantile",
            "verbose": -1,
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        preds = model.predict(X_test)
        pinball = _pinball_loss(y_test, preds, quantile)
        q_key = round(float(quantile), 4)
        models[(horizon_days, q_key)] = model
        results.append(
            {
                "Model": label,
                "Horizon": f"{horizon_days}d",
                "Quantile": quantile,
                "Pinball Loss": round(pinball, 2),
                "N test": len(y_test),
            }
        )
        log.info(f"  [{label} | {horizon_days}d | q={quantile}] pinball={pinball:.2f}")

    return models, pd.DataFrame(results)


def train_forecaster(df: pd.DataFrame) -> dict[tuple[int, float], Any]:
    log.info("Training global probabilistic cash flow forecaster...")
    models, results_df = _train_quantile_models(df, "global")
    log.info(f"\n{results_df.to_string(index=False)}")
    return models


def _resolve_segment_filter_kwargs(segment_column: str, segment_value: str) -> dict[str, str]:
    if segment_column == "REGION":
        return {"region": segment_value}
    if segment_column == "INDUSTRY":
        return {"industry": segment_value}
    raise ValueError(f"Unsupported segment column for fallback forecaster: {segment_column}")


def prepare_training_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    inv, gl, scored = _load_forecast_sources()
    overall_df = _build_monthly_dataset_from_sources(
        inv,
        gl,
        scored,
        min_rows=FORECASTER_CFG.min_rows_for_forecast_dataset,
    )

    segment_column = str(FORECASTER_CFG.segment_column).strip().upper()
    if segment_column not in {"REGION", "INDUSTRY"}:
        raise ValueError(
            f"FORECASTER_CFG.segment_column must be REGION or INDUSTRY, got: {FORECASTER_CFG.segment_column}"
        )
    if segment_column not in inv.columns:
        raise KeyError(f"Segment column '{segment_column}' not found in invoice history")

    values = (
        inv[segment_column]
        .dropna()
        .astype(str)
        .str.strip()
    )
    segment_values = sorted(v for v in values.unique().tolist() if v != "")

    segment_datasets: dict[str, pd.DataFrame] = {}
    segment_summary: dict[str, dict[str, Any]] = {}

    for segment_value in segment_values:
        kwargs = _resolve_segment_filter_kwargs(segment_column, segment_value)
        seg_df = _build_monthly_dataset_from_sources(inv, gl, scored, min_rows=0, **kwargs)

        months = int(len(seg_df))
        cleared_invoices = int(seg_df["N_INVOICES_CLEARED"].sum()) if months > 0 else 0
        eligible = (
            months >= int(FORECASTER_CFG.min_months_for_segment_model)
            and cleared_invoices >= int(FORECASTER_CFG.min_cleared_invoices_for_segment_model)
        )

        reason_parts: list[str] = []
        if months < int(FORECASTER_CFG.min_months_for_segment_model):
            reason_parts.append(
                f"months<{int(FORECASTER_CFG.min_months_for_segment_model)}"
            )
        if cleared_invoices < int(FORECASTER_CFG.min_cleared_invoices_for_segment_model):
            reason_parts.append(
                f"cleared_invoices<{int(FORECASTER_CFG.min_cleared_invoices_for_segment_model)}"
            )

        segment_summary[segment_value] = {
            "months": months,
            "cleared_invoices": cleared_invoices,
            "eligible_for_segment_model": bool(eligible),
            "reason": ",".join(reason_parts) if reason_parts else "eligible",
            "trained": False,
        }

        if months >= int(FORECASTER_CFG.min_rows_for_forecast_dataset):
            segment_datasets[segment_value] = seg_df

    if segment_datasets:
        pooled_df = pd.concat(segment_datasets.values(), ignore_index=True)
        global_training_source = "segment_pooled"
    else:
        pooled_df = overall_df
        global_training_source = "overall_only"

    log.info(
        f"Prepared training data: overall_rows={len(overall_df)}, pooled_rows={len(pooled_df)}, "
        f"segment_candidates={len(segment_summary)}"
    )
    return overall_df, pooled_df, segment_datasets, segment_summary | {"__meta__": {"global_training_source": global_training_source}}


def train_segment_forecasters(
    segment_datasets: dict[str, pd.DataFrame],
    segment_summary: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[tuple[int, float], Any]], dict[str, dict[str, Any]]]:
    segment_models: dict[str, dict[tuple[int, float], Any]] = {}
    updated = dict(segment_summary)

    for segment_value, stats in list(updated.items()):
        if segment_value == "__meta__":
            continue
        if not stats.get("eligible_for_segment_model", False):
            continue

        seg_df = segment_datasets.get(segment_value)
        if seg_df is None or len(seg_df) < int(FORECASTER_CFG.min_months_for_segment_model):
            updated[segment_value]["reason"] = "eligible_but_dataset_not_available_after_feature_engineering"
            continue

        try:
            models, _ = _train_quantile_models(seg_df, f"segment:{segment_value}")
            segment_models[segment_value] = models
            updated[segment_value]["trained"] = True
            updated[segment_value]["reason"] = "trained"
        except Exception as exc:  # pragma: no cover - defensive path
            updated[segment_value]["trained"] = False
            updated[segment_value]["reason"] = f"training_failed:{exc}"
            log.warning(f"Segment model training failed for {segment_value}: {exc}")

    trained_count = sum(1 for k, v in updated.items() if k != "__meta__" and v.get("trained"))
    log.info(f"Trained segment-specific forecasters: {trained_count}")
    return segment_models, updated


def save_forecaster(
    models: dict[tuple[int, float], Any],
    df: pd.DataFrame,
    *,
    segment_models: dict[str, dict[tuple[int, float], Any]] | None = None,
    segment_summary: dict[str, dict[str, Any]] | None = None,
) -> None:
    payload = {
        "models": models,
        "feature_cols": FORECAST_FEATURE_COLS,
        "horizon_map": _build_horizon_map(FORECASTER_CFG.forecast_horizons),
        "quantiles": FORECASTER_CFG.quantiles,
        "horizons": FORECASTER_CFG.forecast_horizons,
        "last_period": df["COLLECT_MONTH"].max(),
        "segment_models": segment_models or {},
        "segment_summary": segment_summary or {},
        "segment_column": str(FORECASTER_CFG.segment_column).strip().upper(),
        "forecaster_version": "hybrid_v2",
    }
    with open(FORECASTER_PATH, "wb") as f:
        pickle.dump(payload, f)
    log.success(f"Forecaster saved -> {format_path(FORECASTER_PATH)}")


def load_forecaster() -> dict[str, Any]:
    with open(FORECASTER_PATH, "rb") as f:
        payload = pickle.load(f)
    log.info(f"Forecaster loaded from {format_path(FORECASTER_PATH)}")
    return payload


def _resolve_model_bundle(
    payload: dict[str, Any],
    *,
    region: str | None = None,
    industry: str | None = None,
) -> tuple[dict[tuple[int, float], Any], str, str | None]:
    global_models = payload["models"]
    segment_models = payload.get("segment_models", {})
    segment_column = str(payload.get("segment_column", str(FORECASTER_CFG.segment_column))).strip().upper()

    if segment_column == "REGION":
        selected = _normalize_selector(region)
    elif segment_column == "INDUSTRY":
        selected = _normalize_selector(industry)
    else:
        selected = None

    if selected and selected in segment_models:
        return segment_models[selected], "segment", selected
    return global_models, "global", selected


def _select_inference_dataset(
    df: pd.DataFrame,
    *,
    region: str | None = None,
    industry: str | None = None,
) -> pd.DataFrame:
    region_norm = _normalize_selector(region)
    industry_norm = _normalize_selector(industry)
    if region_norm is None and industry_norm is None:
        return df

    try:
        seg_df = build_forecast_dataset(region=region_norm, industry=industry_norm)
        if len(seg_df):
            return seg_df
    except Exception as exc:
        log.warning(
            "Segment forecast dataset unavailable "
            f"[region={region_norm or 'All'}, industry={industry_norm or 'All'}]: {exc}. "
            "Using provided baseline dataset."
        )
    return df


def generate_forecast(
    df: pd.DataFrame,
    *,
    region: str | None = None,
    industry: str | None = None,
) -> pd.DataFrame:
    payload = load_forecaster()
    models, model_source, model_segment = _resolve_model_bundle(payload, region=region, industry=industry)
    inference_df = _select_inference_dataset(df, region=region, industry=industry)

    if len(inference_df) == 0:
        raise ValueError("Inference dataset is empty; cannot generate forecast.")

    feature_cols = payload.get("feature_cols", FORECAST_FEATURE_COLS)
    latest_df = inference_df.copy()
    for col in feature_cols:
        if col not in latest_df.columns:
            latest_df[col] = 0.0
    latest = latest_df[feature_cols].iloc[[-1]]

    quantiles = [round(float(q), 4) for q in payload["quantiles"]]
    q_low = min(quantiles, key=lambda q: abs(q - 0.10))
    q_med = min(quantiles, key=lambda q: abs(q - 0.50))
    q_high = min(quantiles, key=lambda q: abs(q - 0.90))

    horizons = payload.get("horizons", FORECASTER_CFG.forecast_horizons)
    global_models = payload["models"]
    records: list[dict[str, Any]] = []

    for horizon_days in horizons:
        low_model = models.get((horizon_days, q_low)) or global_models.get((horizon_days, q_low))
        med_model = models.get((horizon_days, q_med)) or global_models.get((horizon_days, q_med))
        high_model = models.get((horizon_days, q_high)) or global_models.get((horizon_days, q_high))

        if low_model is None or med_model is None or high_model is None:
            raise KeyError(
                f"Missing quantile model(s) for horizon={horizon_days} in selected/global bundle."
            )

        lower = float(low_model.predict(latest)[0])
        median = float(med_model.predict(latest)[0])
        upper = float(high_model.predict(latest)[0])

        lower = min(lower, median)
        upper = max(upper, median)

        records.append(
            {
                "Horizon": f"{horizon_days} days",
                "Horizon_Days": horizon_days,
                "Lower_10pct": round(max(lower, 0), 2),
                "Median_50pct": round(max(median, 0), 2),
                "Upper_90pct": round(max(upper, 0), 2),
                "Range_Width": round(max(upper - lower, 0), 2),
                "Uncertainty_Pct": round(max((upper - lower) / max(median, 1) * 100, 0), 1),
                "Model_Source": model_source,
                "Model_Segment": model_segment or "All",
            }
        )

    forecast_df = pd.DataFrame(records)
    log.info(
        "Cash Flow Forecast "
        f"[source={model_source}, segment={model_segment or 'All'}, "
        f"region={_normalize_selector(region) or 'All'}, industry={_normalize_selector(industry) or 'All'}]"
    )
    log.info(forecast_df.to_string(index=False))
    return forecast_df


def plot_forecast(df: pd.DataFrame, forecast_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6))

    hist = df.tail(12)
    ax.plot(
        range(len(hist)),
        hist["ACTUAL_COLLECTIONS"] / 1e6,
        color="#378ADD",
        lw=2,
        marker="o",
        markersize=4,
        label="Historical collections",
    )

    x_forecast = [len(hist) - 1 + i for i in range(1, len(forecast_df) + 1)]
    medians = forecast_df["Median_50pct"].values / 1e6
    lowers = forecast_df["Lower_10pct"].values / 1e6
    uppers = forecast_df["Upper_90pct"].values / 1e6

    ax.plot(x_forecast, medians, color="#1D9E75", lw=2, marker="s", markersize=5, label="Forecast (median)")
    ax.fill_between(x_forecast, lowers, uppers, alpha=0.2, color="#1D9E75", label="80% confidence band")
    ax.plot(x_forecast, lowers, "--", color="#1D9E75", alpha=0.5, lw=1)
    ax.plot(x_forecast, uppers, "--", color="#1D9E75", alpha=0.5, lw=1)

    ax.axvline(len(hist) - 0.5, color="gray", linestyle=":", alpha=0.6)
    ax.text(len(hist) - 0.3, ax.get_ylim()[1] * 0.95, "Forecast ->", fontsize=10, color="gray")

    x_labels = list(hist["COLLECT_MONTH"].values) + [f"+{r['Horizon_Days']}d" for _, r in forecast_df.iterrows()]
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Collections ($M)")
    ax.set_title("Cash Flow Forecast - Historical + Forward Projection")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.1f}M"))

    plt.tight_layout()
    plot_path = EXPORTS_DIR / "cash_flow_forecast.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    log.success(f"Forecast plot saved -> {format_path(plot_path)}")
    plt.show()


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("Hybrid LightGBM Probabilistic Cash Flow Forecaster")
    log.info("=" * 60)

    overall_df, pooled_df, segment_datasets, segment_summary = prepare_training_data()
    models = train_forecaster(pooled_df)
    segment_models, segment_summary = train_segment_forecasters(segment_datasets, segment_summary)
    save_forecaster(
        models,
        overall_df,
        segment_models=segment_models,
        segment_summary=segment_summary,
    )

    forecast_df = generate_forecast(overall_df)
    plot_forecast(overall_df, forecast_df)

    log.info("\n=== Forecast Summary ===")
    log.info(
        "\n"
        + forecast_df[
            ["Horizon", "Lower_10pct", "Median_50pct", "Upper_90pct", "Uncertainty_Pct", "Model_Source"]
        ].to_string(index=False)
    )
    log.info("\nNext step: python -m src.models.explainer")
