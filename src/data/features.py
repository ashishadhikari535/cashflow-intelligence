"""
Feature engineering pipeline for the Cash Flow Intelligence System.

Raw inputs are SAP-style exports. Model-ready columns are derived in this module.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import (  # noqa: E402
    CLASSIFIER_CFG,
    CUSTOMER_MASTER_PATH,
    DATA_CFG,
    FEATURE_STORE_PATH,
    GL_CASH_PATH,
    INVOICE_HISTORY_PATH,
    TEST_PATH,
    TRAIN_PATH,
)
from src.utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)


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
    df = customers.copy()

    # Backward-compatible aliases if old synthetic raw files are still present.
    if "KLIMK" not in df.columns and "CREDIT_LIMIT" in df.columns:
        df["KLIMK"] = df["CREDIT_LIMIT"]
    if "LAND1" not in df.columns and "COUNTRY" in df.columns:
        df["LAND1"] = df["COUNTRY"]
    if "REGIO" not in df.columns and "REGION" in df.columns:
        inv_region = {v: k for k, v in REGION_MAP.items()}
        df["REGIO"] = df["REGION"].map(inv_region).fillna("INTL")
    if "BRSCH" not in df.columns and "INDUSTRY" in df.columns:
        inv_industry = {v: k for k, v in INDUSTRY_MAP.items()}
        df["BRSCH"] = df["INDUSTRY"].map(inv_industry).fillna("MF")

    df["INDUSTRY"] = df["BRSCH"].astype(str).map(INDUSTRY_MAP).fillna("Other")
    df["REGION"] = df["REGIO"].astype(str).map(REGION_MAP).fillna("International")
    df["KLIMK"] = pd.to_numeric(df["KLIMK"], errors="coerce")

    keep = ["KUNNR", "ZTERM", "KLIMK", "INDUSTRY", "REGION"]
    for col in keep:
        if col not in df.columns:
            raise KeyError(f"Missing required customer master column after normalization: {col}")
    return df[keep].copy()


def _normalize_invoice_history(invoices: pd.DataFrame) -> pd.DataFrame:
    df = invoices.copy()
    for col in ["BLDAT", "FAEDT", "AUGDT"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Derive runtime columns from SAP fields.
    if "STATUS" not in df.columns:
        df["STATUS"] = np.where(df["AUGDT"].isna(), "OPEN", "CLEARED")

    if "DAYS_LATE" not in df.columns:
        days_late = (df["AUGDT"] - df["FAEDT"]).dt.days
        df["DAYS_LATE"] = np.where(df["AUGDT"].isna(), np.nan, days_late)

    if "LATE_FLAG" not in df.columns:
        df["LATE_FLAG"] = np.where(df["AUGDT"].isna(), np.nan, (df["DAYS_LATE"] > 0).astype(float))

    df["DMBTR"] = pd.to_numeric(df["DMBTR"], errors="coerce")
    return df


def _normalize_gl_cash(gl: pd.DataFrame) -> pd.DataFrame:
    df = gl.copy()
    df["HSL"] = pd.to_numeric(df["HSL"], errors="coerce")

    # Support both FAGLFLEXT-style (GJAHR/POPER) and legacy FISCPER.
    if "FISCPER" not in df.columns:
        if "GJAHR" in df.columns and "POPER" in df.columns:
            year = df["GJAHR"].astype(str).str.slice(0, 4)
            period = pd.to_numeric(df["POPER"], errors="coerce").fillna(1).astype(int).clip(1, 12)
            df["FISCPER"] = year + period.astype(str).str.zfill(2)
        else:
            raise KeyError("GL cash file must contain either FISCPER or (GJAHR and POPER).")
    else:
        fis = df["FISCPER"].astype(str)
        if fis.str.len().eq(7).all():
            # Convert YYYYPPP -> YYYYMM (assume PPP is fiscal month period).
            year = fis.str.slice(0, 4)
            per = pd.to_numeric(fis.str.slice(4), errors="coerce").fillna(1).astype(int).clip(1, 12)
            df["FISCPER"] = year + per.astype(str).str.zfill(2)
        else:
            df["FISCPER"] = fis

    return df


def _derive_risk_tier(late_rate: pd.Series) -> pd.Series:
    bins = [-np.inf, 0.20, 0.50, np.inf]
    labels = ["Low", "Medium", "High"]
    return pd.cut(late_rate.fillna(0.0), bins=bins, labels=labels, include_lowest=True).astype(str)


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log.info("Loading raw SAP datasets...")

    customers = pd.read_csv(CUSTOMER_MASTER_PATH)
    invoices = pd.read_csv(INVOICE_HISTORY_PATH)
    gl = pd.read_csv(GL_CASH_PATH)

    customers = _normalize_customer_master(customers)
    invoices = _normalize_invoice_history(invoices)
    gl = _normalize_gl_cash(gl)

    log.info(f"Customers: {len(customers):,} | Invoices: {len(invoices):,} | GL rows: {len(gl):,}")
    return customers, invoices, gl


def build_customer_rolling_features(invoices: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling customer features from cleared invoice history using only prior data.
    """
    log.info("Building customer rolling features...")

    cleared = invoices[invoices["STATUS"] == "CLEARED"].copy()
    cleared = cleared.sort_values(["KUNNR", "BLDAT"]).reset_index(drop=True)
    grp = cleared.groupby("KUNNR")

    cleared["CUST_AVG_DAYS_LATE"] = grp["DAYS_LATE"].transform(lambda s: s.shift(1).expanding().mean())
    cleared["CUST_LATE_RATE"] = grp["LATE_FLAG"].transform(lambda s: s.shift(1).expanding().mean())
    cleared["CUST_INVOICE_COUNT"] = grp.cumcount()
    cleared["CUST_AVG_AMOUNT"] = grp["DMBTR"].transform(lambda s: s.shift(1).expanding().mean())
    cleared["CUST_TOTAL_EXPOSURE"] = grp["DMBTR"].transform(lambda s: s.shift(1).expanding().sum())
    cleared["CUST_ROLLING3_DAYS_LATE"] = grp["DAYS_LATE"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
    )
    cleared["CUST_MAX_INVOICE"] = grp["DMBTR"].transform(lambda s: s.shift(1).expanding().max())

    cleared["CUST_AVG_DAYS_LATE"] = cleared["CUST_AVG_DAYS_LATE"].fillna(0.0)
    cleared["CUST_LATE_RATE"] = cleared["CUST_LATE_RATE"].fillna(0.0)
    cleared["CUST_AVG_AMOUNT"] = cleared["CUST_AVG_AMOUNT"].fillna(cleared["DMBTR"])
    cleared["CUST_TOTAL_EXPOSURE"] = cleared["CUST_TOTAL_EXPOSURE"].fillna(0.0)
    cleared["CUST_ROLLING3_DAYS_LATE"] = cleared["CUST_ROLLING3_DAYS_LATE"].fillna(
        cleared["CUST_AVG_DAYS_LATE"]
    )
    cleared["CUST_MAX_INVOICE"] = cleared["CUST_MAX_INVOICE"].fillna(0.0)
    cleared["CUST_LATE_TREND"] = cleared["CUST_ROLLING3_DAYS_LATE"] - cleared["CUST_AVG_DAYS_LATE"]
    cleared["CUST_AMOUNT_CONCENTRATION"] = (
        cleared["DMBTR"] / cleared["CUST_AVG_AMOUNT"].replace(0, np.nan)
    ).fillna(1.0)

    log.info(f"Customer rolling features built - {len(cleared):,} rows")
    return cleared


def build_invoice_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Building invoice-level features...")
    out = df.copy()

    out["INVOICE_MONTH"] = out["BLDAT"].dt.month
    out["INVOICE_QUARTER"] = out["BLDAT"].dt.quarter
    out["INVOICE_DOW"] = out["BLDAT"].dt.dayofweek
    out["INVOICE_DOM"] = out["BLDAT"].dt.day
    out["IS_QUARTER_END"] = out["INVOICE_MONTH"].isin([3, 6, 9, 12]).astype(int)
    out["IS_MONTH_END"] = (out["INVOICE_DOM"] >= 25).astype(int)
    out["IS_MONDAY"] = (out["INVOICE_DOW"] == 0).astype(int)

    out["LOG_AMOUNT"] = np.log1p(out["DMBTR"])
    out["AMOUNT_BUCKET"] = pd.qcut(
        out["DMBTR"], q=5, labels=["XS", "S", "M", "L", "XL"], duplicates="drop"
    ).cat.codes

    out["PAYMENT_DAYS_NUM"] = out["ZTERM"].map(DATA_CFG.payment_terms)
    out["ACTUAL_TERM_DAYS"] = (out["FAEDT"] - out["BLDAT"]).dt.days
    return out


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Encoding categorical features...")
    out = df.copy()
    out["INDUSTRY_CODE"] = pd.Categorical(out["INDUSTRY"]).codes
    out["REGION_CODE"] = pd.Categorical(out["REGION"]).codes
    return out


def merge_gl_context(df: pd.DataFrame, gl: pd.DataFrame) -> pd.DataFrame:
    log.info("Merging GL cash context...")

    gl_agg = gl.groupby("FISCPER")["HSL"].sum().reset_index().rename(columns={"HSL": "TOTAL_CASH_POSITION"})
    gl_agg["FISCPER"] = gl_agg["FISCPER"].astype(str)

    out = df.copy()
    out["FISCPER"] = out["BLDAT"].dt.strftime("%Y%m")
    out = out.merge(gl_agg, on="FISCPER", how="left")
    out["LOG_CASH_POSITION"] = np.log1p(out["TOTAL_CASH_POSITION"].fillna(0))
    return out


def build_feature_store(customers: pd.DataFrame, invoices: pd.DataFrame, gl: pd.DataFrame) -> pd.DataFrame:
    log.info("=" * 50)
    log.info("Building feature store...")
    log.info("=" * 50)

    cleared = build_customer_rolling_features(invoices)

    # Attach customer dimensions from SAP master.
    cust_dim = customers[["KUNNR", "INDUSTRY", "REGION", "KLIMK"]].drop_duplicates(subset=["KUNNR"])
    cleared = cleared.merge(cust_dim, on="KUNNR", how="left")

    cleared = build_invoice_features(cleared)
    cleared = encode_categoricals(cleared)
    cleared = merge_gl_context(cleared, gl)

    cleared["INVOICE_TO_CREDIT_RATIO"] = cleared["DMBTR"] / cleared["KLIMK"].replace(0, np.nan)
    cleared["INVOICE_TO_CREDIT_RATIO"] = cleared["INVOICE_TO_CREDIT_RATIO"].fillna(0.0)
    cleared["RISK_TIER"] = _derive_risk_tier(cleared["CUST_LATE_RATE"])

    before = len(cleared)
    cleared = cleared.dropna(subset=["LATE_FLAG"] + CLASSIFIER_CFG.feature_cols)
    after = len(cleared)
    if before != after:
        log.warning(f"Dropped {before - after:,} rows with nulls in feature/target cols")

    keep_cols = [
        "BELNR",
        "KUNNR",
        "BLDAT",
        "FAEDT",
        "DMBTR",
        "INDUSTRY",
        "REGION",
        "ZTERM",
        "RISK_TIER",
        "STATUS",
        *CLASSIFIER_CFG.feature_cols,
        "CUST_ROLLING3_DAYS_LATE",
        "CUST_LATE_TREND",
        "CUST_MAX_INVOICE",
        "CUST_AMOUNT_CONCENTRATION",
        "CUST_TOTAL_EXPOSURE",
        "INVOICE_TO_CREDIT_RATIO",
        "ACTUAL_TERM_DAYS",
        "IS_MONTH_END",
        "IS_MONDAY",
        "LOG_CASH_POSITION",
        "FISCPER",
        "LATE_FLAG",
        "DAYS_LATE",
    ]
    keep_cols = [c for c in keep_cols if c in cleared.columns]
    feature_store = cleared[keep_cols].reset_index(drop=True)

    log.success(f"Feature store complete - {len(feature_store):,} rows | {feature_store.shape[1]} columns")
    log.info(f"Late payment rate: {feature_store['LATE_FLAG'].mean():.1%}")
    log.info(f"Risk tier dist:\n{feature_store['RISK_TIER'].value_counts(normalize=True).round(3)}")
    return feature_store


def split_and_save(feature_store: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("Splitting train/test (time-aware)...")
    feature_store = feature_store.sort_values("BLDAT").reset_index(drop=True)
    split_idx = int(len(feature_store) * (1 - CLASSIFIER_CFG.test_size))

    train = feature_store.iloc[:split_idx]
    test = feature_store.iloc[split_idx:]

    log.info(f"Train: {len(train):,} rows ({train['BLDAT'].min()} -> {train['BLDAT'].max()})")
    log.info(f"Test:  {len(test):,} rows ({test['BLDAT'].min()} -> {test['BLDAT'].max()})")

    feature_store.to_csv(FEATURE_STORE_PATH, index=False)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    log.success(f"Saved -> {FEATURE_STORE_PATH.name} | {TRAIN_PATH.name} | {TEST_PATH.name}")
    return train, test


if __name__ == "__main__":
    customers_df, invoices_df, gl_df = load_raw_data()
    fs = build_feature_store(customers_df, invoices_df, gl_df)
    train_df, test_df = split_and_save(fs)

    log.info("\n=== Feature Summary ===")
    log.info(f"Total features : {len(CLASSIFIER_CFG.feature_cols)}")
    log.info(f"Train rows     : {len(train_df):,}")
    log.info(f"Test rows      : {len(test_df):,}")
    log.info("Next step      : make train")
