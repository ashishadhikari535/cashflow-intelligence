"""
Schema and data quality validation for SAP-style raw datasets.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.utils.config import AR_AGING_PATH, CUSTOMER_MASTER_PATH, DATA_CFG, GL_CASH_PATH, INVOICE_HISTORY_PATH
from src.utils.logger import get_logger

log = get_logger(__name__)


INDUSTRY_CODES = ["EN", "MF", "RT", "HC", "CN", "LG"]
REGION_CODES = ["TX", "NY", "IL", "CA", "INTL"]
AGING_BUCKETS = ["Current", "1-30 days", "31-60 days", "61-90 days", "91-180 days", "180+ days"]


SCHEMAS = {
    "customer_master": {
        "required_cols": ["KUNNR", "BUKRS", "NAME1", "LAND1", "REGIO", "BRSCH", "ZTERM", "KLIMK", "ERDAT"],
        "no_nulls": ["KUNNR", "BUKRS", "NAME1", "LAND1", "ZTERM", "KLIMK"],
        "valid_values": {
            "BRSCH": INDUSTRY_CODES,
            "REGIO": REGION_CODES,
            "ZTERM": list(DATA_CFG.payment_terms.keys()),
        },
        "numeric_ranges": {
            "KLIMK": (1_000, 10_000_000),
        },
    },
    "invoice_history": {
        "required_cols": [
            "BELNR",
            "BUKRS",
            "GJAHR",
            "BUZEI",
            "KUNNR",
            "BLDAT",
            "BUDAT",
            "FAEDT",
            "DMBTR",
            "WAERS",
            "SHKZG",
            "ZTERM",
        ],
        "no_nulls": ["BELNR", "BUKRS", "KUNNR", "BLDAT", "FAEDT", "DMBTR", "WAERS", "SHKZG", "ZTERM"],
        "valid_values": {
            "SHKZG": ["S"],
            "WAERS": ["USD"],
            "ZTERM": list(DATA_CFG.payment_terms.keys()),
        },
        "numeric_ranges": {
            "DMBTR": (0, 10_000_000),
        },
    },
    "ar_aging": {
        "required_cols": ["KUNNR", "BUKRS", "STIDA", *AGING_BUCKETS, "TOTAL_OUTSTANDING", "WAERS"],
        "no_nulls": ["KUNNR", "BUKRS", "STIDA", "TOTAL_OUTSTANDING"],
        "valid_values": {"WAERS": ["USD"]},
        "numeric_ranges": {
            "TOTAL_OUTSTANDING": (0, 1_000_000_000),
        },
    },
    "gl_cash_positions": {
        "required_cols": ["RBUKRS", "RACCT", "GJAHR", "POPER", "HSL", "WAERS"],
        "no_nulls": ["RBUKRS", "RACCT", "GJAHR", "POPER", "HSL", "WAERS"],
        "valid_values": {"WAERS": ["USD"]},
        "numeric_ranges": {"HSL": (0, 100_000_000)},
    },
}


def check_columns(df: pd.DataFrame, required: list[str], name: str) -> list[str]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.error(f"[{name}] Missing columns: {missing}")
    else:
        log.info(f"[{name}] All required columns present ({len(required)})")
    return missing


def check_nulls(df: pd.DataFrame, no_null_cols: list[str], name: str) -> dict[str, int]:
    issues: dict[str, int] = {}
    denom = max(len(df), 1)
    for col in no_null_cols:
        if col not in df.columns:
            continue
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            issues[col] = null_count
            log.warning(f"[{name}] NULL in {col}: {null_count} rows ({null_count/denom:.1%})")
    if not issues:
        log.info(f"[{name}] No critical nulls found")
    return issues


def check_valid_values(df: pd.DataFrame, valid_vals: dict[str, list[str]], name: str) -> dict[str, list[str]]:
    issues: dict[str, list[str]] = {}
    for col, allowed in valid_vals.items():
        if col not in df.columns:
            continue
        invalid = df[~df[col].isin(allowed)][col].dropna().astype(str).unique()
        if len(invalid) > 0:
            issues[col] = invalid.tolist()
            log.warning(f"[{name}] Invalid values in {col}: {invalid.tolist()}")
    if not issues:
        log.info(f"[{name}] All categorical values valid")
    return issues


def check_numeric_ranges(df: pd.DataFrame, ranges: dict[str, tuple[float, float]], name: str) -> dict[str, int]:
    issues: dict[str, int] = {}
    for col, (low, high) in ranges.items():
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")

        # Treat non-numeric raw values as data quality failures.
        non_numeric_mask = numeric.isna() & df[col].notna()
        non_numeric_count = int(non_numeric_mask.sum())
        if non_numeric_count > 0:
            issues[f"{col}__non_numeric"] = non_numeric_count
            log.warning(f"[{name}] {col} non-numeric values: {non_numeric_count} rows")

        out_of_range = int(((numeric < low) | (numeric > high)).sum())
        if out_of_range > 0:
            issues[col] = out_of_range
            log.warning(f"[{name}] {col} out of range [{low}, {high}]: {out_of_range} rows")
    if not issues:
        log.info(f"[{name}] All numeric ranges valid")
    return issues


def check_referential_integrity(invoices_df: pd.DataFrame, customers_df: pd.DataFrame) -> int:
    inv_customers = set(invoices_df["KUNNR"].astype(str).unique())
    master_customers = set(customers_df["KUNNR"].astype(str).unique())
    orphans = inv_customers - master_customers
    if orphans:
        log.warning(f"[referential] {len(orphans)} invoice customers not in customer master")
    else:
        log.info("[referential] All invoice customer IDs found in customer master")
    return len(orphans)


def check_invoice_date_consistency(invoices_df: pd.DataFrame) -> int:
    if not {"BLDAT", "FAEDT", "AUGDT"}.issubset(invoices_df.columns):
        return 0

    tmp = invoices_df.copy()
    tmp["BLDAT"] = pd.to_datetime(tmp["BLDAT"], errors="coerce")
    tmp["FAEDT"] = pd.to_datetime(tmp["FAEDT"], errors="coerce")
    tmp["AUGDT"] = pd.to_datetime(tmp["AUGDT"], errors="coerce")

    issues = 0

    due_before_doc = int((tmp["FAEDT"] < tmp["BLDAT"]).sum())
    if due_before_doc > 0:
        issues += due_before_doc
        log.warning(f"[invoice dates] FAEDT earlier than BLDAT: {due_before_doc} rows")

    cleared_before_doc = int(((tmp["AUGDT"].notna()) & (tmp["AUGDT"] < tmp["BLDAT"])).sum())
    if cleared_before_doc > 0:
        issues += cleared_before_doc
        log.warning(f"[invoice dates] AUGDT earlier than BLDAT: {cleared_before_doc} rows")

    if issues == 0:
        log.info("[invoice dates] Invoice date consistency checks passed")
    return issues


def check_ar_aging_consistency(aging_df: pd.DataFrame) -> int:
    required = set(AGING_BUCKETS + ["TOTAL_OUTSTANDING"])
    if not required.issubset(set(aging_df.columns)):
        return 0

    issues = 0
    totals = pd.to_numeric(aging_df["TOTAL_OUTSTANDING"], errors="coerce").fillna(0.0)
    recomputed = (
        aging_df[AGING_BUCKETS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .sum(axis=1)
    )

    mismatch = ~np.isclose(totals.values, recomputed.values, atol=0.01)
    mismatch_count = int(mismatch.sum())
    if mismatch_count > 0:
        issues += mismatch_count
        log.warning(f"[ar_aging] TOTAL_OUTSTANDING mismatch vs bucket sum: {mismatch_count} rows")

    negative_buckets = int((aging_df[AGING_BUCKETS].apply(pd.to_numeric, errors="coerce").fillna(0.0) < 0).sum().sum())
    if negative_buckets > 0:
        issues += negative_buckets
        log.warning(f"[ar_aging] Negative aging bucket values: {negative_buckets} cells")

    if issues == 0:
        log.info("[ar_aging] Aging consistency checks passed")
    return issues


def check_gl_period_bounds(gl_df: pd.DataFrame) -> int:
    if "POPER" not in gl_df.columns:
        return 0

    period = pd.to_numeric(gl_df["POPER"], errors="coerce")
    invalid_period = int(((period < 1) | (period > 12) | period.isna()).sum())
    if invalid_period > 0:
        log.warning(f"[gl_cash_positions] POPER outside 1-12 or invalid: {invalid_period} rows")
    else:
        log.info("[gl_cash_positions] POPER period bounds check passed")
    return invalid_period


def class_balance_report(invoices_df: pd.DataFrame) -> None:
    if "AUGDT" not in invoices_df.columns or "FAEDT" not in invoices_df.columns:
        return

    tmp = invoices_df.copy()
    tmp["AUGDT"] = pd.to_datetime(tmp["AUGDT"], errors="coerce")
    tmp["FAEDT"] = pd.to_datetime(tmp["FAEDT"], errors="coerce")

    cleared = tmp[tmp["AUGDT"].notna()].copy()
    if cleared.empty:
        log.warning("[class balance] No cleared invoices found")
        return

    days_late = (cleared["AUGDT"] - cleared["FAEDT"]).dt.days
    late = int((days_late > 0).sum())
    total = int(len(cleared))
    on_time = total - late

    log.info(f"[class balance] Total cleared: {total:,}")
    log.info(f"[class balance] Paid late:    {late:,} ({late/max(total, 1):.1%})")
    log.info(f"[class balance] Paid on time: {on_time:,} ({on_time/max(total, 1):.1%})")

    if late / max(total, 1) < 0.1:
        log.warning("[class balance] High class imbalance - consider scale_pos_weight in XGBoost")


def validate_all() -> bool:
    log.info("=" * 50)
    log.info("Starting raw data validation...")
    log.info("=" * 50)

    all_passed = True
    datasets = {
        "customer_master": (CUSTOMER_MASTER_PATH, "customer_master"),
        "invoice_history": (INVOICE_HISTORY_PATH, "invoice_history"),
        "ar_aging": (AR_AGING_PATH, "ar_aging"),
        "gl_cash_positions": (GL_CASH_PATH, "gl_cash_positions"),
    }

    loaded: dict[str, pd.DataFrame] = {}
    for key, (path, schema_key) in datasets.items():
        if not path.exists():
            log.error(f"File not found: {path}. Run 'python -m src.data.generator' first.")
            all_passed = False
            continue

        df = pd.read_csv(path)
        loaded[key] = df
        schema = SCHEMAS[schema_key]
        log.info(f"\n-- Validating {key} ({len(df):,} rows) --")

        missing_cols = check_columns(df, schema["required_cols"], key)
        if missing_cols:
            all_passed = False
            continue

        null_issues = check_nulls(df, schema["no_nulls"], key)
        value_issues = check_valid_values(df, schema["valid_values"], key)
        range_issues = check_numeric_ranges(df, schema["numeric_ranges"], key)
        if null_issues or value_issues or range_issues:
            all_passed = False

    if "customer_master" in loaded and "invoice_history" in loaded:
        log.info("\n-- Referential integrity --")
        orphan_count = check_referential_integrity(loaded["invoice_history"], loaded["customer_master"])
        if orphan_count > 0:
            all_passed = False

    if "invoice_history" in loaded:
        log.info("\n-- Invoice date consistency --")
        date_issues = check_invoice_date_consistency(loaded["invoice_history"])
        if date_issues > 0:
            all_passed = False

    if "ar_aging" in loaded:
        log.info("\n-- AR aging consistency --")
        aging_issues = check_ar_aging_consistency(loaded["ar_aging"])
        if aging_issues > 0:
            all_passed = False

    if "gl_cash_positions" in loaded:
        log.info("\n-- GL fiscal period bounds --")
        gl_issues = check_gl_period_bounds(loaded["gl_cash_positions"])
        if gl_issues > 0:
            all_passed = False

    if "invoice_history" in loaded:
        log.info("\n-- Class balance (derived from clearing dates) --")
        class_balance_report(loaded["invoice_history"])

    log.info("\n" + "=" * 50)
    if all_passed:
        log.success("All validations passed - data is ready for feature engineering")
    else:
        log.error("Validation failed - fix schema/file issues above")
    return all_passed


if __name__ == "__main__":
    passed = validate_all()
    sys.exit(0 if passed else 1)
