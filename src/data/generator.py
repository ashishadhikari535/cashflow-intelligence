"""
SAP-style synthetic raw data generator.

Produces only raw files that resemble realistic SAP exports:
  - customer_master.csv (KNA1/KNB1/KNKK-like)
  - invoice_history.csv (BSID/BSAD-like)
  - ar_aging_fbl5n.csv (FBL5N-like snapshot)
  - gl_cash_positions.csv (FAGLFLEXT-like balances)
"""

from __future__ import annotations

from datetime import datetime, timedelta
import os
import random

from faker import Faker
import numpy as np
import pandas as pd


fake = Faker()
np.random.seed(42)
random.seed(42)

N_CUSTOMERS = 200
N_INVOICES = 5000
START_DATE = datetime(2021, 1, 1)
END_DATE = datetime(2024, 6, 30)
OUTPUT_DIR = "data/raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)

PAYMENT_TERMS = {
    "NT14": 14,
    "NT30": 30,
    "NT45": 45,
    "NT60": 60,
    "NT90": 90,
}

RISK_TIERS = ["Low", "Medium", "High"]
RISK_WEIGHTS = [0.55, 0.30, 0.15]
PAYMENT_DELAY_PROFILE = {
    "Low": {"mean": 2, "std": 3},
    "Medium": {"mean": 12, "std": 8},
    "High": {"mean": 28, "std": 15},
}

INDUSTRY_CODES = {
    "EN": "Energy",
    "MF": "Manufacturing",
    "RT": "Retail",
    "HC": "Healthcare",
    "CN": "Construction",
    "LG": "Logistics",
}

REGION_CODES = {
    "TX": "South",
    "NY": "Northeast",
    "IL": "Midwest",
    "CA": "West",
    "INTL": "International",
}

AGING_BUCKETS = ["Current", "1-30 days", "31-60 days", "61-90 days", "91-180 days", "180+ days"]


def generate_customer_master(n: int = N_CUSTOMERS) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Customer master export with SAP-style naming.

    Columns:
      KUNNR, BUKRS, NAME1, LAND1, REGIO, BRSCH, KTOKD, ZTERM, KLIMK, ERDAT
    """
    records: list[dict[str, object]] = []
    risk_profile: dict[str, str] = {}

    industry_keys = list(INDUSTRY_CODES.keys())
    region_keys = list(REGION_CODES.keys())

    for i in range(n):
        kunnr = f"C{str(i + 1).zfill(6)}"
        risk_tier = random.choices(RISK_TIERS, weights=RISK_WEIGHTS)[0]
        zterm = random.choice(list(PAYMENT_TERMS.keys()))
        regio = random.choice(region_keys)
        land1 = "US" if regio != "INTL" else fake.country_code()
        brsch = random.choice(industry_keys)

        records.append(
            {
                "KUNNR": kunnr,
                "BUKRS": "1000",
                "NAME1": fake.company(),
                "LAND1": land1,
                "REGIO": regio,
                "BRSCH": brsch,
                "KTOKD": random.choice(["Z001", "Z002", "Z003"]),
                "ZTERM": zterm,
                "KLIMK": round(random.uniform(50_000, 2_000_000), 2),
                "ERDAT": fake.date_between(start_date="-5y", end_date="-2y"),
            }
        )
        risk_profile[kunnr] = risk_tier

    df = pd.DataFrame(records)
    df.to_csv(f"{OUTPUT_DIR}/customer_master.csv", index=False)
    print(f"[OK] customer_master.csv - {len(df)} customers")
    return df, risk_profile


def generate_invoice_history(
    customers_df: pd.DataFrame,
    risk_profile: dict[str, str],
    n: int = N_INVOICES,
) -> pd.DataFrame:
    """
    Invoice history export with SAP-style naming.

    Columns:
      BELNR, BUKRS, GJAHR, BUZEI, KUNNR, BLART, BLDAT, BUDAT, FAEDT, AUGDT,
      DMBTR, WAERS, SHKZG, ZTERM
    """
    records: list[dict[str, object]] = []
    doc_num = 1800000000

    for _ in range(n):
        cust = customers_df.sample(1).iloc[0]
        kunnr = str(cust["KUNNR"])
        doc_dt = fake.date_between(start_date=START_DATE, end_date=END_DATE)
        due_dt = doc_dt + timedelta(days=int(PAYMENT_TERMS[str(cust["ZTERM"])]))

        delay_cfg = PAYMENT_DELAY_PROFILE.get(risk_profile.get(kunnr, "Medium"), {"mean": 5, "std": 5})
        delay_days = int(np.random.normal(delay_cfg["mean"], delay_cfg["std"]))
        delay_days = max(delay_days, -5)
        paid_dt = due_dt + timedelta(days=delay_days)

        is_open = (
            risk_profile.get(kunnr) == "High"
            and random.random() < 0.15
            and paid_dt > END_DATE.date()
        )

        records.append(
            {
                "BELNR": str(doc_num),
                "BUKRS": "1000",
                "GJAHR": str(doc_dt.year),
                "BUZEI": "001",
                "KUNNR": kunnr,
                "BLART": "DR",
                "BLDAT": doc_dt,
                "BUDAT": doc_dt,
                "FAEDT": due_dt,
                "AUGDT": None if is_open else paid_dt,
                "DMBTR": round(random.uniform(1_000, 250_000), 2),
                "WAERS": "USD",
                "SHKZG": "S",
                "ZTERM": str(cust["ZTERM"]),
            }
        )
        doc_num += 1

    df = pd.DataFrame(records)
    df.to_csv(f"{OUTPUT_DIR}/invoice_history.csv", index=False)
    print(f"[OK] invoice_history.csv - {len(df)} invoices")
    return df


def generate_ar_aging(invoices_df: pd.DataFrame, as_of_date: datetime = END_DATE) -> pd.DataFrame:
    """
    FBL5N-style customer AR aging snapshot.

    Columns:
      KUNNR, BUKRS, STIDA, <aging buckets>, TOTAL_OUTSTANDING, WAERS
    """
    inv = invoices_df.copy()
    inv["AUGDT"] = pd.to_datetime(inv["AUGDT"], errors="coerce")
    inv["FAEDT"] = pd.to_datetime(inv["FAEDT"], errors="coerce")
    open_inv = inv[inv["AUGDT"].isna()].copy()

    if open_inv.empty:
        cols = ["KUNNR", "BUKRS", "STIDA", *AGING_BUCKETS, "TOTAL_OUTSTANDING", "WAERS"]
        aging = pd.DataFrame(columns=cols)
        aging.to_csv(f"{OUTPUT_DIR}/ar_aging_fbl5n.csv", index=False)
        print("[OK] ar_aging_fbl5n.csv - 0 customer aging rows")
        return aging

    open_inv["DAYS_OVERDUE"] = (as_of_date - open_inv["FAEDT"]).dt.days

    def bucket(days: int) -> str:
        if days <= 0:
            return "Current"
        if days <= 30:
            return "1-30 days"
        if days <= 60:
            return "31-60 days"
        if days <= 90:
            return "61-90 days"
        if days <= 180:
            return "91-180 days"
        return "180+ days"

    open_inv["AGING_BUCKET"] = open_inv["DAYS_OVERDUE"].apply(bucket)

    aging = (
        open_inv.groupby(["KUNNR", "AGING_BUCKET"])["DMBTR"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=AGING_BUCKETS, fill_value=0)
        .reset_index()
    )
    aging["BUKRS"] = "1000"
    aging["STIDA"] = as_of_date.date()
    aging["TOTAL_OUTSTANDING"] = aging[AGING_BUCKETS].sum(axis=1)
    aging["WAERS"] = "USD"
    aging = aging[["KUNNR", "BUKRS", "STIDA", *AGING_BUCKETS, "TOTAL_OUTSTANDING", "WAERS"]]

    aging.to_csv(f"{OUTPUT_DIR}/ar_aging_fbl5n.csv", index=False)
    print(f"[OK] ar_aging_fbl5n.csv - {len(aging)} customer aging rows")
    return aging


def generate_gl_cash_positions() -> pd.DataFrame:
    """
    FAGLFLEXT-like GL period balances.

    Columns:
      RBUKRS, RACCT, GJAHR, POPER, HSL, WAERS
    """
    cash_accounts = ["100000", "113100", "113200", "113300"]
    records: list[dict[str, object]] = []

    current = START_DATE
    while current <= END_DATE:
        gjahr = str(current.year)
        poper = f"{current.month:03d}"
        for acct in cash_accounts:
            base_bal = random.uniform(500_000, 5_000_000)
            seasonal = 1 + 0.15 * np.sin(2 * np.pi * current.month / 12)
            hsl = round(base_bal * seasonal + np.random.normal(0, 50_000), 2)
            records.append(
                {
                    "RBUKRS": "1000",
                    "RACCT": acct,
                    "GJAHR": gjahr,
                    "POPER": poper,
                    "HSL": hsl,
                    "WAERS": "USD",
                }
            )
        current += timedelta(days=32)
        current = current.replace(day=1)

    df = pd.DataFrame(records)
    df.to_csv(f"{OUTPUT_DIR}/gl_cash_positions.csv", index=False)
    print(f"[OK] gl_cash_positions.csv - {len(df)} GL rows")
    return df


if __name__ == "__main__":
    print("\n=== SAP-style Raw Data Generator ===\n")

    customers, risk_profile = generate_customer_master()
    invoices = generate_invoice_history(customers, risk_profile)
    aging = generate_ar_aging(invoices)
    gl = generate_gl_cash_positions()

    inv = invoices.copy()
    inv["AUGDT"] = pd.to_datetime(inv["AUGDT"], errors="coerce")
    inv["FAEDT"] = pd.to_datetime(inv["FAEDT"], errors="coerce")
    cleared = inv[inv["AUGDT"].notna()].copy()
    open_items = inv[inv["AUGDT"].isna()].copy()

    if cleared.empty:
        late_rate = 0.0
    else:
        days_late = (cleared["AUGDT"] - cleared["FAEDT"]).dt.days
        late_rate = float((days_late > 0).mean())

    print("=== Summary ===")
    print(f"Customers      : {len(customers)}")
    print(f"Invoices       : {len(invoices)}")
    print(f"Open AR items  : {len(open_items)}")
    print(f"Cleared items  : {len(cleared)}")
    print(f"Late rate      : {late_rate:.1%}")
    print(f"Aging rows     : {len(aging)}")
    print(f"GL rows        : {len(gl)}")
    print(f"\nFiles saved to: ./{OUTPUT_DIR}/")
    print("\nNext steps: python -m src.data.validator, then python -m src.data.features")
