# Dataset Documentation

This project now treats `data/raw/` as SAP-style source exports only.
Model labels and engineered fields are computed later in `src/data/features.py`.

## Raw Files

- `data/raw/customer_master.csv`
- `data/raw/invoice_history.csv`
- `data/raw/ar_aging_fbl5n.csv`
- `data/raw/gl_cash_positions.csv`

## Raw -> Processed Flow

```text
data/raw/*.csv
  -> src/data/validator.py   (raw schema and integrity checks)
  -> src/data/features.py    (derive labels + engineered features)
  -> data/processed/feature_store.csv
  -> data/processed/train.csv + data/processed/test.csv
```

## SAP-Style Schemas

### `customer_master.csv` (KNA1/KNB1/KNKK-like)

- Grain: one row per customer (`KUNNR`)
- Key columns:
  - `KUNNR`: customer number | Unique customer identifier used as the main join key across files.
  - `BUKRS`: company code | Legal entity/company within SAP that owns the customer relationship.
  - `NAME1`: customer name | Customer legal/trade name for reporting and operational visibility.
  - `LAND1`: country key | Country code used for geography, compliance, and regional segmentation.
  - `REGIO`: region/state key | Sub-country region/state useful for location-level analysis.
  - `BRSCH`: industry key | Industry classification code used for sector-level risk segmentation.
  - `KTOKD`: account group | Customer grouping used for master-data control and reporting structure.
  - `ZTERM`: payment terms key | Contracted payment term code (for example, `NT30`, `NT60`).
  - `KLIMK`: credit limit | Approved customer credit exposure limit in local currency.
  - `ERDAT`: record creation date | Date the customer master record was created.

### `invoice_history.csv` (BSID/BSAD-like)

- Grain: one row per invoice line item (`BELNR`, `BUZEI`)
- Key columns:
  - `BELNR`: document number | Accounting document number (invoice-level identifier in FI).
  - `BUKRS`: company code | Company/entity where the invoice is posted.
  - `GJAHR`: fiscal year | Fiscal year of the document posting.
  - `BUZEI`: line item | Line number within the accounting document.
  - `KUNNR`: customer number | Links invoice activity back to customer master data.
  - `BLART`: document type | FI document category (for example, customer invoice type).
  - `BLDAT`: document date | Original invoice/document date.
  - `BUDAT`: posting date | Date posted to the ledger, relevant for accounting periods.
  - `FAEDT`: net due date | Contractual payment due date used for delinquency tracking.
  - `AUGDT`: clearing date (null when open) | Actual settlement date; null indicates open receivable.
  - `DMBTR`: amount in local currency | Invoice amount used for cash and exposure calculations.
  - `WAERS`: currency | Currency key for monetary values.
  - `SHKZG`: debit/credit indicator | Accounting sign/type marker (`S` here denotes receivable-side posting).
  - `ZTERM`: payment terms key | Contracted terms carried to invoice-level behavior analysis.

### `ar_aging_fbl5n.csv` (FBL5N-like snapshot export)

- Grain: one row per customer for a key date
- Key columns:
  - `KUNNR`: customer number | Customer-level aging rollup key.
  - `BUKRS`: company code | Company/entity context for the aging snapshot.
  - `STIDA`: key date | Snapshot date used to calculate aging bucket placement.
  - `WAERS`: currency | Currency of aging balances.
  - `Current`: current bucket | Open amount not yet overdue as of `STIDA`.
  - `1-30 days`: overdue bucket | Amount overdue by 1 to 30 days.
  - `31-60 days`: overdue bucket | Amount overdue by 31 to 60 days.
  - `61-90 days`: overdue bucket | Amount overdue by 61 to 90 days.
  - `91-180 days`: overdue bucket | Amount overdue by 91 to 180 days.
  - `180+ days`: overdue bucket | Severely overdue amount above 180 days.
  - `TOTAL_OUTSTANDING`: total open AR | Sum of all aging buckets for the customer.

### `gl_cash_positions.csv` (FAGLFLEXT-like)

- Grain: one row per account and fiscal period
- Key columns:
  - `RBUKRS`: company code | Legal entity owning the GL balances.
  - `RACCT`: GL account | Cash/bank account code used in treasury/cash-position context.
  - `GJAHR`: fiscal year | Fiscal year for period balance reporting.
  - `POPER`: posting period | Fiscal period (month-style period key).
  - `HSL`: local currency balance | Account balance amount used for cash-context features.
  - `WAERS`: currency | Currency key for GL balances.

## Derived Later (Not Raw)

These are intentionally not stored in raw exports:

- `STATUS` (`OPEN`/`CLEARED`) from `AUGDT`
- `DAYS_LATE` from `AUGDT - FAEDT`
- `LATE_FLAG` from `DAYS_LATE > 0`
- model-facing segment labels (`INDUSTRY`, `REGION`) mapped from SAP keys
- engineered features (`CUST_*`, `LOG_*`, calendar features)

## Validation

Run:

```bash
python -m src.data.validator
```

This validates raw SAP-style schema, value domains, numeric ranges, and referential integrity.
