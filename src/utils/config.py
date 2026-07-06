"""
Central configuration module.

All runtime config is loaded from purpose-specific YAML files in `config/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

_config_dir_env = os.getenv("CFI_CONFIG_DIR")
CONFIG_DIR = Path(_config_dir_env) if _config_dir_env else ROOT_DIR / "config"
if not CONFIG_DIR.is_absolute():
    CONFIG_DIR = ROOT_DIR / CONFIG_DIR
CONFIG_DIR = CONFIG_DIR.resolve()


def _read_yaml(filename: str) -> dict[str, Any]:
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping at top level: {path}")
    return data


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing key '{key}' in {context}")
    return mapping[key]


def _resolve_from_root(path_value: str) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else ROOT_DIR / p


# Paths and files
_paths_cfg = _read_yaml("paths.yaml")
_dirs_cfg = _require(_paths_cfg, "directories", "paths.yaml")
_files_cfg = _require(_paths_cfg, "files", "paths.yaml")

DATA_DIR = _resolve_from_root(str(_require(_dirs_cfg, "data", "paths.yaml:directories")))
RAW_DIR = _resolve_from_root(str(_require(_dirs_cfg, "raw", "paths.yaml:directories")))
PROCESSED_DIR = _resolve_from_root(str(_require(_dirs_cfg, "processed", "paths.yaml:directories")))
EXPORTS_DIR = _resolve_from_root(str(_require(_dirs_cfg, "exports", "paths.yaml:directories")))
MODELS_DIR = _resolve_from_root(str(_require(_dirs_cfg, "models", "paths.yaml:directories")))
NOTEBOOKS_DIR = _resolve_from_root(str(_require(_dirs_cfg, "notebooks", "paths.yaml:directories")))

_raw_files = _require(_files_cfg, "raw", "paths.yaml:files")
_processed_files = _require(_files_cfg, "processed", "paths.yaml:files")
_model_files = _require(_files_cfg, "models", "paths.yaml:files")

CUSTOMER_MASTER_PATH = RAW_DIR / str(_require(_raw_files, "customer_master", "paths.yaml:files.raw"))
INVOICE_HISTORY_PATH = RAW_DIR / str(_require(_raw_files, "invoice_history", "paths.yaml:files.raw"))
AR_AGING_PATH = RAW_DIR / str(_require(_raw_files, "ar_aging", "paths.yaml:files.raw"))
GL_CASH_PATH = RAW_DIR / str(_require(_raw_files, "gl_cash", "paths.yaml:files.raw"))

FEATURE_STORE_PATH = PROCESSED_DIR / str(_require(_processed_files, "feature_store", "paths.yaml:files.processed"))
TRAIN_PATH = PROCESSED_DIR / str(_require(_processed_files, "train", "paths.yaml:files.processed"))
TEST_PATH = PROCESSED_DIR / str(_require(_processed_files, "test", "paths.yaml:files.processed"))

CLASSIFIER_PATH = MODELS_DIR / str(_require(_model_files, "classifier", "paths.yaml:files.models"))
FORECASTER_PATH = MODELS_DIR / str(_require(_model_files, "forecaster", "paths.yaml:files.models"))
SHAP_EXPLAINER_PATH = MODELS_DIR / str(_require(_model_files, "shap_explainer", "paths.yaml:files.models"))

for _dir in [RAW_DIR, PROCESSED_DIR, EXPORTS_DIR, MODELS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


class DataConfig(BaseModel):
    n_customers: int = 200
    n_invoices: int = 5000
    start_date: str = "2021-01-01"
    end_date: str = "2024-06-30"
    random_seed: int = 42
    payment_terms: dict[str, int] = {
        "NT14": 14,
        "NT30": 30,
        "NT45": 45,
        "NT60": 60,
        "NT90": 90,
    }
    industries: list[str] = [
        "Energy",
        "Manufacturing",
        "Retail",
        "Healthcare",
        "Construction",
        "Logistics",
    ]
    regions: list[str] = [
        "South",
        "Northeast",
        "Midwest",
        "West",
        "International",
    ]
    aging_buckets: list[str] = [
        "Current",
        "1-30 days",
        "31-60 days",
        "61-90 days",
        "91-180 days",
        "180+ days",
    ]
    risk_weights: list[float] = [0.55, 0.30, 0.15]
    payment_delay_map: dict[str, dict[str, int]] = {
        "Low": {"mean": 2, "std": 3},
        "Medium": {"mean": 12, "std": 8},
        "High": {"mean": 28, "std": 15},
    }


class ClassifierConfig(BaseModel):
    target_col: str = "LATE_FLAG"
    test_size: float = 0.2
    random_seed: int = 42
    cv_folds: int = 5
    calibration: dict[str, Any] = {
        "enabled": True,
        "method": "sigmoid",
    }
    xgb_params: dict[str, Any] = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "scale_pos_weight": 2,
        "eval_metric": "auc",
        "early_stopping_rounds": 50,
        "random_state": 42,
    }
    feature_cols: list[str] = [
        "LOG_AMOUNT",
        "PAYMENT_DAYS_NUM",
        "INDUSTRY_CODE",
        "REGION_CODE",
        "CUST_AVG_DAYS_LATE",
        "CUST_LATE_RATE",
        "CUST_INVOICE_COUNT",
        "CUST_AVG_AMOUNT",
        "INVOICE_MONTH",
        "INVOICE_QUARTER",
        "INVOICE_DOW",
        "IS_QUARTER_END",
    ]


class ForecasterConfig(BaseModel):
    forecast_horizons: list[int] = [30, 60, 90]
    quantiles: list[float] = [0.1, 0.5, 0.9]
    random_seed: int = 42
    segment_column: str = "REGION"
    min_months_for_segment_model: int = 24
    min_cleared_invoices_for_segment_model: int = 250
    min_rows_for_forecast_dataset: int = 12
    lgbm_params: dict[str, Any] = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "random_state": 42,
    }


class DashboardConfig(BaseModel):
    app_title: str = "CFO Cash Flow Intelligence"
    app_icon: str = "$"
    default_horizon: int = 30
    risk_threshold: float = 0.6
    top_n_risks: int = 20


DATA_CFG = DataConfig.model_validate(_read_yaml("data_generation.yaml"))
CLASSIFIER_CFG = ClassifierConfig.model_validate(_read_yaml("classifier.yaml"))
FORECASTER_CFG = ForecasterConfig.model_validate(_read_yaml("forecaster.yaml"))
DASHBOARD_CFG = DashboardConfig.model_validate(_read_yaml("dashboard.yaml"))
