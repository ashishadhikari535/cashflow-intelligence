# Cash Flow Intelligence System - Makefile
#
# Default usage:
#   make setup
#   make data
#   make train
#   make explain
#   make dashboard
#   make test
#   make clean
#   make all
#
# Notes:
# - This Makefile is conda-first and cross-platform friendly.
# - Override env with: make CONDA_ENV=<your_env> <target>
# - Enforce strict validator exit with: make VALIDATOR_STRICT=1 data

.PHONY: setup data train explain dashboard test clean all

CONDA_ENV ?= cashflow-intelligence
VALIDATOR_STRICT ?= 0

PYTHON ?= conda run -n $(CONDA_ENV) python
PIP ?= $(PYTHON) -m pip
STREAMLIT ?= conda run -n $(CONDA_ENV) streamlit

DATA_DIR = data
MODELS_DIR = models_saved
SRC_DASHBOARD = src/dashboard

define RUN_MODULE_AGG
$(PYTHON) -c "import os,runpy; os.environ.setdefault('MPLBACKEND','Agg'); runpy.run_module('$(1)', run_name='__main__')"
endef

setup:
	@echo "-> Installing dependencies in conda env: $(CONDA_ENV)"
	$(PIP) install -r requirements.txt
	$(PYTHON) -c "from pathlib import Path; [Path(p).mkdir(parents=True, exist_ok=True) for p in ['$(DATA_DIR)/raw','$(DATA_DIR)/processed','$(DATA_DIR)/exports','$(MODELS_DIR)']]"
	@echo "OK: setup complete"

data:
	@echo "-> Generating SAP-structured synthetic data..."
	$(PYTHON) -m src.data.generator
	@echo "-> Validating data quality..."
	$(PYTHON) -c "import subprocess,sys; strict=int('$(VALIDATOR_STRICT)'); rc=subprocess.call([sys.executable,'-m','src.data.validator']); print(f'Validator exit code: {rc} (VALIDATOR_STRICT={strict})'); sys.exit(rc if strict else 0)"
	@echo "-> Building feature store..."
	$(PYTHON) -m src.data.features
	@echo "OK: data pipeline complete"

train:
	@echo "-> Training XGBoost late payment classifier..."
	$(PYTHON) -m src.models.classifier
	@echo "-> Training LightGBM cash flow forecaster..."
	$(call RUN_MODULE_AGG,src.models.forecaster)
	@echo "-> Evaluating models..."
	$(call RUN_MODULE_AGG,src.models.evaluator)
	@echo "OK: training complete (artifacts in $(MODELS_DIR)/)"

explain:
	@echo "-> Running SHAP explainability..."
	$(call RUN_MODULE_AGG,src.models.explainer)
	@echo "OK: explainability complete"

dashboard:
	@echo "-> Launching Streamlit dashboard..."
	$(STREAMLIT) run $(SRC_DASHBOARD)/app.py

test:
	@echo "-> Running test suite..."
	$(PYTHON) -m pytest tests/ -v --tb=short
	@echo "OK: tests complete"

clean:
	@echo "-> Cleaning generated files..."
	$(PYTHON) -c "from pathlib import Path; import shutil; \
[p.unlink() for p in Path('$(DATA_DIR)/raw').glob('*.csv') if p.exists()]; \
[p.unlink() for p in Path('$(DATA_DIR)/processed').glob('*.csv') if p.exists()]; \
[shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True) for p in Path('$(DATA_DIR)/exports').glob('*')]; \
[shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True) for p in Path('$(MODELS_DIR)').glob('*')]; \
[shutil.rmtree(p, ignore_errors=True) for p in Path('.').rglob('__pycache__')]; \
[p.unlink(missing_ok=True) for p in Path('.').rglob('*.pyc')]; \
print('OK: clean complete')"

all: setup data train explain
	@echo ""
	@echo "OK: full pipeline complete. Run 'make dashboard' to launch."
