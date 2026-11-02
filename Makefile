# =============================================================================
# Boréal Marché — Project Automation
# =============================================================================
# Targets required by Section 7.2:
#   setup · etl · rules · spotcheck · tune · train · index · test · up
# =============================================================================

.DEFAULT_GOAL := help
PYTHON ?= .venv/bin/python
VENV := .venv

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
.PHONY: setup
setup: ## Create venv and install all dependencies
	python3.12 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev]"

# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------
.PHONY: etl
etl: ## Download, validate, clean, and write Parquet
	$(PYTHON) -m src.ingest

.PHONY: features
features: ## Engineer 25 features and create labels
	$(PYTHON) -m src.features

# ---------------------------------------------------------------------------
# Head A — Retention
# ---------------------------------------------------------------------------
.PHONY: spotcheck
spotcheck: ## Chronological comparison + calibration, no tuning trials
	$(PYTHON) -m src.train --trials 0

.PHONY: tune
tune: ## Full chronological pipeline with configured Optuna budget
	$(PYTHON) -m src.train

.PHONY: train
train: ## Full pipeline: spot-check → tune → calibrate → threshold → save
	$(PYTHON) -m src.train

.PHONY: figures
figures: ## Redraw the reliability and threshold figures from the saved retention report
	$(PYTHON) -m src.plots

.PHONY: fairness
fairness: ## Subgroup fairness audit of the saved model on the test snapshot
	$(PYTHON) -m src.fairness

# ---------------------------------------------------------------------------
# Head B — Recommendations
# ---------------------------------------------------------------------------
.PHONY: rules
rules: ## Mine association rules (Apriori demo + FP-Growth prod)
	$(PYTHON) -m src.rules

.PHONY: catalogue
catalogue: ## Export the product catalogue from cleaned purchases
	$(PYTHON) -m src.catalogue

# ---------------------------------------------------------------------------
# Head C — Assistant
# ---------------------------------------------------------------------------
.PHONY: index
index: ## Build the RAG document index (ChromaDB)
	$(PYTHON) -m src.rag.index --force

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
.PHONY: test
test: ## Run full test suite with coverage
	$(PYTHON) -m coverage run -m pytest tests/
	$(PYTHON) -m coverage report --fail-under=80

.PHONY: lint
lint: ## Lint and format check
	$(PYTHON) -m ruff check src/ api/ frontend/ tests/ scripts/
	$(PYTHON) -m ruff format --check src/ api/ frontend/ tests/ scripts/

.PHONY: typecheck
typecheck: ## Static type checking
	$(PYTHON) -m mypy src/ api/ --ignore-missing-imports

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
# Compose reads ${VAR} settings (ports, POSTGRES_PASSWORD, OLLAMA_MODEL) from .env when present.
COMPOSE := docker compose -f docker/compose.yml $(if $(wildcard .env),--env-file .env,)

.PHONY: up
up: ## Start full stack (API + Streamlit + PostgreSQL + Ollama) via Docker Compose
	GIT_SHA=$$(git rev-parse HEAD 2>/dev/null || echo unknown) $(COMPOSE) up --build -d

.PHONY: down
down: ## Stop all containers (named volumes, e.g. the database, are kept)
	$(COMPOSE) down

.PHONY: api
api: ## Start the API locally (dev mode)
	$(PYTHON) -m uvicorn api.main:app --reload --port 8000

.PHONY: app
app: ## Start the Streamlit frontend locally
	$(PYTHON) -m streamlit run frontend/streamlit_app.py --server.port 8501

# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
.PHONY: all
all: etl features train rules catalogue index test ## Run everything end to end

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
.PHONY: clean
clean: ## Remove generated artifacts (keeps raw data)
	rm -rf artifacts/*.joblib artifacts/*.json
	rm -rf data/processed/*.parquet
	rm -rf data/chromadb/
	rm -rf mlruns/ mlartifacts/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
