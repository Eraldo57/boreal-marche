# Reproduce the local milestone

Run from the repository root. Python **3.12** was used; allow disk space for workbook, Parquet, models, embedding cache, DVC and MLflow. Internet is required on the first dataset/embedding download. No paid provider is needed for the regex path.

## Environment

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

The executed package snapshot is environment_freeze.txt. It documents the tested machine, not a cross-platform wheel lock. MLflow 2.x requires SQLAlchemy <2.1 here. The raw checksum is measured/pinned, not publisher-signed; see SOURCE_VERIFICATION.md. Do not disable verification if the upstream file changes.

## Generate artifacts in order

```bash
.venv/bin/python -m scripts.run_logged ingest .venv/bin/python -m src.ingest
.venv/bin/python -m scripts.run_logged features .venv/bin/python -m src.features
.venv/bin/python -m scripts.run_logged training .venv/bin/python -m src.train
.venv/bin/python -m scripts.run_logged rules .venv/bin/python -m src.rules
.venv/bin/python -m scripts.run_logged index .venv/bin/python -m src.rag.index
```

The wrapper records exact command, UTC timestamp, elapsed time and exit code in logs/commands.jsonl, with stage output in logs/<stage>.log. A repeated stage name replaces that stage's log; choose unique names when preserving multiple attempts. The ledger contains both failed attempts and successes. Initial setup/direct-download commands are documented in SOURCE_VERIFICATION.md and TESTS_BEFORE.md.

Training uses 12 trials per top tunable family. A faster diagnostic is `--trials 0`, but that creates different artifacts and is not the reported tuned run. Reports are local JSON, not invented README values. Do not alter tuning after viewing the holdout and call the same holdout untouched.

Indexing downloads paraphrase-multilingual-MiniLM-L12-v2 into data/model_cache. First download can take several minutes. Use `--force` only to rebuild this project's index after corpus/model changes. API startup never builds the index.

## Start the main path

Terminal 1:

```bash
.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```bash
API_BASE_URL=http://127.0.0.1:8000 .venv/bin/streamlit run frontend/streamlit_app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

Open http://127.0.0.1:8000/docs and http://127.0.0.1:8501. /health should be 200 even if artifact loading fails; /ready is 503 with useful errors until all required components load. Restart API after changing artifacts. Never load an untrusted joblib file.

## Execute validation

With both servers running:

```bash
.venv/bin/python -m scripts.run_logged api_smoke .venv/bin/python -m scripts.smoke_api
.venv/bin/python -m scripts.run_logged ui_smoke .venv/bin/python -m scripts.smoke_ui
.venv/bin/python -m scripts.run_logged assistant .venv/bin/python -m scripts.evaluate_assistant
.venv/bin/python -m scripts.run_logged monitoring .venv/bin/python -m scripts.simulate_traffic --count 120 --shifted
.venv/bin/python -m scripts.run_logged tests .venv/bin/coverage run --source=src,api -m pytest -q
.venv/bin/coverage report --fail-under=80
.venv/bin/ruff check src api app scripts tests
.venv/bin/ruff format --check src api app scripts tests
.venv/bin/mypy src api --ignore-missing-imports
```

On 2026-11-01 these gates passed locally: 190 tests, coverage 90% (89% on a clean checkout, where 5 integration tests skip), ruff clean, mypy 0 errors in 22 files. No CI run has happened yet. Integration tests skip on a fresh checkout until local artifacts exist; unit tests use labelled mocks or labelled synthetic data. UI smoke is AppTest plus server health, not browser visual QA. Monitoring traffic is deliberately synthetic/historical, not a production benchmark.

Assistant external providers are optional: configure OPENAI_API_KEY or start the configured Ollama server/model, then use evaluate_assistant --provider openai|ollama. No evaluation of either was executed. The Ollama request path was exercised inside Compose with the configured `llama3.2:3b` on three questions (docs/CONTAINERS.md); that is wiring, not an evaluation. Do not send private information to this demo.

## Tracking, DVC and Docker

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
.venv/bin/python -m scripts.retrain --dry-run --skip-etl
.venv/bin/dvc status
docker compose -f docker/compose.yml config --quiet
```

MLflow SQLite tracking and nested comparison/tuning runs exist; registry aliases do not. Retrain without --dry-run generates a timestamped candidate and its own report, never automatically promotes it. The same historical holdout is not fresh promotion evidence.

DVC was initialized locally with `dvc init --no-scm`, analytics disabled, then `dvc commit -f` recorded existing outputs in dvc.lock. This records artifacts; it is not evidence of a full `dvc repro` execution. Since 2026-11-01 `.dvc/config` sets only `analytics = false`; without `no_scm`, DVC works with git. No remote is configured. Chroma mutates its persistence directory on use, so DVC can report the index modified even after read queries. Stop services and design an immutable index export before claiming a clean reusable data registry.

Docker: generate host artifacts first (the API mounts artifacts/ and data/), then one command starts API, interface, PostgreSQL and Ollama:

```bash
make up        # docker compose -f docker/compose.yml up --build -d
make down
```

The interface uses http://api:8000 inside Compose; MLflow stays behind the optional `tracking` profile. On Linux set LOCAL_UID/LOCAL_GID in .env. Both images were built and the stack was run on 2026-10-31 (linux/arm64); measured sizes, the macOS folder-permission prerequisite and what was not verified are in docs/CONTAINERS.md. No AWS command was executed.

## Five-minute screen share

1. 0:00–0:45: architecture and configs/pipeline.yaml; distinguish UCI events from fictional policy corpus.
2. 0:45–1:30: manifest + cleaning_report; show preserved cancellations and fixed caps.
3. 1:30–2:15: temporal cutoffs, canonical features, retention_metrics and real model metadata.
4. 2:15–3:00: recommendation_metrics and rules header; explain popularity comparison.
5. 3:00–4:15: API /ready and Streamlit customer 12347 → Predict; one basket; English/French policy citations.
6. 4:15–5:00: test results, remaining issues and next-phase slide. Explain incomplete source relevance, fairness, CI gates and cloud work.

Do not present notebooks or DEFENCE_PREPARATION.md as current validated evidence. Stop servers with Ctrl-C when finished.

