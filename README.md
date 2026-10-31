# Boréal Marché — Customer Intelligence Platform

Vanier College capstone, Eraldo Ibrushi. **Progress milestone: integration and validation**, not a production launch.

One public-data pipeline supports retention prediction and basket recommendations. A separate fictional policy/product corpus supports the assistant. FastAPI exposes all three heads; Streamlit calls the API.

## Start here

- [Reproduction runbook](reports/current_run/RUNBOOK.md): setup, pipeline, servers and checks.
- [Executed validation](reports/current_run/VALIDATION_RESULTS.md): real results and their limitations.
- [Remaining work](reports/current_run/REMAINING_ISSUES.md): prioritized, including failing quality gates.
- [Student technical guide](reports/STUDENT_TECHNICAL_GUIDE.md): module-by-module explanation.
- [Architecture](ARCHITECTURE.md), [canonical feature inventory](governance/FEATURE_INVENTORY.md).
- [Containers](docs/CONTAINERS.md): Dockerfile choices, measured image sizes, Compose stack.

## Local quickstart

Use Python 3.12 (validated); Python 3.14 is not supported. Docker and paid LLM credentials are not needed.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m src.ingest
.venv/bin/python -m src.features
.venv/bin/python -m src.train
.venv/bin/python -m src.rules
.venv/bin/python -m src.catalogue
.venv/bin/python -m src.rag.index
.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
.venv/bin/streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

Open http://127.0.0.1:8501 and http://127.0.0.1:8000/docs. Use historical customer **12347**, then “Load historical customer features.” These are local demo endpoints with no authentication; do not expose them publicly.

## Docker quickstart

After the pipeline above has produced `artifacts/` and `data/`, one command starts the API, the Streamlit interface, PostgreSQL (prediction log) and Ollama (local LLM):

```bash
make up      # docker compose -f docker/compose.yml up --build -d; stop with: make down
```

Same URLs as above. On macOS, Docker Desktop needs permission to read the folder if the repository is under Desktop, Documents or Downloads. Prerequisites, download sizes and measured image sizes: [docs/CONTAINERS.md](docs/CONTAINERS.md).

## Evidence from the current local run

Measured on the real UCI workbook on 2026-09-30; detailed protocols and JSON sources are in the validation report.

| Component | Executed evidence |
|---|---|
| Data | 1,067,371 raw rows → 820,645 cleaned event rows, including returns |
| Retention | 25 features; chronological holdout AP 0.7430 versus recency baseline 0.6711 |
| Recommendations | 3,567 rules; later-basket Hit-rate@5 0.4867 versus popularity 0.1723 |
| Assistant | Real multilingual index; regex golden-set evaluation, with source-selection gaps |
| Integration | Live HTTP requests and Streamlit widget interactions passed |
| Operations | Local MLflow tracking and measured demo PSI/KS; no cloud deployment |

Average precision here measures **will_return**, not churn. Recommendation evaluation masks half of each eligible later basket; it is not an online sales experiment. Repeated training runs reproduce the same holdout, not independent evidence.

## Current limitations

Local quality gates pass (strict mypy with 0 errors; coverage 90% locally and 89% on a clean checkout, against the 80% gate), but no CI run has happened because the repository has no remote yet. Both images were built and the Compose stack was run locally on linux/arm64; the linux/amd64 build, a fresh clone on another machine and any cloud deployment have not been done. The OpenAI provider was not executed; the Ollama path ran in Compose with the configured `llama3.2:3b` on three questions only, which is a wiring check, not an evaluation. Fluent French generation, authenticated customer access, full fairness evaluation, deletion enforcement, human approval, registry promotion and cloud operations remain unfinished.

## Provenance and safe use

[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) is a public UK giftware dataset, re-skinned as fictional Boréal Marché. Original monetary values remain **GBP**, never silently CAD. The assistant's product lookup reads `artifacts/product_catalogue.csv`, exported from the same cleaned purchases by `python -m src.catalogue` (one row per StockCode, median observed unit price in GBP). The data has no categories, allergens or dietary tags, so the assistant cannot answer product questions about them.

Governance documents describe actual controls and planned work; this repository makes **no Law 25 compliance certification**. Load only trusted joblib artifacts: serialization is executable, and a hash is an integrity check, not a trust guarantee.
