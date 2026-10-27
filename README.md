# Boréal Marché — Customer Intelligence Platform

Vanier College capstone, Eraldo Ibrushi. **Progress milestone: integration and validation**, not a production launch.

One public-data pipeline supports retention prediction and basket recommendations. A separate fictional policy/product corpus supports the assistant. FastAPI exposes all three heads; Streamlit calls the API.

## Start here

- [Reproduction runbook](reports/current_run/RUNBOOK.md): setup, pipeline, servers and checks.
- [Executed validation](reports/current_run/VALIDATION_RESULTS.md): real results and their limitations.
- [Remaining work](reports/current_run/REMAINING_ISSUES.md): prioritized, including failing quality gates.
- [Student technical guide](reports/STUDENT_TECHNICAL_GUIDE.md): module-by-module explanation.
- [Architecture](ARCHITECTURE.md), [canonical feature inventory](governance/FEATURE_INVENTORY.md).

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

Strict mypy and coverage gates still fail. Docker configuration parses, but the Docker daemon was unavailable for building/running containers. OpenAI/Ollama integrations exist but were not executed. Fluent French generation, authenticated customer access, full fairness evaluation, deletion enforcement, human approval, registry promotion and cloud operations remain unfinished.

## Provenance and safe use

[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) is a public UK giftware dataset, re-skinned as fictional Boréal Marché. Original monetary values remain **GBP**, never silently CAD. The assistant's product lookup reads `artifacts/product_catalogue.csv`, exported from the same cleaned purchases by `python -m src.catalogue` (one row per StockCode, median observed unit price in GBP). The data has no categories, allergens or dietary tags, so the assistant cannot answer product questions about them.

Governance documents describe actual controls and planned work; this repository makes **no Law 25 compliance certification**. Load only trusted joblib artifacts: serialization is executable, and a hash is an integrity check, not a trust guarantee.
