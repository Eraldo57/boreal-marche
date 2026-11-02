# Boréal Marché — Customer Intelligence Platform

Vanier College capstone, Eraldo Ibrushi. **Progress milestone: integration and validation**, not a production launch.

One public-data pipeline supports retention prediction and basket recommendations. A separate fictional policy/product corpus supports the assistant. FastAPI exposes all three heads; Streamlit calls the API.

## Start here

- [Reproduction runbook](reports/current_run/RUNBOOK.md): setup, pipeline, servers and checks.
- [Executed validation](reports/current_run/VALIDATION_RESULTS.md): real results and their limitations.
- [Remaining work](reports/current_run/REMAINING_ISSUES.md): prioritized, including the decisions that wait for the project owner.
- [Student technical guide](reports/STUDENT_TECHNICAL_GUIDE.md): module-by-module explanation.
- [Architecture](ARCHITECTURE.md), [model card](MODEL_CARD.md), [data card](DATA_CARD.md).
- Governance:
  - [Law 25 record](GOVERNANCE_LOI25.md) and [automated-decision register](governance/decision_register.md).
  - [Fairness audit](governance/fairness_audit.md) and [feature inventory with the minimisation audit](governance/FEATURE_INVENTORY.md).
  - [Promotion policy](governance/PROMOTION_POLICY.md).
  - Customer notice in [English](governance/customer_notice_en.md) and [French](governance/customer_notice_fr.md).
- Operations:
  - [Containers](docs/CONTAINERS.md): Dockerfile choices, measured image sizes, Compose stack.
  - [Monitoring and alerts](docs/MONITORING.md), including the delayed-label backfill design.
  - [SageMaker](docs/SAGEMAKER.md): built and run locally, not yet on AWS.

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
.venv/bin/streamlit run frontend/streamlit_app.py --server.address 127.0.0.1
```

Open http://127.0.0.1:8501 and http://127.0.0.1:8000/docs. Use historical customer **12347**, then “Load historical customer features.” Change any value and the probability updates: that is the what-if. A reviewer then approves or rejects the model's proposal; nothing is ever sent. These are local demo endpoints with no authentication; do not expose them publicly.

## Docker quickstart

After the pipeline above has produced `artifacts/` and `data/`, one command starts the API, the Streamlit interface, PostgreSQL (prediction log) and Ollama (local LLM):

```bash
make up      # docker compose -f docker/compose.yml up --build -d; stop with: make down
```

Same URLs as above. On macOS, Docker Desktop needs permission to read the folder if the repository is under Desktop, Documents or Downloads. Prerequisites, download sizes and measured image sizes: [docs/CONTAINERS.md](docs/CONTAINERS.md).

## Evidence from the current local run

The model, rules and index were measured on the real UCI workbook on 2026-09-30. The audits, provider evaluation and operations were measured on 2026-11-01. Detailed protocols and JSON sources are in the validation report.

| Component | Executed evidence |
|---|---|
| Data | 1,067,371 raw rows → 820,645 cleaned event rows, including returns |
| Retention | 25 features; chronological holdout AP 0.7430 versus recency baseline 0.6711; isotonic calibration (reliability chart in reports/current_run/figures/) |
| Fairness and minimisation | No region flagged. Five strong recall flags, all customers with a thin purchase history. 13 of 25 features show no reliable contribution (recommendation pending) |
| Recommendations | 3,567 rules; later-basket Hit-rate@5 0.4867 versus popularity 0.1723 |
| Assistant | Tool choice on 40 golden questions: rules 97.5%, Ollama `llama3.2:3b` 80%. Ollama answers stream in the requested language, with a figure check; OpenAI not evaluated |
| Integration | API smoke 16/16 and interface smoke 6/6, including the live what-if and a streamed answer |
| Operations | MLflow tracking and registry (version 1 a candidate, not promoted). Monitoring: drift, data quality and rule fire rate, with automated alerts. Docker stack verified locally; no cloud deployment |

Average precision here measures **will_return**, not churn. Recommendation evaluation masks half of each eligible later basket; it is not an online sales experiment. Repeated training runs reproduce the same holdout, not independent evidence.

## Current limitations

- **Local gates pass, but CI has never run.** 273 tests; coverage 90% locally and 89% on a clean checkout, against the 80% gate; strict mypy with 0 errors. CI has not run because the repository has no remote yet.
- **No cloud.** The images were built and the Compose stack run locally on linux/arm64 only. Still to do: the linux/amd64 build, a fresh clone on another machine, CD, SageMaker on AWS and any cloud deployment.
- **No champion yet.** Promoting one needs the project owner as approver.
- **Decisions waiting for the owner.** Retraining without the 13 unnecessary features, a fairness mitigation, and the business numbers behind the threshold.
- **Model-written answers are a demonstration, not a reliable channel.** The 3B model can contradict its source; the figure check flags such cases.
- **Not built:** authenticated access and deletion enforcement.

## Provenance and safe use

[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) is a public UK giftware dataset, re-skinned as fictional Boréal Marché. Original monetary values remain **GBP**, never silently CAD. The assistant's product lookup reads `artifacts/product_catalogue.csv`, exported from the same cleaned purchases by `python -m src.catalogue` (one row per StockCode, median observed unit price in GBP). The data has no categories, allergens or dietary tags, so the assistant cannot answer product questions about them.

Governance documents describe actual controls and planned work; this repository makes **no Law 25 compliance certification**. Load only trusted joblib artifacts: serialization is executable, and a hash is an integrity check, not a trust guarantee.
