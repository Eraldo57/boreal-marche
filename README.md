# Boréal Marché — Customer Intelligence Platform

Vanier College capstone by Eraldo Ibrushi. A Python application for retention prediction, basket recommendations and a grounded bilingual assistant, with a Streamlit interface and FastAPI backend.

[![CI](https://github.com/Eraldo57/boreal-marche/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Eraldo57/boreal-marche/actions/workflows/ci.yml)

One public-data pipeline supports retention prediction and basket recommendations. A separate fictional policy/product corpus supports the assistant. FastAPI exposes all three heads; Streamlit calls the API.

## Project structure

| Directory | Contents |
|---|---|
| [frontend/](frontend/) | Streamlit: Prediction, Recommendations, Assistant, Monitoring and About |
| [api/](api/) | FastAPI routes, request contracts, logging, monitoring and SageMaker serving |
| [src/](src/) | Ingestion, features, model training, basket rules, retrieval and registry integration |
| [notebooks/](notebooks/) | Executed exploration, modelling and evaluation notebooks |
| [tests/](tests/) | Unit, API, UI and real-artifact integration checks |
| [docker/](docker/) | API, frontend and SageMaker image targets, plus the local Compose stack |
| [scripts/](scripts/) | Evaluation, smoke checks, retraining, SageMaker jobs and submission packaging |
| [.github/workflows/](.github/workflows/) | CI, optional AWS deployment through OIDC and scheduled retraining |
| [configs/](configs/) | Pipeline and application settings |
| [docs/](docs/), [governance/](governance/) | Architecture walkthroughs, AWS setup, reports, notices and governance records |
| [reports/](reports/) | Evaluation figures, measured results and dated verification evidence |
| [deliverables/](deliverables/) | Editable PowerPoint, speaker notes, Word reports and printable PDFs |

Clone this repository to work with the application. The [complete submission ZIP](https://github.com/Eraldo57/boreal-marche/releases/tag/submission-2026-11-05) is an additional download containing the source, documents, data and generated artifacts together. Large datasets, trained binaries and embedding weights are distributed with that bundle; the source quickstart below rebuilds them.

## Documentation

- [Defence walkthrough](docs/PRESENTATION_D7.md), [short cue card](docs/DEFENCE_CUE_CARD.md), and [Appendix C answers](docs/DEFENCE_PREPARATION.md).
- [Original-brief requirements and next actions](docs/CAPSTONE_REQUIREMENTS_STATUS.md), with [2026-11-05 local verification](reports/current_run/DEFENCE_VERIFICATION_20261105.md).
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
git clone https://github.com/Eraldo57/boreal-marche.git
cd boreal-marche
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
| Retention | 25 engineered / 12 serving features; chronological test AP 0.7596 versus recency baseline 0.6711; sigmoid calibration (reliability chart in reports/current_run/figures/) |
| Fairness and minimisation | Five strong thin-history recall flags remain; a tiny unknown-region group is inconclusive. Original 13 unnecessary features removed; contract 2.0. |
| Recommendations | 3,567 rules; later-basket Hit-rate@5 0.4867 versus popularity 0.1723 |
| Assistant | Tool choice on 40 golden questions: rules 97.5%, OpenAI `gpt-4o-mini` 95%, Ollama `llama3.2:3b` 92.5%. Of 18 model-written policy answers, OpenAI got 11 fully correct and Ollama 3 (one rater scored them). The passage with the answer reaches the model for 12 of 14 answerable questions (was 4); questions about deleting your data still fail |
| Integration | API smoke 16/16 and interface smoke 6/6, including the live what-if and a streamed answer |
| Operations | MLflow tracking and registry (version 2 a candidate, not promoted). Monitoring: drift, data quality and rule fire rate, with automated alerts. Docker stack verified locally; no cloud deployment |

Average precision here measures **will_return**, not churn. Recommendation evaluation masks half of each eligible later basket; it is not an online sales experiment. Repeated training runs reproduce the same holdout, not independent evidence.

## Current limitations

- **Local gates and published application CI pass.** On 2026-11-05, 428 tests passed across the local non-integration and artifact integration suites; combined coverage is 90% against the 80% gate; mypy reports no issues in 33 source files. [GitHub CI passed for application commit `2dda6ce`](https://github.com/Eraldo57/boreal-marche/actions/runs/42132384905). The earlier 89% clean-checkout result is historical, not a new clean-clone check.
- **Cloud execution remains pending.** Native arm64 and linux/amd64 API/frontend images were built locally, and amd64 inference was verified; see [completion verification](reports/current_run/COMPLETION_VERIFICATION_20261105.md). A fresh clone on another machine, CD, SageMaker jobs on AWS and hosted deployment remain outstanding. The deployment workflow is disabled until its AWS configuration is supplied.
- **No champion yet.** Promoting one needs the project owner as approver.
- **Decisions waiting for the owner.** A fairness mitigation and the business numbers behind the threshold. The current model has already been retrained without the 13 unnecessary features.
- **Model-written answers are a demonstration, not a reliable channel.** Both models can contradict or invent: on the French bakery-allergen question both implied allergens that no policy states. The figure check flags some cases, and retrieval still misses some passages (see VALIDATION_RESULTS.md).
- **Not built:** authenticated access and deletion enforcement.

## Provenance and safe use

[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) is a public UK giftware dataset, re-skinned as fictional Boréal Marché. Original monetary values remain **GBP**, never silently CAD. The assistant's product lookup reads `artifacts/product_catalogue.csv`, exported from the same cleaned purchases by `python -m src.catalogue` (one row per StockCode, median observed unit price in GBP). The data has no categories, allergens or dietary tags, so the assistant cannot answer product questions about them.

Governance documents describe actual controls and planned work; this repository makes **no Law 25 compliance certification**. Load only trusted joblib artifacts: serialization is executable, and a hash is an integrity check, not a trust guarantee.

## Final submission materials

See [current completion status](docs/CAPSTONE_REQUIREMENTS_STATUS.md), [final technical report](docs/FINAL_TECHNICAL_REPORT.md), [defence cue card](docs/DEFENCE_CUE_CARD.md), and [AI-use record](docs/AI_USE.md). Model 0.3.0 replaces the archived 25-feature model. Current AP is 0.7596 and threshold is 20.5%; historical run logs retain earlier results.

[Download the editable defence PowerPoint](deliverables/Boreal_Marche_Defence.pptx) · [Complete presentation package](deliverables/Boreal_Marche_Presentation_Package.zip) · [Deck PDF](deliverables/Boreal_Marche_Defence.pdf)

[Complete project submission ZIP — 513 MB](https://github.com/Eraldo57/boreal-marche/releases/download/submission-2026-11-05/Boreal_Marche_Complete_Submission.zip) · [Bundle contents and reproduction guide](SUBMISSION.md)
