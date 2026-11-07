# Complete project submission

The complete ZIP combines the editable presentation and all files supporting the Streamlit
demonstration, including the AWS/SageMaker implementation. Extract the whole folder first.

The project is now also deployed on Google Cloud: [Streamlit](https://boreal-ui-qaonyx3dda-nn.a.run.app)
and [API documentation](https://boreal-api-qaonyx3dda-nn.a.run.app/docs). Google Cloud replaces
the assignment's AWS services at the owner's request. Current deployment status and execution
evidence are in `project/docs/GOOGLE_CLOUD.md` and `project/reports/current_run/google_cloud/`.
The November 5 reports and presentations are dated scientific/local verification snapshots;
use the Google Cloud execution report for the subsequent deployment status.

| Material | Location in the ZIP |
|---|---|
| Start guide and file checksums | `START_HERE.md`, `MANIFEST.json`, `SHA256SUMS.txt` |
| Editable 20-slide defence and printable deck | `project/deliverables/Boreal_Marche_Defence.pptx`, `.pdf` |
| Speaker notes and defence cue card | `project/deliverables/Defence_Speaker_Notes.md`, `Defence_Cue_Card.md` |
| Technical report, bilingual executive summary and reflection draft | `project/deliverables/` (Word and PDF) |
| Streamlit screens, API and supporting code | `project/frontend/`, `project/api/`, `project/src/` |
| Raw workbook, processed data, trained model, basket rules and product catalogue | `project/data/`, `project/artifacts/` |
| Assistant policy corpus, retrieval index and cached embedding weights | `project/docs/corpus/`, `project/data/chromadb/`, `project/data/model_cache/` |
| Evaluation figures, notebooks, tests and execution evidence | `project/reports/`, `project/notebooks/`, `project/tests/` |
| AWS/SageMaker, containers and deployment workflows | `AWS_READ_ME.md`, `project/docs/SAGEMAKER.md`, `project/docker/`, `project/.github/workflows/` |
| Google Cloud runbook, source and execution evidence | `GOOGLE_CLOUD_READ_ME.md`, `project/docs/GOOGLE_CLOUD.md`, `project/scripts/gcp.py`, `project/scripts/vertex_jobs.py`, `project/reports/current_run/google_cloud/` |
| Governance, notices, model/data cards and architecture | `project/governance/`, `project/GOVERNANCE_LOI25.md`, `project/*_CARD.md`, `project/ARCHITECTURE.md` |
| Original assignment and earlier presentations, when available locally | `assignment_brief/`, `historical_presentations/` |

All five Streamlit tabs are covered: Prediction (features, model, explanations and coupon
scenario), Recommendations (rules, catalogue and mining diagnostics), Assistant (tools,
corpus, index and provider configuration), Monitoring (rules, logs and captured metrics),
and About (notices, evaluation plots, data provenance, architecture and version).
`project/reports/current_run/submission_snapshot_20261105/` contains read-only responses
from the running local API, including readiness, version, monitoring and the feature contract.
These are dated evidence; live results still require running the application.

## Rebuild the complete folder and ZIP

From the repository root, run:

```bash
python3 scripts/package_submission.py
```

The output is in the sibling `submissions/2026-11-06/` directory so that the ZIP cannot
recursively include itself. The packager checks required files, scans for credentials,
backs up SQLite databases, materializes cached model symlinks for portability, and verifies
every ZIP member against the manifest. The ZIP and its checksum are intended as GitHub
Release assets; large cached model weights are kept out of Git history.

## Run the included Streamlit demonstration

Use Python 3.12. In the extracted `project/` directory:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
cp .env.example .env
MODEL_SOURCE=local .venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In another terminal, from that same directory:

```bash
.venv/bin/streamlit run frontend/streamlit_app.py --server.address 127.0.0.1
```

Open http://127.0.0.1:8501. Use historical demo customer **12347** and the **regex** assistant
provider for a demonstration without paid LLM credentials. Included artifacts avoid having
to train or rebuild the index first. Installing Python dependencies still needs network access.
OpenAI needs your own key; Ollama needs its own server and model download. Docker images
are rebuilt from the included Dockerfile rather than shipped as multi-gigabyte image exports.
Historical MLflow records can contain paths from the original machine; `MODEL_SOURCE=local`
serves the included current artifact directly. See `reports/current_run/RUNBOOK.md` for details.

## AWS material and submission status

AWS material means the SageMaker launcher, container entry point, serving endpoint,
Docker targets, configuration template, tests, deployment workflow and recorded local checks.
No live AWS job is claimed. Google Cloud hosting and its separate execution records are
included as the owner-selected provider substitution. This packaging step does not create
resources. The AWS guide remains available for reproducing the original provider approach.

The reflection remains a draft requiring Eraldo's review. Current verification and remaining
requirements are recorded in `reports/current_run/COMPLETION_VERIFICATION_20261105.md`
and `docs/CAPSTONE_REQUIREMENTS_STATUS.md`. Existing dated reports and logs preserve
historical results, including superseded model versions.

Credentials, `.env`, Git internals, Python environments, redundant caches and editor/runtime
temporary files are excluded. `.env.example` is included. `EXCLUSIONS.json` lists exclusions.
