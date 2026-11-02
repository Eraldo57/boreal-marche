# Executed validation results

Report generated 2026-09-30T22:40:14.370538+00:00. Sources below are executed outputs.

## Data and Head A

- Raw rows: 1,067,371; cleaned event rows: 820,645; positive purchase rows: 802,679.
- Model: RandomForestClassifier; calibration: isotonic.
- Final test rows: 5,233; AP: 0.743038; ROC-AUC: 0.780963; Brier: 0.195399.
- Recency baseline AP: 0.671134; dummy AP: 0.431339.
- Six dated snapshots; feature history strictly before cutoff; positive target in [cutoff, cutoff + 90 days). Repeat customers are allowed across periods.
- Models/tuning use December validation; calibrators fit March; calibration choice and utility threshold use June; final September holdout is scored after decisions lock.
- Monetary utility assumes cost 5 and margin 25; not observed profit or causal uplift.
- Retention ran twice successfully after an initial dependency failure. Final rerun fixed timing/report bookkeeping, not model selection based on test results. Identical held-out AP does not constitute independent validation.

Sources: data/processed/manifest.json, cleaning_report.json, retention_metrics.json, artifacts/retention_model_metadata.json.

## Head B

- 3,567 selected rules; support 0.005, confidence 0.1, lift 1.2.
- Evaluated later baskets: 5,884; excluded single-item baskets: 400.
- Hit-rate@5: 0.486744; popularity: 0.172332.
- Catalogue coverage: 0.076808; rule-firing coverage: 0.836676.
- CSR bytes: 2,707,668; equivalent dense boolean bytes: 119,991,887 (dense size calculated, not allocated).
- Actual timing: {'apriori_seconds': 1.2210091250017285, 'fpgrowth_seconds': 64.45212679100223}. Equal rule keys: True. FP-Growth was slower in this bounded local run.
- Mine before June; select on June–August; evaluate September–November. Deterministic half-basket masking, not an online next-purchase or sales experiment.

Source: recommendation_metrics.json. Timing is machine/workload-specific.

## Head C and integration

- Index manifest: {"embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "corpus_sha256": "b01df048a1a7d286d71f9f4bdee04d44303ae88191fca62aca72889d7c7d17c5", "chunk_count": 144, "document_count": 7, "chunk_size": 420, "chunk_overlap": 64}
- Golden questions: 40; category counts: {'policy': 15, 'tool_call': 10, 'out_of_scope': 8, 'french': 7}.
- Tool/refusal-path accuracy: 0.9750; out-of-scope refusal rate: 1.0000; in-scope refusals: 1.
- Expected policy-source accuracy: 0.8333. Exact excerpt inclusion is a grounding proxy, not correctness/completeness.
- French-to-English retrieval smoke: True.
- 16 live HTTP checks; 6 Streamlit/API interaction checks; 25 feature widgets, no UI exceptions.
- OpenAI/Ollama were not executed. French output may quote English. Golden set is a development regression set, not an unseen language-model benchmark.

Sources: assistant_regex_metrics.json, api_smoke.json, ui_smoke.json.

## Tests and operations

- Baseline: 100 passed, 1 failed (medical refusal); see TESTS_BEFORE.md.
- Final: 123 passed, 4 dependency/deprecation warnings; logs/tests_verified.log.
- Lint passed. Coverage: 64% rounded, below the unchanged 80% gate. Strict mypy: 77 errors in 13 files. CI is NOT claimed green.
- MLflow local experiment and nested tuning runs saved. Registry promotion absent.
- Monitoring demo sent 120 historical feature vectors with a deliberate +365-day recency perturbation; measured PSI and KS from durable local logs. This is synthetic demo traffic, not observed production drift or an SLO test.
- Docker Compose config validated; daemon unavailable; no image build or cloud run.
- DVC no-SCM init/commit recorded existing outputs. No remote/full repro validation. Chroma persistence mutates under serving; current index may be marked modified.
- logs/commands.jsonl and logs/*.log retain successful/failed pipeline attempts.

### Re-run on 2026-10-31

The bullets above record the 2026-09-30 run. On 2026-10-31, after typing fixes, synthetic-data tests and the container work:

- 189 tests passed; coverage 90% locally and 89% on a clean checkout (5 integration tests skip without local artifacts), above the 80% gate; ruff clean; strict mypy 0 errors in 22 files. CI still not run (no remote).
- Prediction logging moved from JSONL to a `prediction_log` database table (SQLite locally, PostgreSQL in Compose). The earlier JSONL log, which holds the 2026-09-30 monitoring traffic described above, is archived at `data/logs/archive/requests.jsonl`, excluded from git and kept until the project is graded; the code no longer reads it.
- Docker images built and the Compose stack run on linux/arm64. API image 7,376 → 1,993 MB uncompressed, 3,691 → 534 MB compressed; interface image 635 MB / 180 MB. Smoke test 10/10, including the documented `make up` with real mounts; the Ollama path worked with the configured `llama3.2:3b` on three questions (not an evaluation). Details and what was not verified: docs/CONTAINERS.md; raw sizes: container_sizes.json.

### Re-run on 2026-11-01

The retention model, rules and index are unchanged from 2026-09-30. Everything below was measured on 2026-11-01.

- **Quality gates.**
  - Locally, 273 tests pass with 90% coverage.
  - On a clean checkout (only the files git would commit, no artifacts), 266 pass and 7 integration tests skip, with 89% coverage, above the 80% gate.
  - ruff is clean, and strict mypy reports 0 errors in 31 files.
  - CI has still not run, because the repository has no remote.
- **Crash logging fixed.** A request whose endpoint crashed was missing from the prediction log and the error rate. It is now logged as the 500 the caller receives, and a test reproduces the crash.
- **Calibration and threshold charts,** drawn from the saved report (`figures/`).
  - June decision-data Brier: 0.1681 uncalibrated, 0.1578 isotonic (selected), 0.1582 sigmoid.
  - Training runs now also log both charts to MLflow, with git and data hashes as tags.
- **Fairness audit** (`fairness_audit.json`).
  - Data: test snapshot 2011-09-01, 5,233 rows, threshold 0.22. Recomputed test scores matched `test_predictions.parquet` exactly.
  - Overall: returner recall 81.4%, precision 57.3%, selection rate 60.7%.
  - No region is flagged. Five groups are strongly flagged, all customers with a thin purchase history: one-time buyers (recall 17.3%), lowest spend quartile (14.2%), second spend quartile (62.7%), tenure under 6 months (69.5%) and tenure 6–12 months (64.9%).
- **Data-minimisation audit** (`minimisation_audit.json`).
  - Refitting reproduced the saved model: decision-period PR-AUC 0.7030.
  - Twelve features are necessary. Thirteen show no reliable contribution.
  - Removing the thirteen together raised decision-period PR-AUC by 0.0052; the bootstrap interval of the loss is −0.0093 to −0.0012.
  - This is a recommendation; nothing was removed.
- **Registry.**
  - The current model is version 1 of `boreal-retention`, a candidate.
  - Its gate check on the test snapshot passes: PR-AUC 0.7430 against 0.6711 for the recency heuristic. Five fairness flags await acknowledgement.
  - It is not promoted, so the API serves `artifacts/` and reports the fallback in `/version`.
- **Assistant providers.** Measured on the 40 golden questions, a development set rather than an unseen benchmark. OpenAI was not evaluated (no key).

  | | Rules (regex) | Ollama `llama3.2:3b` (native app, Apple GPU) |
  |---|---|---|
  | Tool-choice accuracy | 97.5% | 80.0% |
  | Out-of-scope refusal | 100% | 100% |
  | Expected source cited | 83.3% | 72.2% |
  | In-scope refusals | 1 | 4 |
  | Latency p50 / p95 | 29 / 65 ms | 4,163 / 4,931 ms |
  | Cost per question | $0 | $0 (local) |

  - **Ollama latency.** The table holds the final run (22:34 UTC), the one in `assistant_ollama_metrics.json`. An earlier run that afternoon, on the same questions, measured 2,386 / 2,711 ms; its file is in the git history. The cause of the difference was not investigated, so local LLM latency here is approximate.
  - **Rules, re-run on today's code.** Its totals match 2026-09-30. Four product questions now answer "no matching product" from the real catalogue export (the invented catalogue was removed on 2026-10-21), while still choosing the right tool.
  - **Streamed answers written by Ollama.**
    - The model wrote 15 of the 18 policy answers, all in the requested language (3 in French), all cited, with no fallbacks.
    - The figure check flags 7 answers that leave out a figure stated in their main excerpt; some are false alarms when that excerpt was off-topic. It flags none that add a figure.
    - Answer correctness is not scored.
  - **Prompt tuning.** An added instruction to "state deadlines exactly" made the model claim that frozen products can be returned "within 30 days"; they cannot be returned at all. The figure check flagged it. On the same 40 questions, 1 answer added a figure with the instruction and 0 without, so the instruction was removed.
- **Monitoring demonstration** (`monitoring_demo.json`): a fresh local API, with alerts evaluated every 5 s, and two runs of `simulate_traffic`. The file records the second run; its `before` block is the state after the first.
  - **Real traffic** (first run, without `--shifted`: the `before` block). 120 historical rows (20 single requests, the rest in batches) and 40 real September–November 2011 baskets.
    - `/predict` p95 was 49.6 ms (green).
    - The rule fire rate was 85%, against 84% offline coverage.
    - Input and prediction drift were red, with 61% of the tenure-type values outside the 2010 training range: genuine temporal drift.
  - **Deliberate shifts** (second run, `--shifted`: the `after` block). 120 rows with recency moved by 365 days, 20 real and 20 unknown-item baskets, and 10 invalid requests. They raised the error rate (7.1%) and the invalid-request rate (17.2%) to yellow.
  - **Alerts.** The background monitor alerted on all five breached checks without anyone opening the dashboard.
- **Live checks** (API and interface started for the test).
  - API smoke test: 16 of 16.
  - Interface smoke test: 6 of 6, including the live what-if (changing recency changed the probability with no button) and a streamed answer.
  - A decision was recorded and listed back.
  - A French answer streamed from Ollama.
- **Containers** (linux/arm64).
  - API 2,064 MB uncompressed / 553 MB compressed: +71 MB, about 46 MB of it the registry client. Interface 630 / 179 MB. SageMaker 1,044 / 290 MB.
  - With the new images, the Compose stack was healthy. `/version` reported the local fallback while the optional MLflow server was off.
- **SageMaker image, run locally the way SageMaker would** (`sagemaker_local.json`: re-run and recorded on 2026-11-02 with the same image). Nothing ran on AWS.
  - A full training job reproduced the reported test metrics exactly (PR-AUC 0.7430379) in 18.2 s, including container start. The model file is not byte-identical to `artifacts/`, so the match rests on the metrics; the image has numpy 2.5.4, where `artifacts/` was built with 2.5.3.
  - A tuning trial printed `validation_pr_auc=0.673018`.
  - The endpoint answered `/ping`, and `/invocations` scored customer 12347 at 0.711864, as the API does.

See REMAINING_ISSUES.md for the prioritized gaps and RUNBOOK.md for reproduction.
