# Student technical guide — Boréal Marché

## The explanation to start with

“I implemented most of the architecture. I have now run the real public-data pipeline and validated the main local API/UI path. The model results are historical offline results; the assistant and operational controls still have explicit gaps.”

This is one platform with three heads, not three unrelated notebooks. UCI transactions feed Heads A/B; authored policy/product documents feed Head C. Assistant tools connect those heads through the same API contracts. Boréal Marché is fictional, the source retailer sold UK giftware, and source prices remain GBP.

Use reports/current_run/VALIDATION_RESULTS.md for current measured results. Earlier audit/defence documents and notebooks are historical development materials, not the authority for this revision.

## 1. configs/pipeline.yaml

**Purpose:** make data, labels, models, rules and assistant settings visible.  
**Input:** human decisions and verified source URL/checksum.  
**Process:** modules read their relevant sections; features check the ordered list against src/contracts.py.  
**Output:** reproducible settings used at each stage; hashes in manifests/metadata.  
**Important entries:** source_sha256, cleaning.rules, labelling.snapshots, modelling.tuning, threshold.cost_assumptions, association_rules sweeps, assistant embedding model.  
**Next stage:** ingestion reads source/cleaning; later modules read their own settings.  
**Limitations:** not every engineering default is configurable; dependency/model revision locking is incomplete. Coupon cost/margin are assumptions.

Oral point: “Changing a cutoff changes what the model is allowed to know; it is not just changing a file name.”

## 2. src/ingest.py

**Purpose:** produce an auditable event ledger.  
**Input:** both sheets of the official online_retail_II.xlsx.  
**Process:** download if missing; enforce pinned SHA-256; normalize schema/types; apply ordered rules; flag events; cap extremes with fixed limits; write Parquet and reports.  
**Output:** transactions.parquet, manifest.json and cleaning_report.json under data/processed.  
**Important functions:** _sha256, download_dataset, verify_checksum, clean, to_parquet, run.  
**Next stage:** features and rules both consume the ledger but take positive-purchase views for purchases.  
**Limitations:** dropping missing IDs creates selection bias; capping wholesale quantities needs sensitivity analysis; no stable line IDs justify automatic duplicate deletion.

Oral point: “I keep cancellation events so the model can measure cancellation history, but they never become positive purchases.”

## 3. src/features.py and src/contracts.py

**Purpose:** convert transaction lines into point-in-time customer rows and future targets.  
**Input:** cleaned ledger and six configured cutoff snapshots.  
**Process:** purchases before cutoff establish eligible customers; purchase labels are in [cutoff, cutoff + 90 days). RFM, cadence, basket, category, behavioral and geography builders see only pre-cutoff data. Country mapping is fixed from earliest training history.  
**Output:** X.parquet, y.parquet, snapshots.parquet, feature_metadata.json and latest historical customer_features.parquet.  
**Important functions:** purchases, create_labels, build_features, _rfm_features, _behavioural_features, save_features, run; FEATURE_NAMES is the ordered contract.  
**Next stage:** training reads X/y/snapshot alignment; API exposes historical rows to make valid demo requests.  
**Limitations:** description-first-token categories, redundant average-value features and ordinal countries are simple approximations. One-order gap uses tenure; remaining NaNs belong to the train-only imputer.

Exactly 25 features are explained in governance/FEATURE_INVENTORY.md. No customer ID is a model feature. Snapshots repeat customers; row counts are not counts of distinct people.

## 4. src/train.py

**Purpose:** select, calibrate and evaluate a return-prediction model without choosing on the test set.  
**Input:** ordered feature matrix, labels, snapshot roles and cost assumptions.  
**Process:** check label maturity → compare eight classifiers/two baselines → tune top two tunable families → refit chosen estimator on train+selection validation → fit calibrators on March → choose calibration and threshold on June → score September once after decisions lock.  
**Output:** model, metadata, drift reference, test predictions; separate retention_metrics.json; real MLflow runs.  
**Important functions:** temporal_partitions, pipeline, _get_models, trial_model, select_threshold, utility, error_analysis, run.  
**Next stage:** API lifespan verifies and loads model/metadata; monitoring uses the training reference.  
**Limitations:** a small temporal development set; no independent new holdout after future revisions; no causal coupon uplift, full fairness study or cloud HPO. Re-running the same protocol is reproducibility, not new evidence.

| Role | Cutoffs | What it may decide |
|---|---|---|
| Training | June + September 2010 | Estimator/preprocessing fit |
| Selection validation | December 2010 | Family and hyperparameters; later included in estimator refit |
| Calibration | March 2011 | Fit sigmoid/isotonic maps only |
| Decision | June 2011 | Calibration choice, utility threshold |
| Final test | September 2011 | Report locked model and baseline performance |

Positive means **will return**. AP ranks returners. High return probability means low non-return risk. Threshold 0.22 reflects assumed targeting economics; it is not an optimized churn intervention policy. The observed one-time-buyer recall gap is important, not a cosmetic issue.

## 5. src/rules.py and src/recommend.py

**Purpose:** offline basket-pattern mining and fast online lookup.  
**Input:** positive invoices and canonical product descriptions.  
**Process:** unique descriptions per invoice → CSR/Pandas sparse matrix → support AND confidence exploration with lift ≥1.2 → validation selection → later-basket evaluation → export. Apriori/FP-Growth timing uses the same selected settings and maximum itemset length.  
**Output:** rules.json, popularity.json, recommendation_metrics.json.  
**Important functions:** baskets, build_transaction_matrix, temporal_baskets, evaluate_baskets, run; RecommendationEngine.load/from_records/recommend.  
**Next stage:** API loads an item-to-rule index once; matching rules rank consequents by lift, deduplicate, exclude basket items and include firing-rule evidence.  
**Limitations:** description identity is not a curated SKU master; maximum itemset length is three. Popularity fills gaps. Offline Hit-rate@5 masks half a later basket and asks whether any hidden item appears in five suggestions; it does not measure incremental revenue.

Explain support as basket frequency, confidence as conditional co-occurrence, lift as confidence divided by consequent popularity. CSR memory is measured; dense memory is a computed equivalent, not a dense allocation. In this workload Apriori was faster—do not claim FP-Growth always wins.

## 6. src/rag components

**Purpose:** answer from authored documents and real tools rather than inventing account results.  
**Input:** docs/corpus Markdown plus a question, optional demo ID and selected provider.  
**Process:** index.py chunks/embeds documents offline; RagIndex embeds questions online and retrieves cosine neighbors. providers.py routes to one of four tools. tools.py validates arguments and executes; guardrails.py checks scope/privacy; API chat orchestrates and formats grounded results.  
**Output:** persistent Chroma index/manifest, cited policy excerpts, typed tool results or refusals; assistant evaluation JSON.  
**Important functions/classes:** build_index, RagIndex.query, embedding_model, get_provider, RegexProvider, TOOL_REGISTRY; each tool's input_model and execute.  
**Next stage:** /chat uses this same abstraction; Streamlit displays reply, citations and provider.  
**Limitations:** multilingual retrieval works, but regex may quote English for French questions. Paragraph boundaries/relevance need improvement. OpenAI/Ollama are unvalidated optional routers, not a completed free-form bilingual generator. Guardrails are heuristic, not an authorization system.

Four tools: policy_search (index), retention_score (look up all features, call /predict), recommendations (/recommend), product_lookup (fictional catalogue). Policy citations prove where an excerpt came from, not that it fully answers the question. The 40-question set is preserved: 15 policy, 10 tools, 8 out-of-scope, 7 French.

## 7. FastAPI

**Purpose:** one serving interface and enforceable data contracts.  
**Input:** trusted artifacts plus strict Pydantic v2 requests.  
**Process:** lifespan loads/validates metadata, hashes, feature order, rules/popularity, historical feature store and RagIndex. Middleware records actual monitored traffic; routes run the corresponding head.  
**Output:** JSON predictions, recommendations, chat, readiness/version/monitoring.  
**Important modules/functions:** api/main.py app and middleware; api/lifespan.py lifespan and PredictionLog; api/schemas.py; routers/predict.py _predict_one.  
**Next stage:** Streamlit and typed tools use these routes rather than importing/training ML in the UI.  
**Limitations:** localhost demo only; no auth, production quotas, worker-safe centralized telemetry or sophisticated per-person explanations.

Health is process liveness; readiness is artifact availability. A running process can be not ready. Exact features/order come from the canonical contract; /predict/batch validates every payload and supports up to 500 customers.

## 8. frontend/streamlit_app.py

**Purpose:** make the system demonstrable through one interface.  
**Input:** API_BASE_URL, historical customer ID, basket, question and provider selection.  
**Process:** HTTP calls only; load canonical 25-value record, show editable what-if controls, call API and display errors instead of fake results.  
**Output:** predictions, rule explanations, policy citations, monitoring and real metadata.  
**Important function:** api_call; five tab blocks.  
**Next stage:** user inspection and presentation, not automated business actions.  
**Limitations:** interface is not fully localized or production-polished. AppTest validates widget/API interactions, not browser layout or multi-user behavior.

## 9. Artifacts and provenance

**Purpose:** separate generated evidence from code/configuration.  
**Input:** executed offline stages.  
**Process:** save artifacts and generation-time hashes, model/library versions, git commit plus dirty flag.  
**Output:** see reports/current_run/artifact_inventory.json; canonical metadata sidecar is retention_model_metadata.json.  
**Important interfaces:** contracts.validate_metadata and API lifespan; DVC lock and MLflow run ID.  
**Next stage:** startup consumes artifacts; slides summarize JSON evidence.  
**Limitations:** timestamps/binary serialization can change between runs; hash evidence is not authentication. Chroma is mutable during queries. No DVC remote or registry promotion exists. Formatting/config cleanup after generation can change current hashes without retroactively changing generation-time provenance.

## 10. Tests and evaluation

**Purpose:** catch regressions and distinguish code tests from real-world validation.  
**Input:** small deterministic fixtures, preserved golden questions, optionally real generated artifacts.  
**Process:** pytest tests contracts, checksum rejection, cleaning, cutoff boundaries, train-only medians, temporal maturity, readiness, tools, guards, recommendation behavior and drift. Separate scripts execute real API, UI and golden-set evaluation.  
**Output:** logs, JSON reports and coverage.  
**Important files:** tests/test_pipeline.py, test_real_artifacts.py, test_data_quality.py, existing route/provider tests; scripts/smoke_api.py, smoke_ui.py, evaluate_assistant.py.  
**Next stage:** CI gates and regression protection. CI's data-quality job rebuilds data/processed from the public workbook, checked against its pinned SHA-256, and runs test_data_quality.py: each assertion follows a cleaning rule or a feature definition.  
**Limitations:** On 2026-11-01, 273 tests pass locally with 90% coverage. On a clean checkout, 266 pass and 7 integration tests skip, with 89% coverage; the gate is 80%. The 2026-11-02 re-run gave the same counts (reports/current_run/logs/tests_20261102.log and tests_clean_checkout_20261102.log). Strict mypy reports 0 errors. Mocked golden API probabilities are fixtures, not model results. The golden set's tool-choice test (tests/test_assistant_eval.py) uses pass marks of 40–60%, well below the measured 97.5%, so it would miss small regressions.

## 11. Docker and workflows

**Purpose:** reproducible service packaging and future delivery gates.  
**Input:** project dependencies/source plus host-generated data/model/index artifacts.  
**Process:** multi-stage, digest-pinned Python 3.12 build with CPU-only PyTorch; separate API and Streamlit images; non-root runtime; /ready health check; Compose runs API, interface, PostgreSQL (prediction_log) and Ollama, with artifacts and data mounted from the host.  
**Output:** the API, interface and SageMaker images built and the stack run locally on linux/arm64 (sizes and checks in docs/CONTAINERS.md); no cloud deployment.  
**Important files:** docker/Dockerfile, docker/compose.yml, .dockerignore, and the .github/workflows/ files ci.yml, cd.yml and retrain.yml.  
**Next stage:** a GitHub remote so CI runs (it builds linux/amd64), a fresh-clone test on another machine, then AWS for CD.  
**Limitations:** No GitHub run or AWS deployment has happened. CD triggers on merge to main but stays off until `ENABLE_CLOUD_DEPLOYMENT` is set. The weekly retraining workflow stays off until `ENABLE_RETRAINING` is set.

## 12. MLflow, retraining and monitoring

**Purpose:** keep experiments and make operations observable.  
**Input:** training metrics/artifacts; actual route requests and prediction feature vectors.  
**Process:**
- **MLflow** logs the comparison, the tuning trials, the final model, provenance tags (git commit, data hashes) and two plots. It then registers the model as a candidate version.
- **PredictionLog** writes one row per request to the prediction_log table: SQLite locally, PostgreSQL in Compose, with the URL from DATABASE_URL. It replays the table on restart and keeps a bounded recent window.
- **api/monitoring.py** turns that window into checks, each with its rule from configs/pipeline.yaml: error rate, `/predict` p95 latency, input drift and prediction drift (quantile-bin PSI and two-sided KS against drift_reference), data quality and the rule fire rate.
- **A background task** re-evaluates the checks every ALERT_INTERVAL_SECONDS. It raises an alert when a check turns yellow or red, and resolves it when the check recovers.

**Output:** MLflow runs and registry versions, prediction_log and alert_log rows, /metrics and /alerts.  
**Important functions:** src.train.run; src.registry; api.monitoring.snapshot and AlertManager; scripts/simulate_traffic.main.  
**Next stage:** the project owner promotes a version; a webhook makes alerts reach a person.  
**Limitations:** no purge, no multi-worker aggregation, and delayed-label backfill is only designed (docs/MONITORING.md). KS p-value <0.05 is the alert condition, not KS statistic >0.05. Insufficient traffic is reported as `not_evaluated`, not as green or zero. Real 2011 demo traffic already drifts from the 2010 training reference, which is genuine temporal drift; the `--shifted` perturbations are deliberate and synthetic.

## 13. Fairness and data minimisation (src/fairness.py, src/minimisation.py)

**Purpose:** answer the brief's two governance audits (section 3.7) with code, so every number in the documents is regenerated, not typed.  
**Input:** the saved model (checksum verified), the processed features, and the June 2011 decision period and September 2011 test period.  
**Process:**
- **Fairness.** Recall, precision and selection rate by region, tenure band, spend quartile and purchase history. Wilson 95% intervals. Flags groups more than 5 points below overall recall. Estimates on June data the group threshold that would close each gap.
- **Minimisation.** Refits the selected pipeline without each feature, then without all the doubtful ones together, and measures the PR-AUC loss with paired-bootstrap intervals on June data.

**Output:** fairness_audit.json, minimisation_audit.json, and the generated tables inside governance/fairness_audit.md and governance/FEATURE_INVENTORY.md (src/generated_docs.py).  
**Next stage:** the owner chooses a fairness mitigation and decides on removing the 13 features.  
**Limitations:** the groups are behavioural proxies, not protected characteristics. The minimisation audit recommends; it changes nothing.

## 14. Registry and promotion (src/registry.py)

**Purpose:** serve a model chosen by a person, with rollback that needs no new image (section 3.5).  
**Input:** registered versions; the holdout from data/processed.  
**Process:** gates run on one shared holdout:
- beats the baseline (first champion only);
- same feature encoding;
- PR-AUC and Brier not worse than the champion, within a tolerance;
- no new or widened fairness flags.

`promote` needs a named approver, a reason, and an acknowledgement of known flags, then moves the `champion` alias. `rollback` moves it back. The API resolves the alias at startup and otherwise falls back to artifacts/.  
**Output:** alias, approval tags, and evidence in reports/current_run/registry/.  
**Limitations:** nothing promoted yet; the local registry is a SQLite file. MLflow cannot stop someone with write access from moving the alias by hand.

## 15. Human approval and the customer notice

**Purpose:** the design consequence of Law 25, section 12.1: the model proposes and a person decides.  
**Process:**
- The Prediction tab shows the model's proposed action; a reviewer approves or rejects it with a reason (`POST /decisions`, stored in `decision_log`).
- The About tab renders the French or English customer notice, the model card, the privacy notice, the data card and the architecture diagram.

**Limitations:** no authentication, so the reviewer's name is declared, not verified. Nothing is ever sent to a customer.

## 16. Streaming assistant (api/routers/chat.py, src/rag/providers.py)

**Purpose:** answers that appear word by word, and fluent answers in the requested language (section 3.6).  
**Process:** `/chat/stream` uses the same guardrails and tools as `/chat`.
- **OpenAI or Ollama, policy question:** the model writes the answer from the cited excerpts.
- **Figure check:** flags figures the answer adds, or figures the main excerpt states that the answer leaves out.
- **Fallback:** if generation fails, the stream shows the source text instead.
- **Regex:** streams its quote.

**Limitations:** on 2026-11-01, `llama3.2:3b` wrote in the right language 15 times out of 15. During prompt tuning it also stated a false policy (frozen returns "within 30 days"), which the figure check caught. Model-written policy answers are a demonstration, not a reliable channel.

## 17. SageMaker (scripts/sagemaker_entry.py, api/sagemaker_serve.py, scripts/sagemaker_jobs.py)

**Purpose:** the same training file locally and on SageMaker, a managed Bayesian tuning job, and an endpoint deployed once and then deleted (section 3.5).  
**Output:** the image (target `sagemaker`) and launcher commands that record evidence.  
**Limitations:** verified only locally in the container (`reports/current_run/sagemaker_local.json`). The full training job reproduced PR-AUC 0.7430379, and the endpoint scored customer 12347. Nothing has run on AWS.

## Questions to be ready for

- **Is it finished?** No. The local artifact-backed path is validated; the remaining issues report is part of the deliverable.
- **Why is the model not HistGradientBoosting?** The current temporal comparison selected RandomForest; architecture does not predetermine the winner.
- **Does a citation guarantee truth?** It identifies retrieved evidence. Wrong-source retrieval and incomplete excerpts remain measured errors.
- **Does the utility prove profit?** No, it uses assumed costs/margins and does not estimate intervention effects.
- **Why not deploy now?** The cloud side needs a GitHub remote and an AWS account. Authentication, deletion enforcement and a decision on the fairness gaps are also unfinished.
- **Does the language model help?** Not for routing: on the golden set, rules picked the right tool 97.5% of the time and `llama3.2:3b` 80%. It does help with answers in the user's language, but it can contradict the source, so its answers come with the excerpt and a figure check.
- **Does drift mean the model is broken?** No. Real 2011 traffic drifts from the 2010 training reference, yet the model scored PR-AUC 0.743 on the 2011 test period. Drift is a reason to look, and only labels (delayed by 90 days) can show accuracy.
- **What next?** Push to GitHub so CI runs; promote a champion; decide on minimisation and the fairness mitigations; then AWS (CD with OIDC, SageMaker, App Runner or Render).
