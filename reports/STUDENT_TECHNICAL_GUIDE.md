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

## 8. app/streamlit_app.py

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
**Important files:** tests/test_pipeline.py, test_real_artifacts.py, existing route/provider tests; scripts/smoke_api.py, smoke_ui.py, evaluate_assistant.py.  
**Next stage:** CI gates and regression protection.  
**Limitations:** 123 local tests pass, but roughly 64% src/api coverage misses the unchanged 80% gate; strict mypy also fails. Mocked golden API probabilities are fixtures, not model results. Real integration tests skip when artifacts are absent.

## 11. Docker and workflows

**Purpose:** reproducible service packaging and future delivery gates.  
**Input:** project dependencies/source plus host-generated data/model/index artifacts.  
**Process:** multi-stage Python 3.12 build; non-root runtime; Compose networking uses api:8000; /ready health check; API/cache/log volumes.  
**Output:** configuration validated, not an executed image/deployment.  
**Important files:** Dockerfile, docker-compose.yml, .github/workflows/ci.yml and cd.yml.  
**Next stage:** validate containers when Docker is running, then review opt-in cloud configuration.  
**Limitations:** Docker daemon was unavailable. No GitHub run or AWS deployment was executed; CI has known failing gates. CD is explicitly manual/opt-in; cloud credentials and real deployment review remain future work.

## 12. MLflow, retraining and monitoring

**Purpose:** keep experiments and make operations observable.  
**Input:** training metrics/artifacts; actual route requests and prediction feature vectors.  
**Process:** MLflow logs comparison/tuning and final model to local SQLite/files. PredictionLog persists JSONL and maintains a bounded recent buffer. monitor.py compares recent predictions against drift_reference with quantile-bin PSI and two-sided KS.  
**Output:** MLflow run records, data/logs telemetry, /metrics; timestamped retraining candidates.  
**Important functions:** src.train.run; scripts/retrain.main; PredictionLog; _compute_drift; scripts/simulate_traffic.main.  
**Next stage:** human investigation and future approved promotion, not automatic retraining/deployment.  
**Limitations:** no purge, distributed aggregation, delayed labels, automatic alerts or registry aliases. KS p-value <0.05 is the alert condition, not KS statistic >0.05. Insufficient traffic returns unavailable drift, not fake zero. Deliberately shifted demo traffic cannot establish production drift.

## Questions to be ready for

- **Is it finished?** No. The local artifact-backed path is validated; the remaining issues report is part of the deliverable.
- **Why is the model not HistGradientBoosting?** The current temporal comparison selected RandomForest; architecture does not predetermine the winner.
- **Does a citation guarantee truth?** It identifies retrieved evidence. Wrong-source retrieval and incomplete excerpts remain measured errors.
- **Does the utility prove profit?** No, it uses assumed costs/margins and does not estimate intervention effects.
- **Why not deploy now?** Auth/privacy controls, fairer segment behavior, CI gates and container/cloud validation are unfinished.
- **What next?** Improve retrieval and cold-start validation, close typing/coverage gaps, make immutable data/model promotion, then evaluate deployment and governance enforcement.
