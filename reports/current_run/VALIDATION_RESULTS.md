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

See REMAINING_ISSUES.md for the prioritized gaps and RUNBOOK.md for reproduction.
