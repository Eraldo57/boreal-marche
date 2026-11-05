> Latest completion verification: 428 tests pass (421 unit, 7 artifact integration), 90% source/API coverage; 16 HTTP and 7 Streamlit checks against rebuilt local stack.
> See completion_coverage_summary.txt and logs/completion_*. Historical entries below retain their original model/version dates.

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
- **Assistant providers.** Measured on the 40 golden questions, a development set rather than an unseen benchmark. There are two rounds. The first used the original index, and the router also handed the retrieved text to the language models. The second (2026-11-02, after the retrieval and routing changes described below) is the one the reports now describe. OpenAI is `gpt-4o-mini`; its files are stamped in UTC, so its 2026-11-02 evening runs read 2026-11-03.

  | | Rules (regex) | Ollama `llama3.2:3b` (native app, Apple GPU) | OpenAI `gpt-4o-mini` |
  |---|---|---|---|
  | Tool-choice accuracy | 97.5% (same) | 80.0% → **92.5%** | 90.0% → **95.0%** |
  | Out-of-scope refusal | 100% | 100% | 100% |
  | Expected source cited | 83.3% → 77.8% | 72.2% → 77.8% | 77.8% (same) |
  | In-scope refusals | 1 (same) | 4 → 2 | 3 → 1 |
  | Latency p50 / p95 | 29 / 65 → 27 / 59 ms | 4,163 / 4,931 → 1,370 / 1,793 ms | 1,004 / 1,572 → 1,015 / 1,336 ms |
  | Cost per question (routing only) | $0 | $0 (local) | $0.000107 → $0.000062 |
  | Cost per streamed policy answer | n/a | $0 (local) | $0.000232 → $0.000220 |
  | Policy answers written by the model (of 18) | n/a | 15 → 18 | 17 → 18 |

  - **What changed between the rounds** (details below). The index now matches small titled chunks but returns the whole policy section. The routing prompt no longer includes retrieved text and no longer offers a "direct answer" option. **Both changes were made after looking at this development set**, so its tool-choice scores are partly tuned to it. The first round is kept in `assistant_*_metrics_before_sections.json`.
  - **Rules provider.** Its source score fell because one English question (Q7) now returns the French twin of the same privacy notice first; the content is the same.
  - **OpenAI cost.** It is measured tokens at the list price, $0.15 per million input and $0.60 per million output tokens (OpenAI's pricing page, read 2026-11-02; prices change). Second round: 31 routing calls used 11,935 input and 1,171 output tokens, and the 18 answers used 10,805 input and 1,428 output tokens. A full evaluation costs under a cent. Routing got cheaper because its prompt shrank.
  - **OpenAI run-to-run variation.** Two first-round runs with identical settings (temperature 0) gave the same tool choices, refusals and sources, but changed the text of 8 of the 18 written answers (`assistant_openai_metrics_first_run.json`). Treat small differences between runs as noise.
  - **Ollama latency varies between runs.** The first round was measured twice on the old prompt, at 2,386 and 4,163 ms; the cause was not investigated. The second round's 1,370 ms is one run on a shorter prompt.
  - **Rules, re-run on today's code.** Four product questions answer "no matching product" from the real catalogue export (the invented catalogue was removed on 2026-10-21), while still choosing the right tool.
  - **Streamed answers.** In the second round both models wrote all 18 policy answers in the requested language (4 in French), all cited, with no fallbacks. The figure check flags 11 (Ollama) and 9 (OpenAI) answers as leaving out a figure from their main excerpt, and none as adding one. Those flags rose because the main excerpt is now a whole section with many figures, so the check is a weaker signal than before.
  - **Answer correctness** (`answer_correctness.json`; facts and verdicts in `tests/golden/policy_answer_rubric.json`, produced by `scripts/score_answers.py`). Each of the 18 written policy answers was scored against the policy text. **A single AI evaluator supplied the ratings, so the verdicts are not independent; please spot-check them.** Each verdict is tied to a hash of the exact answer it judged, and each fact to a phrase quoted from `docs/corpus`. The first round's verdicts are kept in `policy_answer_rubric_before_sections.json`.

    | Verdict | OpenAI | Ollama |
    |---|---|---|
    | Correct | 2 → **11** | 1 → **3** |
    | Partly correct (true, but leaves out a fact the question needs) | 5 → 2 | 1 → 4 |
    | Declined, and the corpus really lacks the answer | 3 → 3 | 1 → 3 |
    | Declined, although the corpus answers | 5 → 1 | 7 → 4 |
    | Wrong (contradicts the corpus or states what it lacks) | 2 → 1 | 5 → 4 |
    | No answer (the routing step failed or refused) | 1 → 0 | 3 → 0 |

    - **With the passage in front of it** (12 of the 14 answerable questions), OpenAI was correct on 11 and partly correct on 1. Ollama was correct on 3, partly correct on 4, declined 3 and was wrong on 2.
    - **Without it** (2 questions: Q12 and Q38, deleting an account and its data), OpenAI declined once and answered partly once; Ollama declined once and was wrong once.
    - **Questions the corpus cannot answer (4).** Both declined 3 rightly. Both were wrong on the fourth, Q40, which asks about bakery allergens in French: OpenAI said bakery products "may therefore contain" the priority allergens, and Ollama listed allergens including ones the policy never names. That is a safety-relevant error in a French answer. OpenAI's English version, Q4, declined correctly.
    - **Limits.** Eighteen questions are too few for percentages to mean much. The 3B model's errors are of two kinds: refusing when it has the answer, and garbling or inventing details.
  - **Retrieval** (`retrieval_metrics.json`, from `scripts/evaluate_retrieval.py`; the unchanged index is in `retrieval_metrics_baseline.json`). The model is handed the first 3 retrieved passages. The table counts the questions whose answering passage was among them:

    | Question set | Before | After |
    |---|---|---|
    | 14 answerable golden questions | 4 | **12** |
    | 20 indirect (paraphrased) hold-out questions | 7 | **18** |
    | 26 direct hold-out questions | 22 | 23 |

    - **The hold-out set** (`tests/golden/retrieval_holdout.json`, 46 questions) was written by me, before any retrieval change, from the policy files, so changes were not tuned on the questions they are scored on. My first 26 questions turned out too easy to show any difference, so I added the 20 paraphrased ones before changing anything. One author and 46 questions make it a sanity check, not a benchmark.
    - **The change.** The index splits each policy into sections, cuts them into small chunks that each begin with the policy and section title, and matches those. A match returns the whole section. The encoder reads only 128 tokens, so matching has to use small pieces, but a fact often sits in a table or list that a small piece would cut. The old index had 120 chunks from 6 documents; the new one has 130 chunks from 59 sections of the same 6.
    - **What was tried and left out** (`retrieval_experiments/results.txt`, top-3 hits out of 60 questions): plain chunks 37; plus titles 42; plus whole sections (adopted) **54**; plus keyword search 48; one chunk per list item or table row 55; giving the model 5 passages instead of 3 54 for the adopted design, with keyword search 57 at 68% more text per question. The gains over the adopted design were within noise on a set this size, so they were not worth the extra complexity.
    - **Still failing.** Questions phrased as "delete or erase my data" (Q12, Q38, I09), a pickup-location question (H09), a failed-delivery question (H10), the privacy-officer question (H23) and the Sapin-versus-Pousse comparison (I03) do not bring the answering passage into the first 3.
    - **The routing side effect.** With better text in front of it, the routing model started answering questions itself, and the router discarded those answers as refusals (Q1, then Q9 and Q13 once the prompt was partly fixed). The router never needed the policy text, because `policy_search` retrieves its own, so the routing prompt now holds the question only.
    - **A correction.** The index manifest quoted in the 2026-09-30 section above (144 chunks, 7 documents) describes an earlier corpus. The index on 2026-11-02, before this change, had 120 chunks from the current 6 documents.
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

### Re-run on 2026-11-02

The same code, plus the CI changes below. Logs are in `logs/` under the names given.

- **Quality gates** (final run of the day, after the assistant and retrieval changes).
  - 369 tests pass locally with 90% coverage (`tests_20261102d.log`, `coverage_20261102d.log`). Earlier that day, before those changes, the counts were 273 and 266.
  - On a clean checkout, 362 pass and 7 integration tests skip, with 89% coverage (`tests_clean_checkout_20261102b.log`).
  - ruff, formatting and mypy (0 errors in 32 files) are clean, and actionlint finds no problems in the three workflows (`actionlint_20261102.log`).
  - The live checks were repeated on the rebuilt index: API smoke test 16 of 16 and interface smoke test 6 of 6 (`api_smoke_sections_20261102.log`, `ui_smoke_sections_20261102.log`).
- **Data-quality suite.**
  - The assertions moved to `tests/test_data_quality.py`, which needs only the processed data. They now cover every cleaning rule and the bounds that the feature definitions imply. They pass on the real data.
  - CI's data-quality job used to skip every time, because CI has no data. It now rebuilds the processed data from the public workbook, checked against its pinned SHA-256, and runs the suite.
- **Assistant, updated on 2026-11-02.** The assistant results in the 2026-11-01 section above were re-measured after two changes (a section-based index and a question-only routing prompt). OpenAI was evaluated for the first time, answer correctness was scored, and a hold-out set measured retrieval. See that section for the numbers.
- **CI reproducibility.**
  - ruff, mypy and the secret scanner are pinned.
  - PyTorch is installed CPU-only, as in the Docker image.
  - pip downloads are cached.
- **Not yet verified:** CI itself, because the repository still has no remote.

See REMAINING_ISSUES.md for the prioritized gaps and RUNBOOK.md for reproduction.

### Re-run on 2026-11-04

Defence-question review: each answer in `docs/DEFENCE_PREPARATION.md` was checked against the code and the saved results, and three gaps it exposed were fixed.

- **Interface wording** (question 15). A recommendation list now says which strategy produced it, with a plain-language notice for popularity fallbacks and a "because you added X" line per rule-based item (`frontend/notes.py`). The interface smoke test now feeds an unknown item and asserts the notice.
- **Relevance check** (question 19, `retrieval_floor.json`). No single similarity cutoff separates answerable from undocumented questions: on 60 answerable and 24 undocumented questions, a floor of 0.40 would decline 18 of 24 undocumented ones but wrongly decline 9 of 60 answerable ones. The assistant now has two thresholds, a floor of 0.25 (decline) and a confident score of 0.45 (below it, a passage is shown with a caution and the support contact). Of the 24 undocumented questions, 18 get the caution, 2 are declined and 4 pass as confident (before: 22 of 24 looked like plain answers). Of the 60 answerable ones, 46 are confident, 12 get a caution and 2 are declined (the same 2 as before). Both values are judgement values chosen from these sets.
- **Per-customer explanation** (question 32). `/predict` returns `explanation`: the five largest effects of this customer's own feature values on their probability, against the training median, from the same single prediction call (the probability is unchanged, 0.711864 for customer 12347). With recency raised by 365 days, recency becomes the largest effect (−0.185), although it is 6th in the global importances. It is a one-at-a-time ablation, not SHAP, it is not stored with the decision, and its wording is unreviewed.
- **Numbers behind the answers** (`defence_facts.json`, `rules_scaling.json`). Median gap between a customer's purchases: 32 days; 80.6% of repeat gaps are 90 days or less. At the model's operating point the model and a "most recent customers first" rule nearly tie (recall 0.814 and 0.810; precision 0.573 and 0.570). The lift filter removes 12 of 3,579 rules, none above confidence 0.17, so the brief's "high-confidence, low-lift" example does not exist in this data. Lookup time with 500,000 synthetic rules: median 3.1 ms, 95th percentile 14.8 ms.
- **Live checks** repeated on the final code: API smoke test 16 of 16 and interface smoke test 7 of 7 (`api_smoke_fixes_20261104.log`, `ui_smoke_fixes_20261104.log`).
- **Quality gates** on the final code: 410 tests pass with 91% coverage (`tests_20261104.log`, `coverage_20261104.log`); on a clean checkout 402 pass, 8 skip, 89% (`tests_clean_checkout_20261104.log`); ruff, formatting and mypy (0 errors in 32 files) are clean; actionlint finds no problems; gitleaks 8.30.1 finds no secrets in the files git would commit.
