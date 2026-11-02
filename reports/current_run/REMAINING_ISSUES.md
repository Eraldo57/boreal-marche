# Remaining work — prioritized, not hidden

Updated 2026-11-01.

## P0 — before any real/customer-facing exposure

- **No authentication or authorization.** There is no authenticated identity, authorization, rate limiting or secure customer feature lookup, so bind local servers to localhost only. Matching a caller-supplied ID is not authentication, and the approval step's reviewer name is declared, not verified.
- **No downstream processes.** There is no appeal or correction workflow, coupon dispatch or support handoff. The human approval step records decisions; it does not act on them.
- **No enforced retention or purge.** This applies across:
  - the prediction, alert and decision logs (`decision_log` holds customer IDs);
  - features, raw data and the DVC cache;
  - backups, the MLflow registry and model lineage.

  Prediction vectors can be identifying.
- **Joblib deserialization requires trusted artifacts,** whether local or from the registry. A hash match does not authenticate a publisher, and anyone with registry write access can move the champion alias.

## P1 — next integration/validation milestone

- **CI has never run.** Locally on 2026-11-01:
  - 273 tests pass with 90% coverage (89% on a clean checkout);
  - strict mypy reports 0 errors in 31 files, and ruff is clean.

  Without a GitHub remote there is no Actions run, branch protection, green pipeline or linux/amd64 image build. The golden-set tool-choice test's pass marks (40–60%) sit far below the measured 97.5%, so it would miss small regressions.
- **Registry: nothing promoted yet.** Version 1 is a candidate whose gates pass. The project owner must promote it and acknowledge its five fairness flags. The local registry is a SQLite file; CD and the cloud need a shared tracking server.
- **Decisions waiting for the project owner:**
  1. **Data minimisation.** Retrain on the 12 necessary features (governance/FEATURE_INVENTORY.md). This changes the API contract and every metric.
  2. **Fairness.** Choose a mitigation for customers with a thin purchase history; five options with trade-offs are in governance/fairness_audit.md.
  3. **Business numbers.** The coupon cost (5) and margin (25) behind the threshold are placeholders, not the owner's assumptions.
  4. **API image size.** Keep the MLflow registry client in the API image (about 46 MB; uncompressed size now 2,064 MB, just over the brief's 2 GB line), or resolve the registry over plain HTTP instead.
- **Assistant quality.**
  - **Regex:** 39 of 40 tool paths; 15 of 18 expected sources (Q12, Q14 and Q38 lead with the wrong source; Q16 asks about an unavailable customer).
  - **Ollama `llama3.2:3b`:** 80% tool choice, 72% expected sources, 4 in-scope refusals, 4.2 s per question (median of the recorded run; an earlier run measured 2.4 s).
  - **Model-written answers.** They are fluent and in the requested language, but they can contradict the source: a false frozen-returns answer appeared during prompt tuning. The figure check flags such cases but cannot prove correctness.
  - **OpenAI:** not evaluated (no key).
  - **Next steps:** improve chunking and reranking, score answer correctness on a new holdout, consider a larger local model.
- **Retention limitations.**
  - Low returner recall for customers with a thin purchase history.
  - No uncertainty intervals for the headline metrics.
  - No outlier sensitivity analysis.
  - Scores do not estimate causal coupon uplift.
- **Recommendations.** The half-basket offline task, description-based identity and popularity fallback work. Still to do: SKU normalization, curated categories and online evaluation. FP-Growth was slower than Apriori on the bounded comparison; do not claim otherwise.
- **Docker.**
  - **Not yet done:**
    - the linux/amd64 build (CI does it);
    - the fresh-clone test on another machine (section 7.3, Week 11);
    - a cloud image that carries or fetches the Chroma index, the feature store and the embedding model (today they are mounted);
    - verifying the registry inside Compose (it needs the `tracking` profile and a promoted model).
  - **Measured:** the API image is 2,064 MB uncompressed and 553 MB compressed. `/predict` p95 was 49.6 ms in the 2026-11-01 demo and 45.7–57.8 ms in the 2026-10-31 runs.
- **Monitoring.**
  - **Drift reference.** Real 2011 traffic drifts from the 2010 training reference, because tenure features grow with time. Decide whether to refresh the reference with recent data or monitor tenure differently.
  - **Alerts.** No webhook is configured, so alerts reach only the log and the table.
  - **Statistics.** One process only. KS over-alerts on large samples, and there is no multiple-comparison correction.
  - **Labels.** Delayed-label backfill is designed (docs/MONITORING.md), not built.
- **DVC.** There is a local cache and lock file, but no remote and no full `dvc repro` validation. Chroma's mutable SQLite directory becomes dirty during serving.
- **Reproducibility.** There is no hash-locked dependency set: the API rebuild picked up newer package versions, about 25 MB of its growth. The embedding model's revision is not pinned. Tests raise thousands of numpy deprecation warnings. Most come from a test helper, but the pipeline code itself triggers one at `src/features.py:484`, which a future numpy could turn into an error.

## P2 — needs accounts, or a later phase

- **GitHub.** A remote, CI runs, branch protection (a private repository needs GitHub Pro), and the instructor added as a collaborator.
- **AWS.**
  - An OIDC role and an ECR repository.
  - Deployment on App Runner or Render through CD.
  - SageMaker training, tuning and the endpoint, run with evidence and torn down (`scripts/sagemaker_jobs.py`).
  - A cost analysis with real prices.
- **Weekly retraining runs.** They need `ENABLE_RETRAINING`, and ideally a shared MLflow server and a pull-request token.
- **Operations at scale.** A live load study against an SLO, alerts delivered to a person, the backfill built, and multi-worker monitoring.
- **Law 25.** A complete review: a named owner, rights and incident procedures, and retention periods, including for `decision_log`.
- **Data.** A mapping from a grocery catalogue to UCI products, or an appropriately governed grocery dataset.
- **Deliverables.** The demo video, the deployment note, and version tags (v0.9, v1.0, and v0.1–v0.3 for deliverables already submitted).

The local path is complete and verified: API, interface, monitoring, approval step, registry candidate and Docker stack. That is not the same as a finished capstone or a deployable production system.
