# Remaining work — prioritized, not hidden

## P0 — before any real/customer-facing exposure

- No authenticated identity, authorization, rate limiting or secure customer feature lookup. Bind local servers to localhost only. Matching a caller-supplied ID is not authentication.
- No implemented human approval, appeal/correction workflow, coupon dispatch or support handoff.
- No enforced retention/purge across logs, features, raw data, DVC cache, backups and model lineage. Prediction vectors can be identifying.
- Joblib deserialization requires trusted artifacts. Hash matching alone does not authenticate a publisher.

## P1 — next integration/validation milestone

- **CI not green:** strict mypy reports 77 errors in 13 files; executed coverage is about 64%, below the unchanged 80% gate. Tests and lint passing do not mean all quality gates pass. Expand meaningful offline orchestration/provider tests and proper typing; do not lower assertions/gates.
- **Assistant relevance:** 39/40 tool/refusal paths match the development golden set, but expected policy source accuracy is 15/18. Questions 12, 14 and 38 retrieve the wrong leading source. Question 16 references an unavailable historical customer and correctly returns no invented score. Improve paragraph-aware chunking, retrieval/reranking and completeness evaluation on a new holdout.
- **Bilingual generation partial:** French questions retrieve English policy chunks; regex can quote English with a French wrapper. OpenAI/Ollama routing implementations were not run; full fluent bilingual grounded generation and adversarial evaluation remain.
- **Retention limitations:** very low one-time-buyer returner recall at the selected threshold; full subgroup/uncertainty analysis, feature ablation, outlier sensitivity and stronger business-action framing required. Scores do not estimate causal coupon uplift.
- **Recommendations:** half-basket offline task, description-based identity and popularity fallback are working. SKU normalization, curated category support, online evaluation and full unbounded itemset scalability remain. FP-Growth was slower than Apriori on this bounded local comparison; do not claim otherwise.
- **Docker:** Compose config validated, but Docker daemon was unavailable, so build/runtime/networking are unvalidated.
- **DVC:** local cache and lock generated, no remote or full dvc repro validation. Chroma's mutable SQLite directory becomes dirty during serving; separate immutable index export from writable runtime state.
- **Reproducibility:** exact package snapshot recorded, but no portable hash-locked dependency set or pinned embedding revision. Data/source/config hashes record generation-time state; source formatting/documentation changed afterward. Older ingest/feature config hashes differ after removal of an unused mining config key, which does not change those transformations.

## P2 — next phase / not executed

- MLflow model registry aliases, promotion comparison and reviewed rollback.
- Monitoring log rotation, multi-worker aggregation, multiple-testing-aware drift analysis, alert delivery, delayed-label backfill and performance tracking.
- SageMaker HPO/spot training/endpoints, AWS IAM/ECR/App Runner deployment, live SLO/load study, cloud cost analysis.
- Actual CI/CD runs in GitHub, deployment approvals/secrets, container security validation.
- Complete Law 25/privacy review, documented owner and operational rights/incident procedures.
- Grocery catalogue ↔ UCI product mapping, or an appropriately governed genuine grocery dataset.

There are no remaining blockers to the demonstrated **local regex/API/Streamlit path** with the generated artifacts. This is not equivalent to a finished capstone or deployable production system.

