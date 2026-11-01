# Remaining work — prioritized, not hidden

## P0 — before any real/customer-facing exposure

- No authenticated identity, authorization, rate limiting or secure customer feature lookup. Bind local servers to localhost only. Matching a caller-supplied ID is not authentication.
- No implemented human approval, appeal/correction workflow, coupon dispatch or support handoff.
- No enforced retention/purge across logs, features, raw data, DVC cache, backups and model lineage. Prediction vectors can be identifying.
- Joblib deserialization requires trusted artifacts. Hash matching alone does not authenticate a publisher.

## P1 — next integration/validation milestone

- **CI never run:** locally on 2026-11-01, strict mypy reports 0 errors in 22 files and coverage is 90% (89% on a clean checkout), above the unchanged 80% gate, with 190 tests passing. The repository has no remote, so no GitHub Actions run, branch protection or green pipeline exists yet. Coverage shows the code paths run; model quality still comes from the real-data runs.
- **Assistant relevance:** 39/40 tool/refusal paths match the development golden set, but expected policy source accuracy is 15/18. Questions 12, 14 and 38 retrieve the wrong leading source. Question 16 references an unavailable historical customer and correctly returns no invented score. Improve paragraph-aware chunking, retrieval/reranking and completeness evaluation on a new holdout.
- **Bilingual generation partial:** French questions retrieve English policy chunks; regex can quote English with a French wrapper. OpenAI was not run. Ollama ran in Compose with the configured `llama3.2:3b` (changed from `llama3.1` on 2026-10-31) on three questions, a wiring check; it has not been evaluated. Inside Docker on a Mac it runs on the CPU (7–18 s per answer), whereas the native Ollama app uses the Apple GPU. Provider evaluation, full fluent bilingual grounded generation and adversarial evaluation remain.
- **Retention limitations:** very low one-time-buyer returner recall at the selected threshold; full subgroup/uncertainty analysis, feature ablation, outlier sensitivity and stronger business-action framing required. Scores do not estimate causal coupon uplift.
- **Recommendations:** half-basket offline task, description-based identity and popularity fallback are working. SKU normalization, curated category support, online evaluation and full unbounded itemset scalability remain. FP-Growth was slower than Apriori on this bounded local comparison; do not claim otherwise.
- **Docker:** both images build and the five-service Compose stack ran locally on linux/arm64 (docs/CONTAINERS.md). Open: the linux/amd64 build (CI does it), a fresh-clone test on another machine, and a cloud image that carries or fetches the Chroma index, feature store and embedding model (the API image alone is live but not ready). The API image is 1,993 MB uncompressed and 534 MB compressed; its p95 `/predict` latency was 45.7, 57.8 and 54.4 ms in three local runs, so not reliably under 50 ms.
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

