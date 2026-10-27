# Defence Preparation — Appendix C Question Bank

> HISTORICAL DRAFT — superseded for the current revision. This file contains old
> target semantics, planned controls and unsupported numeric examples. Do not use
> it as presentation evidence. Use reports/STUDENT_TECHNICAL_GUIDE.md and the
> current progress presentation instead. Preserved to avoid discarding prior work.

**Boréal Marché — Customer Intelligence Platform**
**Prepared by:** Eraldo Ibrushi | **Course:** 420-977-VA

---

## Data and Modelling

### Q4. Walk me through your observation and label windows. Why a 90-day horizon?

The cutoff date is **2011-09-01**, declared in `configs/pipeline.yaml`, not buried in a notebook cell.

- **Observation window:** Every transaction *before* 2011-09-01. This is the data the model is allowed to see — we engineer all 25 features from this window only.
- **Label window:** The 90 days *after* the cutoff (2011-09-01 to 2011-11-30). If a customer placed at least one order in this window, `will_return = True`.

The 90-day horizon is a business choice, not a statistical one. It balances two tensions:

1. **Too short (e.g. 30 days):** Many legitimate customers buy monthly or quarterly — they'd be mislabelled as churned simply because their natural purchase cadence is longer than the window. The false-positive rate would inflate.
2. **Too long (e.g. 180 days):** By the time we confirm a customer has churned, the retention coupon arrives six months late. The intervention window is too narrow to be useful to marketing.

90 days sits in the sweet spot for a grocery retailer: it captures most repeat purchase cycles (the median customer in this dataset reorders within 30–60 days) while leaving enough runway for marketing to act on the flagged list before the customer is truly gone.

There's also a practical constraint: the UCI dataset runs from December 2009 to December 2011. Pushing the cutoff later would shrink the label window past the end of the data, and pushing it earlier would lose observation-window data. 2011-09-01 maximises the amount of usable data on both sides.

---

### Q5. Why PR-AUC rather than accuracy or ROC-AUC for this problem?

**Accuracy is misleading on imbalanced data.** If 62% of customers return, a model that predicts "everyone returns" gets 62% accuracy — and catches zero churners. It looks good on paper and is useless to the marketing team.

**ROC-AUC is insensitive to class imbalance.** ROC-AUC measures the trade-off between the true positive rate and the false positive rate. But FPR is computed against the *negative* class (returners), which is large. A model can accumulate many false positives and still show a low FPR because the denominator is so big. ROC-AUC therefore tends to paint an optimistic picture when the positive class (churners) is in the minority.

**PR-AUC (Average Precision) focuses on the positive class.** It measures the trade-off between precision (of the flagged customers, how many actually churned?) and recall (of the actual churners, how many did we catch?). Both axes involve the positive class, so it punishes a model that cries wolf. For a retention campaign, this is exactly what matters: the marketing team has a limited coupon budget and needs to trust the flagged list.

I still report ROC-AUC as a secondary metric for completeness, and Brier score to verify calibration quality. But the leaderboard is sorted by PR-AUC.

---

### Q6. Your operating threshold is not 0.5. Defend the number you chose.

The default 0.5 threshold assumes equal costs for false positives and false negatives. That assumption is wrong here.

I select the threshold by maximising **expected profit** across a grid of 181 thresholds (0.05 to 0.95):

```
Expected profit = TP × (margin_retained - cost_coupon) + FP × (-cost_coupon) + FN × (-cost_FN)
```

The cost assumptions (stated in `configs/pipeline.yaml`):
- **Cost of sending a coupon:** $5.00
- **Average margin from a retained customer:** $25.00
- **Cost of missing a churner (false negative):** $0 (no coupon sent, no money spent)

The asymmetry is 5:1 — saving a real churner is worth five times more than the wasted coupon sent to a loyal customer. This pushes the optimal threshold *below* 0.5, because we'd rather over-send a few cheap coupons than miss high-value churners.

The exact threshold is computed on the **calibrated** validation set and recorded in `artifacts/retention_model_metadata.json`. It shifts with every retraining cycle. The business should review the cost assumptions quarterly — if coupon costs rise or margins shrink, the threshold moves.

**Why not just rank?** Because the marketing team doesn't always want a fixed top-K — they want a yes/no flag with an adjustable budget. The threshold turns a continuous score into an actionable decision. But the score and the rank are always available in the API response for users who prefer to work with probabilities directly.

---

### Q7. What does calibration buy you if you only ever rank customers?

If you only rank, calibration is unnecessary — any monotone transformation of the scores preserves the ranking. But in practice we don't only rank:

1. **The threshold is a probability.** My profit-optimal threshold is a point on the probability axis. If the model says 0.35 but the true probability is 0.55, the threshold misfires. Calibration ensures that "0.35 means 35% probability of churning," so the cost-based threshold selection works correctly.

2. **The score is customer-facing.** Under Law 25, a customer can ask why they received (or didn't receive) a coupon. The API returns `probability: 0.72` and `risk_label: high`. If the model's probabilities are badly calibrated — say, it reports 0.72 when the true rate is 0.45 — we're lying to the customer. Calibration makes the explanation honest.

3. **Mixing models across segments.** If we later calibrate different models per region (e.g. domestic vs. international), their scores need to be comparable. Uncalibrated scores from different models can't be compared on the same scale.

4. **Business communication.** "There's a 70% chance this customer won't return" is actionable. "This customer has a score of 2.3" is not. Calibrated probabilities let non-technical stakeholders interpret the output without asking what the number means.

I apply isotonic regression (non-parametric, flexible) and sigmoid calibration (Platt scaling, parametric), compare both by Brier score on the validation set, and keep the one with the lower Brier score. The before/after reliability curves are logged as MLflow artifacts so the improvement is visible.

---

### Q8. Which feature mattered most, and did that surprise you?

After training, the model's `feature_importances_` attribute (from HistGradientBoosting) ranks the features. Based on the RFM literature and the structure of this problem, the expected top features are:

- **`recency_days`** — consistently the strongest single predictor in nearly every retention model. Customers who bought recently are dramatically more likely to return. This is unsurprising; recency dominates RFM analysis.
- **`frequency`** — the second most important. A customer with 20 orders behaves differently from a one-timer.
- **`is_one_time_buyer`** — strongly anti-predictive. One-timers almost never return.

**What might surprise:** Features like `category_entropy` or `single_item_basket_ratio` can rank higher than `monetary_total`. This makes business sense — a customer who buys from many categories is exploring the store, signalling engagement. A high single-item basket ratio suggests impulse buys or gift purchases, not habitual shopping.

The error analysis in `train.py` reports feature importances and segments where the model fails, so the answer is grounded in actual data, not speculation. The top factors are also returned per-prediction through the API (`top_factors` in the response), making the model interpretable at the individual customer level.

---

### Q9. Suppose your model loses to the business rule on recall. What do you do?

The business heuristic is "flag everyone whose last purchase was more than 60 days ago." If it achieves higher recall than the tuned model, that's a legitimate finding — not a failure to hide.

First, **verify it's real.** Check if the recall gap holds across all CV folds, not just one unlucky split. If it's consistent:

1. **Report it honestly.** The project spec explicitly rewards honest negative results (+5 bonus on D7). I'd document it in the validation report: "The recency heuristic achieves X% recall vs. the model's Y% recall."

2. **Check precision.** Recall alone isn't the full story. The business rule probably has terrible precision — it flags every inactive customer, including seasonal buyers who always come back in December. The model likely has better precision and therefore better PR-AUC, which means the marketing budget is better allocated.

3. **Check the profit curve.** Even if recall is lower, the model's profit-optimal operating point may generate more expected profit because it wastes fewer coupons on false positives. The cost-based evaluation resolves the ambiguity.

4. **Investigate the error segments.** Where does the model underperform? If it's failing on one-time buyers with high recency (they look recent but never come back), that's a feature-engineering problem — maybe I need a feature like "time since registration without a second order."

5. **Consider an ensemble.** The simplest win is often: flag anyone the heuristic *or* the model catches. This combines the heuristic's recall with the model's precision. Document the trade-off.

**What I would NOT do:** tune the model until it beats the baseline on recall by cherry-picking hyperparameters. That's overfitting the leaderboard.

---

### Q10. How did you protect your test set, and how many times did you look at it?

The test set is created **once** by `train_test_split` with `test_size=0.2`, stratified by the target, using `random_state=42` for reproducibility. It is stored in memory — the file is not written separately.

**During spot-check and tuning, the test set is never touched.** All model comparison and hyperparameter selection use 5-fold stratified cross-validation on the training set only. Optuna's 100 trials all evaluate on CV folds — they never see the test rows.

**The test set is looked at exactly once:** after the final model is selected, calibrated, and the threshold is set. This single evaluation produces the final metrics reported in the README and model card. The code path is linear — `train.py` does spot-check → HPO → calibrate → threshold → final eval → save. There is no loop back.

The temporal structure adds a natural safeguard: the observation window (before 2011-09-01) and the label window (after 2011-09-01) do not overlap. The model cannot "peek" at future transactions.

A golden-prediction regression test in `tests/golden/golden_predictions.json` ensures that the model's outputs are deterministic — if someone accidentally retrains and overwrites the artifact, the test catches it.

---

## Recommendation

### Q11. Why association rules rather than collaborative filtering? When would you switch?

Association rules are the right choice for this project for three reasons:

1. **Interpretability.** The spec requires that "the interface can explain the recommendation." A rule like `{bread, butter} → {jam} (confidence 72%, lift 3.1)` is self-explanatory. Try explaining a 64-dimensional latent factor to a marketing analyst.

2. **Cold-start friendliness.** Association rules work on the *current basket*, not on user history. A first-time visitor who puts bread in their cart gets recommendations immediately. Collaborative filtering needs a user profile — it fails on cold-start.

3. **Offline/online separation.** Mining runs once (offline, minutes); serving is a hash-table lookup (online, microseconds). This separation is architecturally clean and teaches the right pattern for production ML systems — which is the pedagogical point of Head B.

**When I'd switch to collaborative filtering (or embeddings):**
- When the product catalogue grows to tens of thousands of SKUs and association rules suffer from coverage issues (long-tail items never appear in any rule).
- When we have persistent user profiles and want to personalise recommendations based on individual history, not just the current basket.
- When the business wants "customers like you also bought" rather than "items frequently bought together."

The README's "What I Would Do Next" section lists embedding-based recommendations as the first upgrade.

---

### Q12. Explain the lift trap: high confidence with low lift. Show me an example from your data.

**Confidence** = P(consequent | antecedent). **Lift** = confidence / P(consequent).

The trap: if the consequent is an item *everybody* buys, confidence can be high even when the rule tells you nothing. Example:

Suppose "WHITE HANGING HEART T-LIGHT HOLDER" appears in 40% of all baskets. A rule like:

> {JUMBO BAG RED RETROSPOT} → {WHITE HANGING HEART T-LIGHT HOLDER}
> confidence = 0.45, lift = 0.45 / 0.40 = **1.12**

This rule has 45% confidence — it *looks* strong. But the lift is barely above 1.0, meaning the customer was almost as likely to buy the t-light holder anyway. Recommending it adds almost no value — it's like suggesting "bread" at a grocery store.

**My filtering rule:** `configs/pipeline.yaml` sets `min_lift: 1.0`, so rules with lift ≤ 1.0 (the item is *less* likely given the antecedent) are discarded. In practice, I sort by lift (highest first) in the serving engine, which pushes genuinely surprising co-purchases to the top and buries the obvious ones.

The spec asks for "at least one concrete example of a high-confidence, low-lift rule that must be discarded, and an explanation of why." This is that example.

---

### Q13. Why can you not run the mining algorithm on every request?

FP-Growth scans the entire transaction matrix to find frequent itemsets, then generates candidate rules. On the Boréal Marché dataset (~400K cleaned transactions), this takes **seconds to minutes** depending on the support threshold.

A customer request should return in **under 150 ms** (the target in Appendix B). Running FP-Growth per request would introduce seconds of latency and multiply CPU cost by the number of concurrent users.

The solution: **mine offline, serve online.** `src/rules.py` runs once (or on a weekly schedule via `scripts/retrain.py`), exports rules to `artifacts/rules.json` with a header recording the mining date, parameters, data hash, and rule count. The API loads this JSON at startup into an inverted index (`lifespan.py`), and recommendation requests do an O(1) set-intersection lookup.

This separation generalises to every ML serving problem: training and batch inference are expensive; serving must be cheap.

---

### Q14. How would your serving engine behave with 500,000 rules instead of 1,400?

The `RuleEngine` in `api/lifespan.py` builds an inverted index keyed by `frozenset(antecedent)`. At query time, it iterates over the index keys and checks `key.issubset(basket_set)`.

With 500K rules:

- **Memory:** Each rule is a dict with antecedent (list of strings), consequent (list of strings), confidence, lift, support. At ~500 bytes per rule, 500K rules ≈ 250 MB in memory. Non-trivial but manageable on a 1 GB container.
- **Startup time:** Loading and indexing 500K rules from JSON would take a few seconds instead of milliseconds. Acceptable — it happens once at startup.
- **Query latency:** The bottleneck is iterating over the index. With 500K rules there could be 100K+ unique antecedent keys. The subset check is O(len(key)) for each, and we check all keys. This would push query time from microseconds to tens of milliseconds — still within the 150 ms budget, but getting close.

**What I'd change first:**
1. **Prefix filtering.** Only check keys whose items all appear in the basket. Build a secondary index from individual items → sets of antecedent keys. Intersect the candidate sets before doing subset checks.
2. **Trie-based lookup.** A hash-map-of-frozensets doesn't scale to millions of keys. A trie (prefix tree) over sorted antecedent items gives logarithmic lookup.
3. **Prune aggressively.** Do we really need 500K rules? Tighten `min_support` and `min_lift` to keep only high-signal rules. The top 5K rules by lift probably cover 95% of recommendations.

---

### Q15. What happens when the basket matches no rule, and how does the interface stay honest about it?

The `RuleEngine.recommend()` method has three explicit strategies:

1. **`empty_basket`** — The basket is empty. Return the popularity fallback (most frequently purchased items).
2. **`rule_based`** — At least one rule's antecedent is a subset of the basket. Return the top-K consequents by lift.
3. **`popularity_fallback`** — The basket is non-empty but no rule matches. Return the popularity fallback.

The API response always includes a `strategy` field that tells the frontend *which* path fired. The Streamlit UI uses this:

- For `popularity_fallback`: shows a caption "ℹ️ No matching rules — showing popular items instead."
- For `empty_basket`: shows "ℹ️ Empty basket — showing trending items."

This is the "honesty" requirement from Section 3.3 of the spec: the interface never pretends a popularity recommendation came from a personalised rule. The customer and the analyst can both see the difference.

---

## Generative AI

### Q16. Retrieval-augmented generation or fine-tuning? Justify the choice.

RAG is the right choice here for four reasons:

1. **The corpus changes.** Policy documents (delivery windows, allergen lists, loyalty terms) are updated by the business — potentially weekly. Fine-tuning requires retraining the model every time a policy changes. RAG just updates the document in `docs/corpus/` and rebuilds the index (seconds).

2. **Grounding and citations.** RAG naturally produces citations — every answer can point to the specific chunk it drew from. Fine-tuning internalises the knowledge, making it impossible to trace which document informed the answer. Under Law 25, the customer has a right to know how decisions are made. Citations provide that transparency.

3. **Hallucination control.** A fine-tuned model can confidently produce answers that sound like policy but are fabricated. With RAG, if the retriever finds nothing above the similarity floor, the system refuses rather than guessing. The guardrail pipeline (`src/rag/guardrails.py`) enforces this.

4. **Cost and compute.** Fine-tuning GPT-4o-mini or Llama 3.1 on a 15-page corpus is disproportionate — the corpus is too small to justify the cost, the risk of overfitting to training examples, and the ongoing retraining overhead.

**When fine-tuning would be better:** If the task required the model to learn a *style* (e.g. Boréal Marché's brand voice) or *reasoning patterns* not present in the documents. For factual Q&A over a small, frequently updated corpus, RAG wins.

---

### Q17. How do you prevent the assistant from inventing a product identifier or a probability?

This is enforced at three levels:

**Level 1 — Architecture.** The providers (`src/rag/providers.py`) do NOT generate customer-facing text. Their only job is to decide which tool to call and extract its arguments. The system prompt explicitly states: "You NEVER answer the customer directly. You NEVER invent data."

**Level 2 — Tool boundary.** When the provider says `tool: "retention_score"`, the orchestrator calls `retention_score()` in `src/rag/tools.py`, which sends an HTTP request to the `/predict` endpoint. The probability comes from the real model, not from the LLM. Same for recommendations: the tool calls `/recommend`, which uses the real rule engine. The LLM never sees the numeric output until it's already been computed by the model.

**Level 3 — Typed validation.** Every tool has a Pydantic input model and output model. The LLM's arguments are validated against `RetentionScoreInput`, `RecommendationsInput`, etc. If the LLM tries to pass a probability as an argument (which no input model accepts), Pydantic rejects it.

The regex provider demonstrates this most clearly: it uses pattern matching to extract a customer ID and route to the right tool. It has no capacity to generate a number. The OpenAI and Ollama providers use JSON mode with the same constraint — they output `{"tool": "retention_score", "args": {"customer_id": "12345"}}`, not `{"probability": 0.73}`.

---

### Q18. How do you measure whether the assistant is any good? Show me the numbers.

The spec requires a golden set of 40 questions (15 policy, 10 tool-call, 8 out-of-scope, 7 French), scored on:

1. **Tool-choice accuracy** — Did the provider select the correct tool? (e.g. a delivery question should route to `policy_search`, a retention question to `retention_score`)
2. **Groundedness** — Does the answer come from a retrieved chunk, not hallucinated?
3. **Refusal rate on out-of-scope questions** — Should be 100% for the 8 OOS questions.
4. **Latency and cost per query** — by provider.

The evaluation runs per provider (regex, OpenAI, Ollama), producing a comparison table:

| Provider | Tool-choice accuracy | Refusal rate (OOS) | Avg. latency | Cost/1K queries |
|---|---|---|---|---|
| Regex | ~85% | ~75% | <10ms | $0 |
| OpenAI (gpt-4o-mini) | ~92% | ~95% | ~800ms | ~$0.15 |
| Ollama (llama3.1) | ~88% | ~90% | ~2s | $0 (local) |

The regex baseline is the floor — it proves how far pattern matching gets you. The OpenAI provider should beat it on nuanced routing (e.g. "Can I get a refund for my frozen blueberries?" requires understanding that this is a return question, not a product question). The gap between regex and LLM providers is the measurable value the language model adds.

The tool-choice portion of this evaluation must run in CI so it's automated and reproducible.

---

### Q19. What happens when retrieval returns nothing relevant, and how did you test that?

The `check_similarity_floor` guardrail in `src/rag/guardrails.py` handles this. It checks the cosine similarity score of the best-matching chunk:

- If **no chunks are returned** (empty context): the guardrail fires immediately with a bilingual refusal: "I couldn't find any information about that in our policies. Please contact our support team."
- If the **best score is below the threshold** (0.25, configurable in `pipeline.yaml`): the guardrail fires with a handoff message: "I'm not confident I have the right answer. Let me hand you off to a human agent."

Both paths include the support email and phone number — the customer is never left hanging.

**How I tested it:** The golden set includes 8 out-of-scope questions like "What's the weather in Montreal?" and "Write me a poem about groceries." The `check_out_of_scope` guardrail catches obviously off-topic questions via keyword patterns, and the similarity floor catches subtler ones where the question looks plausible but no policy document covers it.

The tests in `tests/test_chat.py` verify that these guardrails fire correctly. Edge cases: a question about a real but undocumented policy (e.g. "What's your policy on live animals?") — no chunk should score above 0.25, so the system correctly refuses rather than hallucinating a policy.

---

### Q20. Why offer a locally hosted model option at all?

Two reasons — one technical, one legal:

**1. Data residency (Law 25 / privacy).** When a customer types "What's the status of my order #12345?" into the assistant, the message is sent to whichever LLM provider is selected. If that provider is OpenAI, the message leaves Canada and travels to OpenAI's servers. Under Québec's Law 25, transferring personal information outside the province (or Canada) requires additional safeguards — a privacy impact assessment, ensuring the receiving jurisdiction offers equivalent protections, and informing the individual.

The Ollama provider runs **entirely locally** — customer messages never leave the deployment. This is not just a nice-to-have; it's the legally conservative default. The `pipeline.yaml` sets `default_provider: "regex"` (no external calls at all), and Ollama is the recommended upgrade path for production.

**2. Demonstrating where an LLM adds value.** By offering three providers behind the same interface, I can compare tool-choice accuracy, latency, and cost. The regex baseline proves that simple routing works for ~85% of queries. The LLM providers only justify their cost if they handle the remaining 15% (ambiguous questions, French-English code-switching, complex multi-part queries) measurably better.

If the LLM providers showed no improvement over regex, that would be a legitimate finding — it would mean the problem doesn't need a language model. The three-provider design lets me prove or disprove that empirically.

---

## Engineering and MLOps

### Q21. joblib or pickle? What are the risks of each?

Both execute arbitrary code on deserialization — a malicious `.pkl` or `.joblib` file can run shell commands when loaded. This is the fundamental risk.

**Why joblib over pickle:**
- **Compression.** Joblib compresses numpy arrays efficiently, producing smaller files.
- **Transparency.** Joblib's internal format is explicit about numpy/scipy dtypes and versions. A version mismatch raises a clear error rather than silently producing wrong outputs.
- **Ecosystem fit.** scikit-learn's `save`/`load` documentation recommends joblib.

**Why not ONNX?** ONNX would be the zero-trust option — it serialises only the mathematical graph, not executable Python. But my pipeline includes scikit-learn preprocessing (StandardScaler, label encoding) that ONNX cannot represent without custom operators. Exporting only the estimator would break the pipeline guarantee: the served model would need separate preprocessing code that could drift from training.

**Mitigation:** The model artifact is built by our own pipeline and stored in `artifacts/`. It's never loaded from user input or an untrusted source. The metadata sidecar (`retention_model_metadata.json`) records the git commit, library versions, and training timestamp, so we can verify provenance.

---

### Q22. What exactly is contained in a "model version"?

The `artifacts/` directory contains two files that together constitute a model version:

**1. `retention_model.joblib`** — The complete inference pipeline. This is a `CalibratedClassifierCV` wrapping a `HistGradientBoostingClassifier`. If the model used preprocessing (e.g. `StandardScaler`), it's a `Pipeline` object with preprocessing and estimator together. One file, one `predict_proba()` call — no external preprocessing code needed.

**2. `retention_model_metadata.json`** — The sidecar with everything a human or system needs to trust and reproduce the artifact:
- `model_version`: semantic version (0.1.0)
- `trained_at`: ISO timestamp
- `git_sha`: the commit that produced the artifact
- `feature_names`: ordered list of the 25 features the model expects
- `feature_count`: 25
- `decision_threshold`: the profit-optimal threshold
- `metrics`: PR-AUC, ROC-AUC, Brier, recall@top-decile on the test set
- `calibration`: Brier scores and reliability-curve data for isotonic, sigmoid, and uncalibrated
- `threshold_analysis`: the profit curve and cost assumptions
- `error_analysis`: segment-level error rates and feature importances
- `pipeline_steps`: ordered list of pipeline step names
- `libraries`: exact versions of scikit-learn, optuna, mlflow

Additionally, every training run is logged to **MLflow** with parameters, metrics, the git SHA, and the serialised model. The MLflow model registry allows tagging versions as Staging or Production with a promotion policy.

---

### Q23. What is the difference between /health and /ready, and why does a load balancer care?

**`/health` (liveness probe):** Returns 200 if the process is alive. It checks nothing — if FastAPI can respond to HTTP, the process is running. If this fails, the container is **dead** and should be restarted (killed and replaced).

**`/ready` (readiness probe):** Returns 200 only if the models are loaded and the system can serve real predictions. It checks `app.state.model`, `app.state.rule_engine`, and `app.state.rag_index`. If any are missing, it reports which ones are loaded and returns a degraded status.

**Why a load balancer cares:** During startup, the container takes 10–30 seconds to load the joblib model, parse the rules JSON, and build the RAG index. During this window, `/health` returns 200 (the process is alive) but `/ready` returns not-ready (models aren't loaded yet).

A load balancer that routes traffic based on `/health` would send requests to a container that can't serve predictions — every request would 500. By routing based on `/ready`, the load balancer waits until the container is warm before sending traffic.

Kubernetes uses the same pattern: `livenessProbe` → `/health` (restart if dead), `readinessProbe` → `/ready` (remove from service pool if not ready). Docker Compose's healthcheck in `docker-compose.yml` uses the same `/health` endpoint.

---

### Q24. Why OIDC rather than storing AWS keys as repository secrets?

Stored credentials are **static, long-lived secrets** with several risks:

1. **Rotation burden.** If a key leaks (accidental log output, a dependency vulnerability, a stolen laptop), every repository that uses it must be updated. AWS keys don't expire unless you explicitly rotate them.
2. **Blast radius.** A repository secret is available to every workflow in the repository. A compromised workflow (malicious PR, supply-chain attack on an action) can exfiltrate the key.
3. **Audit opacity.** When three CI jobs use the same static key, CloudTrail can't distinguish which job did what.

**OIDC (OpenID Connect)** eliminates stored secrets entirely:

1. GitHub Actions requests a short-lived token from AWS STS by presenting a signed JWT. The token lives for the duration of the workflow run — typically minutes.
2. The AWS IAM role's trust policy is scoped: it only accepts tokens from this specific repository, this specific branch, and this specific workflow. A forked repository or a PR from an outside contributor cannot assume the role.
3. No secret to rotate, leak, or exfiltrate. The trust relationship is in AWS IAM, not in GitHub.

The `cd.yml` workflow uses `aws-actions/configure-aws-credentials@v4` with `role-to-assume` — no `aws-access-key-id` or `aws-secret-access-key` fields.

---

### Q25. Your scheduled retraining opens a pull request instead of deploying. Why?

Because **automated deployment of a retrained model is dangerous.** The retraining pipeline runs weekly on a cron schedule. If the new data has a quality issue, the new model could be worse than the current one. Auto-deploying it would degrade production with no human check.

The design:

1. `scripts/retrain.py` re-runs ETL, re-mines rules, retrains the model, and checks promotion criteria (new PR-AUC must be within 1% of production).
2. If criteria pass, the CD workflow creates a branch (`retrain/2026-09-21`), commits the updated artifacts, and opens a pull request.
3. The PR body says: "⚠️ This does NOT auto-deploy. A human must review the MLflow experiment comparison, verify metrics, approve and merge."
4. Only a merge to `main` triggers the deployment pipeline (build → push → App Runner update → smoke test).

This gives the team:
- A diff to review (what changed in the artifacts?)
- CI checks on the new artifacts (tests, golden predictions)
- An MLflow comparison link to see metrics side by side
- The ability to reject the retrain if something looks wrong

The spec explicitly states: "It must not deploy automatically. A human merges."

---

### Q26. How would you know tomorrow morning if this model had degraded overnight?

The monitoring pipeline has three layers:

**1. Input drift (immediate).** The `/metrics` endpoint computes PSI (Population Stability Index) and KS (Kolmogorov-Smirnov) statistics on each feature, comparing incoming traffic against the training-time distribution. If PSI exceeds 0.2 or KS exceeds 0.05, the monitoring dashboard flips to yellow/red. This catches data pipeline bugs (a new ETL job corrupts a feature) or distribution shifts (a marketing campaign changes the customer mix).

**2. Operational metrics (immediate).** Request count, error rate, and latency percentiles (p50, p95, p99) are tracked in the in-memory prediction log. A sudden latency spike or error rate above 10% triggers a red status. If the model file got corrupted or a dependency broke, this catches it.

**3. Prediction drift.** The distribution of predicted probabilities is compared against the training-time distribution. If the model suddenly predicts everyone as high-risk or low-risk, that's a signal the model or the features have shifted, even if no individual feature triggered PSI.

**What's missing (by design):** Ground truth. The label "did the customer return?" is only observable 90 days after the prediction. There's no way to measure actual accuracy overnight. This is why input drift matters — it's the early-warning proxy until ground truth arrives. See Q27.

The `scripts/simulate_traffic.py` script populates the dashboard with normal traffic and deliberately shifted cohorts (higher recency, different country mix) to demonstrate that the alerts fire correctly.

---

### Q27. Your ground-truth label arrives 90 days late. What do you monitor in the meantime?

This is the delayed-ground-truth problem. For 90 days, I cannot compute real accuracy, precision, or recall. Here's what I can do:

**1. Input drift (continuous).** PSI and KS on every feature tell me if the customers I'm scoring today look different from the customers I trained on. If recency values suddenly spike, or if the country distribution shifts, the model's assumptions may no longer hold.

**2. Prediction distribution drift.** Are the predicted probabilities stable? If the model was outputting a mean probability of 0.35 and it suddenly jumps to 0.55, something changed — even if I don't know yet whether the new predictions are right.

**3. Operational health.** Latency, error rate, throughput. These catch infrastructure degradation, not model degradation, but they're table stakes.

**4. Proxy labels (partial ground truth).** Some labels arrive faster than 90 days. A customer who makes a purchase 10 days after being scored is a confirmed positive — their label resolves early. By tracking these early positives, I can compute a lower bound on recall. If the fraction of early positives drops relative to historical patterns, that's a soft signal.

**5. Backfill.** Once the 90-day window closes for a cohort, I compute retrospective accuracy and backfill the monitoring dashboard. This is the "backfill design" the spec mentions. The first true accuracy signal for today's predictions arrives on December 15 (90 days from today). The monitoring dashboard documents this delay explicitly.

---

### Q28. When would managed cloud training have been worth it for this project, and why was it not?

The `train.py` script honours `SM_CHANNEL_TRAIN` and `SM_MODEL_DIR`, so it runs as a SageMaker training job without modification. But for this project, managed training was not worth it:

**Cost.** An Optuna run with 100 trials on ~4,300 customers and 25 features takes **5–15 minutes on a laptop CPU.** The dataset fits in memory. SageMaker's minimum billing is per-second with a startup overhead of ~2 minutes per job. Running 100 Optuna trials as separate SageMaker jobs would cost more in startup overhead than the actual computation.

**Complexity.** SageMaker requires an ECR image, IAM roles, S3 buckets for data and artifacts, and a job definition. For a dataset that fits in a laptop's RAM, this infrastructure adds cost and debugging surface without a performance benefit.

**When it would be worth it:**
- **Data scale.** If the transaction table grew to 100M+ rows and feature engineering required distributed compute (Spark, Dask), SageMaker's managed infrastructure would justify itself.
- **Model complexity.** If the model were a neural network requiring GPU hours, SageMaker's managed spot instances would reduce cost by 60–70%.
- **Team scale.** If multiple data scientists are running experiments concurrently, SageMaker provides job scheduling, resource management, and experiment tracking that a single laptop doesn't.
- **Reproducibility in regulated environments.** SageMaker logs the exact container, data, and code for every training job. In a regulated setting (banking, healthcare), this audit trail may be required.

For a ~4K customer dataset with tree-based models, local training is faster, cheaper, and simpler. The SageMaker compatibility is there for when the data outgrows the laptop.

---

## Governance

### Q29. Which decisions in your system are based exclusively on automated processing, and what does that oblige you to do?

Three automated decision points, documented in `governance/GOVERNANCE_LOI25.md`:

**Head A — Retention prediction:** A probability score and risk label that determines whether a customer is flagged for a coupon campaign. This is *not* fully automated — a human marketing analyst reviews and approves the flagged list before any coupon is sent. Still, the model produces the initial recommendation, which triggers Law 25 obligations.

**Head B — Product recommendations:** Fully automated — suggestions appear in real time without human review. However, the decision is a product suggestion (analogous to shelf placement), not a consequential decision affecting the customer's rights. The rule set is reviewed by a merchandising analyst before deployment.

**Head C — Customer assistant:** Partially automated — the assistant generates responses, but guardrails enforce refusal for out-of-scope questions and a human-handoff path when confidence is low. The assistant never modifies customer data.

**What Law 25 obliges me to do:**

1. **Inform the individual** — before or at the time the decision is made — that automated processing was used. The Streamlit "About" tab displays this notice in both French and English.
2. **Disclose the principal factors.** The API returns `top_factors` with each prediction. The recommendations endpoint returns the firing rule with confidence and lift.
3. **Provide a right to human review.** The governance document specifies the path: contact support, and a human re-evaluates.
4. **Allow access and correction.** The customer can request to see or correct their data.

The automated-decision register in `GOVERNANCE_LOI25.md` documents all of this for each head, including what personal information is used, what is explicitly *not* used, and the retention period for logs.

---

### Q30. You found a recall gap for one subgroup. Walk me through your options and their trade-offs.

The fairness audit (`governance/fairness_audit.md`) checks recall, precision, and selection rate by region, tenure band, and spend quartile. Any group with recall more than 5 percentage points below overall recall is flagged.

**Expected gap:** International customers (~10% of data) likely have lower recall — the model has fewer examples to learn from, and their patterns differ (longer shipping times, different product preferences).

**Option 1 — Do nothing, document it.**
- *Trade-off:* Honest, but the affected group receives worse service. Under Law 25, failing to flag at-risk international customers means they miss retention coupons. Acceptable only if the gap is small and the business acknowledges it.

**Option 2 — Adjust the threshold per subgroup.**
- Use a lower threshold for the underperforming group (e.g. international customers get flagged at 0.25 instead of 0.35). This boosts their recall at the cost of precision — more coupons wasted on false positives in that group.
- *Trade-off:* Improved fairness, but higher cost. The coupon budget for international customers increases disproportionately. The business must decide if equity justifies the spend.

**Option 3 — Collect more data or engineer group-specific features.**
- The gap may be caused by data scarcity. International customers have fewer transactions, so feature estimates (e.g. `avg_days_between_orders`) are noisier. Adding features like `shipping_region_latency` or oversampling the minority group during training could close the gap.
- *Trade-off:* Most principled, but requires time and new data. Does not help the current model immediately.

**What I would NOT do:** Add demographic features (age, gender, ethnicity) to "fix" the gap. This introduces new fairness risks, violates data minimisation, and the data isn't even available.

I'd recommend Option 2 as the immediate mitigation (it's deployable now) while pursuing Option 3 for the next model version. Both the gap and the mitigation are documented in the fairness audit with exact numbers.

---

### Q31. The client wants to send customer data to a hosted LLM provider outside Canada. What is your advice?

**My advice: don't, unless you've completed a privacy impact assessment and can demonstrate equivalent protection.**

Under Québec's Law 25, transferring personal information outside the province requires:

1. **A privacy impact assessment (PIA)** evaluating the sensitivity of the data, the purpose of the transfer, and the protections available in the receiving jurisdiction.
2. **Contractual safeguards** — a written agreement with the provider ensuring equivalent protections (comparable to what the Commission d'accès à l'information would require).
3. **Informing the individuals** whose data is being transferred.

The assistant receives customer messages that may contain order numbers, names, account identifiers, and the substance of their complaint. Sending this to OpenAI's US servers means the data is processed under US law, which does not offer the same protections as Québec's privacy framework.

**My recommendation:**

1. **Default to the local provider (Ollama).** Customer messages never leave the deployment. This eliminates the compliance question entirely.
2. **If an external LLM is needed for quality:** Use the OpenAI provider only for questions that do NOT contain personal information. Strip or anonymise customer identifiers before sending the prompt. Implement this as a pre-processing guardrail.
3. **If the client insists on full external LLM usage:** Engage a privacy lawyer, complete the PIA, sign a DPA (Data Processing Agreement) with the provider that includes data residency guarantees (some providers offer Canadian data residency), and update the privacy notice to inform customers.

The three-provider architecture was designed with exactly this scenario in mind. The regex provider and the Ollama provider are the data-residency-safe options. The OpenAI provider exists for demonstration and evaluation, not as the production default.

---

### Q32. A customer asks why they did not receive an offer. What can you tell them, and how quickly?

**What you can tell them:**

1. **The score and the threshold.** "Your retention probability was 0.72, which is above our intervention threshold of 0.45. This means the model predicted you were likely to return on your own, so you were not flagged for a retention coupon."

2. **The top factors.** The API returns `top_factors` with every prediction — the features that most influenced the score. For example: "The main factors were your recent purchase (12 days ago), your order frequency (15 orders in the past year), and your basket diversity (8 categories). These all indicated high engagement."

3. **The right to review.** "If you believe this assessment is incorrect, you have the right to request a human review. A member of our marketing team will re-evaluate your case."

**How quickly:**

The prediction is stored in the in-memory prediction log (`api/lifespan.py`), which holds up to 50,000 structured records. Looking up a customer's most recent prediction is a linear scan of the log — sub-second for the in-memory buffer.

For older predictions (beyond the buffer), the structured JSON logs are persisted (retention: 12 months per the governance policy). A log query by customer ID would return the prediction within minutes.

The explanation can be delivered **immediately** if the customer contacts support while the prediction is still in the log. For older predictions, it requires a log query — minutes, not days. The key design decision is that **every prediction is logged with its full feature vector, score, threshold, and top factors** at the time it's made. We never need to re-run the model to explain a past decision.

This satisfies Law 25's requirement that a customer be able to learn the principal factors behind an automated decision and request a human review.
