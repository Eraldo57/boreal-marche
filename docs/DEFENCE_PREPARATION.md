> Current artifact: 0.3.0, contract 2.0, 12 serving features, sigmoid calibration, 20.5% threshold.
> September was reused; the archive contains earlier 25-feature results. Use MODEL_CARD.md and
> retention_diagnostics.json if a historical example below conflicts with the current artifact.

# Appendix C defence answers — evidence and limits

Revised 2026-11-05 against the original Capstone_Project1.pdf, pp. 30–31, current code and
saved results. Questions start at 4 in the PDF; the numbering below preserves that source.
Use [the cue card](DEFENCE_CUE_CARD.md) for navigation and [the walkthrough](PRESENTATION_D7.md)
for timing. These are prompts for reasoning, not a speech to memorize.

Say whether a statement is measured, configured, planned or unknown. Saved model results are
historical; current working-tree test results are separate. Do not claim a study was your own
independent human evaluation unless you performed and documented it.

## Data and modelling

### Q4. Observation and label windows; why 90 days?

Features use events strictly before a cutoff; the target is a positive purchase during the following
90 days. Six dated snapshots serve five stages: train June/September 2010; validation December;
calibration March 2011; decision June; test September. Earlier labels mature before the next stage.
The brief specifies 90 days. Observed repeat gaps have median 32 days; 80.6% are within 90 days.
Customers recur: this is forecasting known customers, not evaluating unseen identities. Longer-gap
customers can return after the target window. Show `src/features.py:create_labels` and
`configs/pipeline.yaml:labelling`; counts are in `retention_metrics.json:splits`.

### Q5. Why PR-AUC rather than accuracy or ROC-AUC?

The brief requires PR-AUC as primary. The implementation uses average precision (AP), which is not
trapezoidal PR area. Precision asks how many selected customers return; recall asks how many actual
returners were selected. AP summarizes that tradeoff across thresholds. Always predicting non-return
gives 57.3% test accuracy and no positive predictions. The class imbalance is mild, so do not dismiss
ROC-AUC: report it too. AP is 0.760 versus recency 0.671; ROC-AUC is 0.786 versus 0.761. The saved
stratified dummy AP is 0.431; test prevalence is 0.4273. A finite random realization need not equal
prevalence exactly. Neither ranking measure evaluates coupon uplift. Show Model card plots and
`src/train.py:evaluate`. Headline AP 95% interval is 0.7425–0.7749; paired AP-difference interval is +0.0708–+0.1049 (1,000 customer bootstraps, conditional on this model and historical period).

### Q6. Why threshold 0.205 rather than 0.5?

The June decision data selected 0.205 from 201 candidate thresholds, maximizing the coded utility.
With coupon cost C=5, return margin M=25 and missed-return cost F=0, expected offer utility is
25p−5 and the theoretical break-even is 20%. The saved realized validation utility peaks at 20.5%
(20,005), versus 19,915 at 20%. The numbers are placeholders, not measured campaign economics.

The objective itself has a limitation: it assigns all selected returners' margin to the coupon and
assigns no value to their purchases without an offer. Consequently 20.5% is not a validated incremental
profit optimum. High purchase propensity does not identify customers persuaded by a discount.
Use an offer/control experiment and margin data before claiming coupon effectiveness. Do not simply
reverse the threshold or give larger discounts to low-propensity customers without evidence.
Show Prediction → Coupon math and `src/train.py:utility/select_threshold`.

### Q7. What does calibration buy you if you only rank customers?

A strictly increasing remapping preserves ranking; isotonic calibration can create ties and can
therefore change ranking metrics. Calibration's main purpose here is probability interpretation for
the reviewer and the assumed utility calculation. Among many predictions near 80%, approximately
80% should be positive in a well-calibrated population; calibration does not guarantee this exactly.
March fits the mappings; June selects between raw/sigmoid/isotonic. June Brier: 0.1651 raw, 0.1563
sigmoid (selected), 0.1566 isotonic. Test Brier is 0.1940 on a different period. Brier is mean squared
probability error, not an error percentage or a calibration-only measure. Show the reliability PNG
and `src/train.py`'s calibrator selection. The two calibrated methods nearly tie.

### Q8. Which feature mattered most?

The current forest ranks `std_days_between_orders` first (0.2239), then frequency (0.1858)
and avg_days_between_orders (0.1219). This is global impurity importance, not a causal or
customer-specific explanation. The original audit led to 13 inputs being removed. In the current
12-feature model, individually redundant signals remain jointly useful: removing nine together
loses June AP by 0.0334 (95% interval 0.0206–0.0466). Show `defence_facts.json` and
`governance/FEATURE_INVENTORY.md`. Do not confuse importance with necessity.

### Q9. What if the model loses to the simple rule on recall?

Compare at equal selection counts or budgets, not at the same numeric score threshold. Selecting
61.80% of test customers gives model recall 0.8242 versus recency 0.8193; precision 0.5699 versus
0.5665. This is a small observed difference; marginal metric intervals do not establish a paired recall-difference test. Do not assert its significance. The stronger recorded gains are overall AP and top-decile recall (0.2156 versus
0.1775). Investigate segment errors, measure uncertainty and accept a simpler rule if complexity
is not justified. Source: `defence_facts.json:q9_model_versus_recency_rule`.

### Q10. How was the test protected, and how often was it inspected?

Model selection, tuning, calibration and threshold choice use earlier dates, before final test
scoring. Test results have subsequently been inspected and reproduced for reports/audits; do not say
nobody looked more than once. Repeated identical runs are not independent validation. The saved
record does not establish an untouched second future holdout. Any further model change must be
chosen using development evidence and transparently report renewed use of this test period.
Show the temporal protocol in `src/train.py` and the validation report.

## Recommendation

### Q11. Why association rules rather than collaborative filtering?

The brief requires Apriori/FP-Growth. Rules also fit an explainable current-basket task without an
identified user's history. User-based collaborative filtering needs user interaction histories;
item-based collaborative approaches can also use a current basket, so they are a legitimate
alternative, not impossible here. Compare them if broader catalogue coverage or personalization
becomes the priority. Current evidence is 48.67% Hit-rate@5 versus 17.23% popularity under
half-basket masking. It measures basket completion, not sales or next-order forecasting.

Catalogue coverage is 360/4,687 = 7.68%; rule-firing coverage is 83.67%. Appendix B says above 40%
coverage without naming its definition. Report both measures and clarify the intended rubric
interpretation rather than asserting that 7.68% unambiguously fails that target.

### Q12. Explain high confidence with low lift using your data.

Confidence is P(B|A); lift divides it by P(B). Popular B can produce substantial confidence without
useful association. In the measured support/confidence grid, removing the lift filter gives 3,579
rules; the lift>=1.2 filter removes 12. The largest confidence among discarded rules is 0.1725:
“jumbo shopper vintage red paisley → white hanging heart t-light holder”, lift about 1.13.
B already occurs in 15.3% of training baskets. This is the closest measured example, not a genuinely
high-confidence (e.g. 30%+) trap. Do not fabricate the stronger example requested by the brief or
claim that none can exist under any mining settings. Source: `defence_facts.json:q12_lift_filter`.

### Q13. Why not mine on every request?

Mining traverses the historical basket matrix and discovers frequent combinations. Serving only
matches exported rules through an item-to-rules index, checks antecedents, excludes existing basket
items, deduplicates and ranks. Repeating mining per request wastes time and mixes offline/online
responsibilities. Actual bounded mining times: Apriori 1.22 s; FP-Growth 64.45 s, same rule keys.
CSR encoded storage is 2.71 MB versus 119.99 MB dense equivalent; this is not an algorithm-specific
memory comparison or proof that dense data could not fit. See the mining artifact header.

### Q14. What about 500,000 rules?

The real export has 3,567 rules. A local engine benchmark over 300 historical baskets measured
median/p95 0.025/0.222 ms. Synthetic 500,000-rule data measured 3.144/14.808 ms, max 31.115 ms.
These are local engine calls, not HTTP latency, load testing or measured recommendation quality at
500,000 real rules. Popular antecedents can create long candidate lists. Pruning/capping rules per
item and profiling the real workload are next investigations. Reported process peak-memory growth
is not a precise allocation measurement for the index alone. Show Measured conclusions and
`reports/current_run/rules_scaling.json`.

### Q15. Empty or unmatched basket?

Empty baskets use popularity; unmatched non-empty baskets use popularity with existing items
excluded. The API returns `empty_basket`, `popularity_fallback` or `rule_based`; Streamlit identifies
fallback explicitly and explains firing rules. Popularity counts distinct training invoices, not
quantities. Demonstrate with the picker cleared and `never-seen-item` entered manually. Rule-based
outputs served 83.67% of eligible test baskets; fallback served 16.33%. The two rates differ from
catalogue diversity. Tests cover both fallbacks and basket-item exclusion.

## Generative AI

### Q16. Why RAG rather than fine-tuning?

The task is answering from changing policies with inspectable evidence. Retrieval selects source
passages at request time; updating the corpus/index does not require fitting LLM weights. Fine-tuning
may help style or task behavior, but alone does not supply explicit retrieved provenance; it can also
be combined with retrieval. RAG itself does not guarantee correctness. In this project, section-based
retrieval improved answering-passage availability from 4/14 to 12/14 development questions.
Show `src/rag/index.py` and the source expander.

### Q17. How do you prevent invented product IDs or probabilities?

Typed tools obtain scores through Head A, recommendations through Head B and products through the
exported catalogue. Those outputs are formatted from validated tool results, not generated by the
LLM. Providers route the question; OpenAI/Ollama can separately write policy prose from excerpts.
The figure check flags some unsupported numeric content but cannot prove correctness. Matching a
supplied customer context is not authentication. Show Typed tool execution and `src/rag/tools.py`.

### Q18. How is assistant quality measured?

The 40-question development set has 15 policy, 10 tool-call, 8 out-of-scope and 7 French questions.
Routing accuracy: regex 97.5%, OpenAI 95%, Ollama 92.5%; the eight out-of-scope cases were refused
in all three saved runs. These are not guarantees on arbitrary questions. Source correctness,
retrieval availability and written-answer correctness are separate measures.

The saved single-rater assessment marked 11/18 OpenAI and 3/18 Ollama policy answers fully correct.
Do not present it as independent human validation. Routing median latency is about 27 ms, 1.0 s and
1.4 s respectively; full generation is a separate call. OpenAI's saved token-cost estimates depend
on the price assumptions at the evaluation date, not a verified current price. The development set
influenced changes, so independent fresh questions and human review remain necessary.
Show `assistant_*_metrics.json`, `answer_correctness.json` and `retrieval_metrics.json`.

### Q19. What happens when retrieval is irrelevant?

Below similarity 0.25, policy answering is refused. Between 0.25 and 0.45 a caution is added; 0.45+
allows the normal response path. Similarity is not the probability the answer is correct. The values
were judged using small development question sets; irrelevant passages can still score highly.
Out-of-scope matching is another heuristic, not a complete security boundary.

Check the actual response: the low-similarity refusal currently lacks the explicit support contact
required by the brief; the intermediate-confidence caution contains a contact. The generic helper's
handoff text does not prove the live route uses it. This is an implementation gap to fix/test, not
something to claim complete. Show `api/routers/chat.py:_orchestrate` and passage-confidence tests.

### Q20. Why offer a local provider?

It provides a way to route/generate without sending that call to a hosted LLM and permits a measured
quality/latency comparison. Regex uses text patterns plus local semantic retrieval; Ollama performs
local LLM inference. Local execution alone does not guarantee security or legal compliance. Hosted
calls receive the typed question and, for policy generation, excerpts; a user can type personal data
and no redaction currently removes it. The fictional notice promises Canadian processing, so the
implemented data flows and wording require review before any real use.

## Engineering and MLOps

### Q21. joblib versus pickle risks?

Both can execute code during deserialization. joblib is convenient for Python estimators/arrays,
not a security boundary. The loader verifies checksum and feature contract before loading; record
library versions and trust the artifact source. A checksum next to a model does not authenticate
an attacker who can replace both. Safer serialization requires its own format/runtime review;
do not label another format universally safe. Show `src/registry.py:load_checked`.

### Q22. What is a model version?

The inference artifact includes preprocessing and the calibrated estimator. Its metadata records
feature order/contract, versions, timestamps, row counts, git/data/config/model hashes, metrics,
threshold and MLflow run ID. Registry aliases and approval tags determine what is served. The last
recorded model is a candidate without a champion, so startup reports local fallback. The saved
artifact has a dirty working-tree provenance; the commit alone does not identify every source edit.
Show About → Version, sidecar metadata and promotion policy; do not promote merely for a screenshot.

### Q23. /health versus /ready?

Health means process liveness. Ready means required artifacts are loaded and startup errors are
absent; it returns 503 otherwise. Traffic routing must wait for readiness; a process being alive
is not enough. Docker checks `/ready`. This is application behavior, not evidence that a deployed
load balancer is configured. Show the live endpoints and `api/main.py`.

### Q24. Why OIDC instead of long-lived AWS keys?

The workflow is designed to exchange a scoped identity token for temporary role credentials rather
than store reusable AWS access keys. Trust must restrict the intended repository/branch/role;
temporary credentials can still be misused while valid. The workflow file alone proves only design.
On 2026-11-05 a GitHub origin exists, but GitHub queries returned 401, so current CI/CD runs and
branch protection were not verified. Reauthenticate and inspect execution evidence before making
claims. No current cloud-service availability/pricing claim is made here.

### Q25. Why a retraining PR rather than automatic deployment?

Training can improve one metric while changing fairness, calibration or business effects. The
configured workflow registers a candidate, evaluates gates and proposes review through a PR; it
does not auto-promote. Accountable approval controls promotion. Workflow files do not establish
that a scheduled run has executed. Reusing an old holdout for gates is not fresh validation.
Show `.github/workflows/retrain.yml` and `governance/PROMOTION_POLICY.md`.

### Q26. How would you detect overnight degradation?

Monitor operational errors/latency, input and output drift, data quality and rule fire rate. A
background loop raises/resolves local alerts and optionally calls a webhook. No external paging
is evidenced; the monitor lives in the API process, so external availability monitoring is needed
if it dies. Input drift suggests investigation, not demonstrated accuracy loss. Reference aging,
large-sample KS sensitivity and multiple comparisons can also produce alerts.
Show Monitoring after preloaded ordinary/shifted demo traffic.

### Q27. What do you monitor while labels are 90 days late?

Use the operational and distributional proxies above, while making clear that outcomes are not yet
known. A cohort's genuine performance can only be computed after its 90-day outcome window closes
and outcomes are joined. The backfill design needs a governed join identifier and retention policy;
it is documented, not implemented. Do not show drift as a replacement for measured predictive
accuracy. See `docs/MONITORING.md`.

### Q28. When is managed training worthwhile?

Potential reasons include larger workloads, parallel search, specialized compute and managed
execution. Compare measured training time, overhead and cost for this task rather than assume the
cloud is cheaper/faster. Script-mode training and the endpoint image were tested locally. Actual
SageMaker training/HPO, invocation/teardown and cost/crossover evidence are required by the PDF;
the latest local records do not establish those AWS executions. Do not invent a cost comparison.
Show `docs/SAGEMAKER.md` and the capped-job launcher.

## Governance

### Q29. Which decisions are automated, and what follows?

Use the decision register: retention scores/proposals, recommendation ranking, assistant routing
and refusals, and retraining/promotion controls. A reviewer records approve/reject with a reason for
the coupon proposal; no coupon is dispatched. Do not assert that a reviewer's typed name alone
establishes legal compliance or that other automated outputs are automatically exempt. The brief
requires disclosure, a human-approval design consequence and primary-source privacy review.
Governance documents distinguish implemented controls from plans. Authentication, rights procedures
and enforcement remain incomplete; verify actual obligations with the primary guidance and
qualified review. Show Human review, Recent decisions and the bilingual customer notice.

### Q30. Subgroup recall gap: options and tradeoffs?

Five strong thin-history flags remain; a tiny unknown-region group is flagged with inconclusive evidence. The overall returner
recall is 82.4%. The exact current group recalls and intervals are in fairness_audit.json; earlier 25-feature group numbers are archived. These are
behavioral groups, not evidence about protected demographics. Possible mitigations include better
thin-history features/models, reweighting, justified threshold changes, targeted human review and
an uplift study with new data. Compare recall, precision, selection rate and calibration; one can
improve while another worsens. Investigations belong on development data, not repeated test tuning.
Human review and abstention from automatic dispatch are the operational mitigation; no statistical correction is claimed. Show `governance/fairness_audit.md` and its measured tradeoffs.

### Q31. Customer data sent to a hosted provider outside Canada?

Explain actual outbound content, distinguish fictional policy promises from enforced behavior,
and do not send identifiable data from this demo. Obtain the appropriate privacy/security review
of transfers, contracts, retention, notices and legal obligations before real use. Local options
reduce that particular outbound flow but do not solve every privacy issue. No redaction/PIA or
provider-contract review is evidenced. This answer is a project-status explanation, not legal advice.

### Q32. Explain why a customer did not receive an offer.

In this prototype nobody receives an offer: the system records a proposal and review decision.
Show probability, threshold, reviewer reason and the per-customer feature effects against training
medians. These effects describe associations, not causes or coupon response; they do not add up
because features are correlated. Explanations are computed on request and not stored with the
decision, so exact historical reproduction depends on preserving model and input state. A real
rights workflow, verified identity and response process remain unfinished; the fictional notice's
30-day response promise is not an implemented service guarantee.

## When the panel asks something unfamiliar

Explain the relevant input, computation, output and failure case. Open the function. If you do not
know, say what evidence or experiment would decide it. A measured modest result is defensible;
a guessed metric, invented cloud execution or fabricated personal experience is not.
