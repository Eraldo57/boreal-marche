# Model card — retention milestone

Updated 2026-11-01. This describes the generated artifact, not an assumed winning algorithm. The model itself was trained on 2026-09-30 and has not changed since.

## Artifact and intended use

RandomForestClassifier, training-only median-imputation pipeline, isotonic calibration selected on a separate decision period. Model version 0.2.0. Exact timestamp, run ID, library versions, dirty git state, feature/data/config/code/model hashes and threshold are in artifacts/retention_model_metadata.json.

Positive target: **will place a positive purchase within 90 days**. Risk labels describe NON-return and therefore decrease as the probability increases. This is a historical academic demonstration, not an authorized campaign or deployment on contemporary customers. No coupon dispatch is implemented.

## Inputs and evaluation

Exactly 25 ordered features in src/contracts.py and governance/FEATURE_INVENTORY.md. Training snapshots precede model-selection validation, calibration, decision and final test periods. Details and every comparison are in reports/current_run/retention_metrics.json. The final estimator fits 10,253 snapshot rows, not 10,253 distinct people. Calibration uses 4,522 rows; final holdout uses 5,233.

| Final chronological holdout | Average precision | ROC-AUC | Brier |
|---|---:|---:|---:|
| Selected calibrated model | 0.743038 | 0.780963 | 0.195399 |
| Recency-ranking heuristic | 0.671134 | 0.760756 | 0.396624 |
| Stratified dummy | 0.431339 | 0.508110 | 0.485572 |

AP is reported under the historical name PR-AUC, but implemented with sklearn average_precision_score (not trapezoidal integration). Recency is an uncalibrated ranking baseline, so its Brier score should not be interpreted as a fair comparison of learned calibration. Test prevalence is 0.427288. Recall of returners in the highest-scored decile is 0.215564.

Eight classifier families and two baselines were compared. Twelve Optuna trials for each of the top two tunable families were executed; no pruning is claimed. Sigmoid, isotonic and uncalibrated Brier were compared on June decision data. No model/calibration/threshold choice uses final test data.

Two charts are drawn from the saved report (`python -m src.plots`); every training run also logs them to MLflow:

- **Reliability before and after calibration:** `reports/current_run/figures/reliability_curve.png`. On June decision data the Brier score goes from 0.1681 uncalibrated to 0.1578 with isotonic and 0.1582 with sigmoid; isotonic was selected.
- **Utility across thresholds:** `reports/current_run/figures/threshold_utility.png`.

## Registry and promotion

- **Version 1.** This model is version 1 of `boreal-retention` in the local MLflow registry, registered as a candidate.
- **Gate check.** It passes the first-champion gate on the test snapshot: PR-AUC 0.7430 against 0.6711 for the recency heuristic.
- **Fairness flags.** It carries five strong flags that an approver must acknowledge.
- **Not promoted.** The approver must be the project owner (`governance/PROMOTION_POLICY.md`). Until then the API serves `artifacts/` and `/version` says so.

## Fairness and minimisation

- **Fairness audit** (`governance/fairness_audit.md`, test snapshot, threshold 0.22). No region is flagged. Five groups are strongly flagged, all customers with a thin purchase history, against an overall returner recall of 81.4%:
  - one-time buyers, recall 17.3%;
  - lowest spend quartile, 14.2%;
  - second spend quartile, 62.7%;
  - tenure under 6 months, 69.5%;
  - tenure 6–12 months, 64.9%.

  Five mitigations are listed with their trade-offs; none is chosen.
- **Data minimisation** (`governance/FEATURE_INVENTORY.md`). A drop-column test on June decision data:
  - Twelve features are necessary.
  - Thirteen show no reliable contribution, including all three location features and two exact duplicates.
  - Removing all thirteen together scored higher, not lower.

  The recommendation to retrain on the twelve awaits the project owner's decision; it would change the API contract and every metric.

## Assumption-based threshold

Threshold 0.22 maximizes June validation utility:
selected × (actual_return × margin − coupon_cost), with an optional missed-return cost.
Coupon cost 5, margin 25 and missed-return cost 0 are **assumed monetary units**, not learned economics. This targets likely returners, not proven persuadable churners. The numerical utility is not observed profit, causal uplift or ROI; it should not be presented as such.

## Important limitations

- Customers recur across snapshots; observation history expands and seasons differ. Unseen-customer performance is not measured.
- Customers with a thin purchase history have much lower returner recall; see governance/fairness_audit.md.
- Country coding is ordinal and category labels are weak first-token proxies. The redundant features found by the minimisation audit are still in the model, pending a decision.
- Geographic/temporal/source-domain transfer is unvalidated. Live monitoring shows real 2011 traffic drifting from the 2010 training reference, especially the tenure features.
- Metric confidence intervals are not reported (subgroup recall intervals are). There is no coupon experiment. Drift-to-performance backfill is designed but not built. SageMaker tuning is built but has not run on AWS.
- Repeated runs reproduced the same locked protocol. They are not additional independent holdouts.
- API top_factors are optional global importance hints, not individual SHAP explanations; calibrated wrappers may return an empty list.

Joblib has pickle-equivalent arbitrary-code execution risks. Only load trusted artifacts; hashes detect change, not malicious intent.

- **Ran:** MLflow tracking and registry registration.
- **Not yet run:** registry promotion, which waits for the owner. SageMaker spot training and the endpoint ran only locally in their container (docs/SAGEMAKER.md), not on AWS.
