# Model card — retention milestone

Updated 2026-09-30. This describes the generated artifact, not an assumed winning algorithm.

## Artifact and intended use

RandomForestClassifier, training-only median-imputation pipeline, isotonic calibration selected on a separate decision period. Model version 0.2.0. Exact timestamp, run ID, library versions, dirty git state, feature/data/config/code/model hashes and threshold are in artifacts/retention_model_metadata.json.

Positive target: **will place a positive purchase within 90 days**. Risk labels describe NON-return and therefore decrease as the probability increases. This is a historical academic demonstration, not an authorized campaign or deployment on contemporary customers. No coupon dispatch is implemented.

## Inputs and evaluation

Exactly 25 ordered features in src/contracts.py and FEATURE_INVENTORY.md. Training snapshots precede model-selection validation, calibration, decision and final test periods. Details and every comparison are in reports/current_run/retention_metrics.json. The final estimator fits 10,253 snapshot rows, not 10,253 distinct people. Calibration uses 4,522 rows; final holdout uses 5,233.

| Final chronological holdout | Average precision | ROC-AUC | Brier |
|---|---:|---:|---:|
| Selected calibrated model | 0.743038 | 0.780963 | 0.195399 |
| Recency-ranking heuristic | 0.671134 | 0.760756 | 0.396624 |
| Stratified dummy | 0.431339 | 0.508110 | 0.485572 |

AP is reported under the historical name PR-AUC, but implemented with sklearn average_precision_score (not trapezoidal integration). Recency is an uncalibrated ranking baseline, so its Brier score should not be interpreted as a fair comparison of learned calibration. Test prevalence is 0.427288. Recall of returners in the highest-scored decile is 0.215564.

Eight classifier families and two baselines were compared. Twelve Optuna trials for each of the top two tunable families were executed; no pruning is claimed. Sigmoid, isotonic and uncalibrated Brier were compared on June decision data. No model/calibration/threshold choice uses final test data.

## Assumption-based threshold

Threshold 0.22 maximizes June validation utility:
selected × (actual_return × margin − coupon_cost), with an optional missed-return cost.
Coupon cost 5, margin 25 and missed-return cost 0 are **assumed monetary units**, not learned economics. This targets likely returners, not proven persuadable churners. The numerical utility is not observed profit, causal uplift or ROI; it should not be presented as such.

## Important limitations

- Customers recur across snapshots; observation history expands and seasons differ. Unseen-customer performance is not measured.
- One-time buyers have much lower observed returner recall; see fairness_audit.md.
- Country coding is ordinal and category labels are weak first-token proxies; redundant monetary features remain. Necessity/ablation studies are unfinished.
- Geographic/temporal/source-domain transfer is unvalidated.
- No confidence intervals, full subgroup audit, coupon experiment, drift-to-performance backfill or cloud HPO exists.
- Repeated runs reproduced the same locked protocol. They are not additional independent holdouts.
- API top_factors are optional global importance hints, not individual SHAP explanations; calibrated wrappers may return an empty list.

Joblib has pickle-equivalent arbitrary-code execution risks. Only load trusted artifacts; hashes detect change, not malicious intent. MLflow tracking ran; model registry promotion, spot training and cloud endpoints did not.
