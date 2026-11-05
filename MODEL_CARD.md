# Model card — retention 0.3.0

Updated 2026-11-05. Authoritative metadata: `artifacts/retention_model_metadata.json`.
The archived 25-feature version is in `reports/archive/20261105_25_feature_model/`.

## Intended use and model

RandomForestClassifier with training-only median imputation and sigmoid calibration. Positive
means a positive purchase in the following 90 days. This historical academic demo estimates
purchase probability. It does not estimate incremental coupon response and sends no coupons.

Contract 2.0 serves **12 ordered features** from `src/contracts.py`; 25 remain engineered and
documented for offline analysis. The 13 original removals were chosen from the June development
audit before the new test evaluation. Location fields are excluded from model inputs and retained
only for offline group auditing. Training fits 10,253 snapshot rows, calibration 4,522 rows and
final test evaluates 5,233 customers. Repeat customers can occur across snapshots.

## Chronological protocol

Train: 2010-06-01 and 2010-09-01; validation: 2010-12-01; calibrate: 2011-03-01;
decision: 2011-06-01; test: 2011-09-01. Six cutoffs, five stages. Features strictly precede each
cutoff and 90-day labels mature before the next stage. Eight classifier families, dummy and
recency baselines are compared on validation. The top two tunable families receive 12 Optuna
trials each. MedianPruner evaluates genuinely fitted 50%, 75% and 100% training resources;
9 of 24 trials were pruned. March fits calibrators; June chooses calibration and threshold.

| September test | Average precision | ROC-AUC | Brier |
|---|---:|---:|---:|
| Calibrated forest | 0.759631 | 0.786325 | 0.193983 |
| Recency ranking | 0.671134 | 0.760756 | 0.396624 |
| Stratified dummy | 0.431339 | 0.508110 | 0.485572 |

AP is sklearn average precision, historically labelled PR-AUC here, rather than trapezoidal PR
integration. Test prevalence is 42.73%. Recency scores are 1/(1+recency_days), an uncalibrated
ranking; their Brier score is not a fair comparison of learned calibration.
AP 95% interval: 0.7425–0.7749; paired AP difference versus recency: +0.0708–+0.1049.
These are 1,000 paired customer bootstraps conditional on this model and this one historical
period, not uncertainty over model selection, retraining or future domain shifts.

## Calibration and decision policy

June Brier: raw 0.16506, isotonic 0.15665, sigmoid 0.15631 (selected). September Brier 0.19398
is on a later population. Plots and confusion counts appear under Streamlit About → Model card.

Offer utility: pM−C; no-offer utility: −pF. Break-even C/(M+F) is 20% with assumed C=5, M=25,
F=0. The 201-point June threshold sweep selected **20.5%**, utility 20,005 assumed units.
At this threshold, test recall is 82.42%, precision 56.99%, selecting 61.80% of customers.
An equal-count recency rule recalls 81.93%. The practical binary advantage is small.
Assumed test utility 29,905 is neither observed profit nor causal uplift; purchases without an
offer are not credited in this objective. Equal coupons are issued as a hypothetical policy only.

## Governance and registry

MLflow registered version 2 as a candidate, without champion promotion. The API's artifact fallback
is reported by `/version`. Owner promotion requires explicit acknowledgement of fairness flags.
Five strong thin-history recall disparities remain; a tiny unknown-region group is inconclusive.
See `governance/fairness_audit.md` for intervals, overlap and mitigation trade-offs. The selected
operational mitigation is human review of every proposed action and abstention from automatic
dispatch; it does not repair the statistical recall gap.

The current 12-feature ablation finds nine individually inconclusive features. Dropping those nine
jointly loses June AP by 0.0334 (95% interval 0.0206–0.0466), so they are retained together.
The brief's minimisation requirement is addressed by removing the original 13, not by treating
individually redundant signals as jointly dispensable.

## Limits and reproducibility

The September test snapshot was previously examined with the original model and re-evaluated
under the locked retraining protocol. It is **not a new independent holdout**. The first retraining
attempt failed during audit generation; the fixed audit and successful rerun are both logged.
No selection between model versions used the new September result.

Historical UK giftware, weak first-token category proxies, repeated customers and seasonal changes
limit transfer to fictional Québec grocery operations. Thin-history recall remains weak; removing
location inputs does not prove fairness. No treatment/control coupon experiment is available.
Authentication, real rights handling, verified reviewer identity and enforced retention are absent.
Cloud execution and current GitHub CI require credentials; local runs do not substitute for them.

`top_factors` are global importances. Per-customer `explanation` uses one-at-a-time median
substitution, not SHAP or causality; correlated contributions do not add up and explanations are
not stored with decisions. Trusted joblib artifacts only: hashes establish integrity, not trust.
See `docs/AI_USE.md` for assistance and verification provenance.
