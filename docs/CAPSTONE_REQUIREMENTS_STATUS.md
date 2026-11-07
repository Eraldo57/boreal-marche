# Capstone completion — current status

Updated 2026-11-06 against the original 31-page Capstone_Project1.pdf. This records evidence,
not a grade or certification. Source brief is in sibling Version_1; no historical tags are fabricated.

| Requirement | Current local result | Remaining evidence |
|---|---|---|
| Data | UCI workbook checksum, DVC pointer/cache and updated lockfile, cleaning manifest, Parquet, policy corpus | Remote DVC storage; independent fresh-machine reproduction |
| Head A | 25 engineered / 12 served features; 8 families, 2 baselines; 24 real-resource Optuna trials, 9 pruned; calibration, utility, error analysis; 1,000-draw paired CIs | New future-period holdout and coupon/control data; real economic inputs |
| Head B | Timed Apriori/FP-Growth, sweep, sparse matrix, indexed 3,567 rules, temporal Hit@5 and two coverage measures, fallback | SKU/category improvements; online impact test. No fabricated high-confidence/low-lift example |
| Head C | Four typed tools, three providers, multilingual retrieval, streaming/citations, low-floor bilingual fictional-support handoff, 40 development questions | Independent answer evaluation; golden-set chronology proof; stronger French correctness |
| Governance | Original 13 unnecessary model features removed; audit fields offline only; refreshed fairness audit, notice, human-review workflow | Verified identities, operational rights/retention, legal review; statistical fairness gaps remain |
| API/UI | Five-tab Streamlit and FastAPI deployed publicly on Google Cloud; live API controls and shifted-cohort alerts verified | Independent fresh-machine reproduction; browser visual check |
| Cloud / registry | Dedicated Google Cloud project, private shared MLflow, Cloud SQL, Artifact Registry, Storage and budget alert; original AWS source retained | See cloud execution report for managed training/HPO/endpoint results, cost estimates and remaining owner promotion |
| CI/CD | Public repository, protected main, ten green required checks, keyless merge-triggered Cloud Run deployment with exact-SHA smoke | See cloud execution report for scheduled-retraining verification |
| D6 package | Technical report, bilingual two-page summary, editable deck with notes, deck PDF and live links | Owner content review; required video |
| D7 | Hosted system, public repository, defence cue card and reflection draft | Final release evidence, actual defence, owner-verified personal reflection |

## Decisions made locally

The feature-removal decision used June development evidence before new September results.
Original outputs are preserved in `reports/archive/20261105_25_feature_model/`. The final
September snapshot was reused; reported intervals condition on this one period and selected model.
The first retraining failed during audit generation, was corrected and rerun; both logs remain.

Coupon amounts remain explicit scenarios (C=5, M=25, F=0). No campaign is sent. Human review
and abstention from dispatch are the operational mitigation; no statistical fairness repair is claimed.
Five strong thin-history recall flags remain. A tiny unknown-region group is inconclusive.

## External blockers and owner tasks

1. Disclose the owner-selected Google Cloud substitution for the assignment's AWS services.
2. Review the current [cloud runbook](GOOGLE_CLOUD.md) and execution report; distinguish actual
   managed-job evidence from configuration, and estimates from billed charges.
3. Review personal reflection and AI attribution; record the required video and rehearse the demo.
4. Review the final release after green CI. Old milestone tags cannot truthfully be backdated.
5. Resolve the named human model-approval decision and remaining fairness/business assumptions.

GitHub authentication and public hosting have been completed. The November 5 reports preserve
the earlier local-only status; the Google Cloud execution report records the subsequent rollout.
Owner review, the required video and an actual defence are not inferred from deployment.

DVC captured the existing manually produced outputs with `dvc commit --force`, after separating small Git-versioned metadata as noncached outputs. This updates the local cache/lockfile; it is not a claim that a new machine reproduced every stage. Raw data is now an explicit ingestion dependency.
