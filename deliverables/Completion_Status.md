# Capstone completion — current status

Updated 2026-11-05 against the original 31-page Capstone_Project1.pdf. This records evidence,
not a grade or certification. Source brief is in sibling Version_1; no historical tags are fabricated.

| Requirement | Current local result | Remaining evidence |
|---|---|---|
| Data | UCI workbook checksum, DVC pointer/cache and updated lockfile, cleaning manifest, Parquet, policy corpus | Remote DVC storage; independent fresh-machine reproduction |
| Head A | 25 engineered / 12 served features; 8 families, 2 baselines; 24 real-resource Optuna trials, 9 pruned; calibration, utility, error analysis; 1,000-draw paired CIs | New future-period holdout and coupon/control data; real economic inputs |
| Head B | Timed Apriori/FP-Growth, sweep, sparse matrix, indexed 3,567 rules, temporal Hit@5 and two coverage measures, fallback | SKU/category improvements; online impact test. No fabricated high-confidence/low-lift example |
| Head C | Four typed tools, three providers, multilingual retrieval, streaming/citations, low-floor bilingual fictional-support handoff, 40 development questions | Independent answer evaluation; golden-set chronology proof; stronger French correctness |
| Governance | Original 13 unnecessary model features removed; audit fields offline only; refreshed fairness audit, notice, human-review workflow | Verified identities, operational rights/retention, legal review; statistical fairness gaps remain |
| API/UI | Versioned 12-feature contract, diagnostics, utility explanations, rule explorer, logging and monitoring | Native + linux/amd64 images built; amd64 inference verified. Hosted access and independent reproduction pending |
| Cloud / registry | SageMaker scripts, local MLflow version 2 candidate, explicit promotion gates | Actual AWS training/HPO/cost/crossover and endpoint invoke/teardown; shared registry; owner promotion |
| CI/CD | Workflow source and local checks | GitHub auth, green run, branch protection, OIDC deployment, retraining execution |
| D6 package | Technical report, bilingual two-page summary, editable deck with notes, deck PDF | Owner content review, video/live links |
| D7 | Defence cue card and reflection draft | Hosted system, public repo/access, reviewed v1.0 release, actual defence, owner-verified personal reflection |

## Decisions made locally

The feature-removal decision used June development evidence before new September results.
Original outputs are preserved in `reports/archive/20261105_25_feature_model/`. The final
September snapshot was reused; reported intervals condition on this one period and selected model.
The first retraining failed during audit generation, was corrected and rerun; both logs remain.

Coupon amounts remain explicit scenarios (C=5, M=25, F=0). No campaign is sent. Human review
and abstention from dispatch are the operational mitigation; no statistical fairness repair is claimed.
Five strong thin-history recall flags remain. A tiny unknown-region group is inconclusive.

## External blockers and owner tasks

1. Authenticate GitHub (`gh auth login`); verify public access and workflow results before releasing.
2. Supply an AWS account/profile, region ca-central-1 and maximum remaining spend. No paid jobs
   or public deployment can be executed with absent credentials and an unspecified spend budget.
3. Complete the documented cloud runs, endpoint cleanup, hosted demo, OIDC and cost evidence.
4. Review personal reflection and AI attribution; record the required video and rehearse the live demo.
5. Create reviewed release tags after green CI. Old milestone tags cannot truthfully be backdated.

Authentication checks found invalid GitHub credentials and no AWS credentials. This is why the
project cannot honestly be called fully deployed or submission-complete yet. A local Docker run,
PowerPoint or report does not replace these external requirements.

DVC captured the existing manually produced outputs with `dvc commit --force`, after separating small Git-versioned metadata as noncached outputs. This updates the local cache/lockfile; it is not a claim that a new machine reproduced every stage. Raw data is now an explicit ingestion dependency.
