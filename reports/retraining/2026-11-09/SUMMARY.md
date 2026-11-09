## Weekly retraining candidate, 2026-11-09

Registered as `boreal-retention` version 3 in `https://boreal-mlflow-qaonyx3dda-nn.a.run.app`: a candidate only. Merging this pull request changes nothing that is served, and nothing is deployed.

| Metric (test snapshot 2011-09-01, 5,233 rows) | Candidate | Champion |
|---|---:|---:|
| pr_auc | 0.7611 | none yet |
| brier | 0.1934 | none yet |

| Gate | Passed | Detail |
|---|---|---|
| beats_recency_baseline | yes | PR-AUC 0.7611 vs recency heuristic 0.6711 (first champion) |
| no_new_fairness_flags | yes | new strong flags: none |
| no_widened_fairness_gaps | yes | flags widened by more than 2.0 pp: none |

Fairness flags an approver must acknowledge: purchase_history=one-time, spend_quartile=Q1 (lowest), spend_quartile=Q2, tenure_band=6-12 months, tenure_band=under 6 months.

To serve this candidate after review, a named approver runs:

```bash
python -m src.registry promote --version 3 --approver "Name" --reason "why" --acknowledge-fairness-flags
```

The UCI workbook is a fixed historical dataset, so a weekly run reproduces the same candidate unless code or configuration changed. Policy: governance/PROMOTION_POLICY.md.
