# Fairness and segment diagnostics — partial

Updated 2026-09-30. Positive class is **will_return**, not churn.

The current training run generates limited descriptive segment confusion counts on the final chronological test set. These are not a completed fairness audit. Source: reports/current_run/retention_metrics.json, error_analysis, threshold 0.22.

| Segment | Rows | Returner recall | Precision |
|---|---:|---:|---:|
| Repeat buyers | 3,645 | 0.9202 | 0.5785 |
| One-time buyers | 1,588 | 0.1730 | 0.4297 |
| International primary country | 458 | 0.8571 | 0.5902 |
| UK primary country | 4,775 | 0.8095 | 0.5706 |

These are observed associations, not established causes. The one-time-buyer recall gap deserves investigation; the results do **not** support the former blanket claim that international customers necessarily perform worse.

## Next analysis

- Report uncertainty, prevalence, selection rates and sample sizes by country, tenure and training-defined spend bands.
- Investigate cold-start feature choices and the trade-off imposed by a threshold targeting likely returners.
- Test removal/alternative encoding of geographic proxies and redundant features using development data.
- Compare calibration by segment; reserve a new future period before making deployment claims.
- Consider representative data collection, training weights, feature redesign or carefully reviewed threshold alternatives; none is implemented or selected yet.

No protected demographic attributes are available. Proxy groups cannot establish demographic fairness, and absence of demographic inputs does not rule out discrimination. Do not optimize on this reported holdout and then call it untouched.
