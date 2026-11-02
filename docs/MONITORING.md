# Monitoring, alerts and delayed ground truth

Updated 2026-11-01. Covers brief section 3.6, "Monitoring". Code: `api/monitoring.py`. Thresholds: `configs/pipeline.yaml` under `monitoring`. Dashboard: the Streamlit **Monitoring** tab, fed by `GET /metrics` and `GET /alerts`.

## What is measured

Every business request (`/predict`, `/predict/batch`, `/recommend`, `/chat`) is one row in the `prediction_log` table. The metrics use the newest 50,000 rows and the training reference saved with the model (`artifacts/drift_reference.parquet`). No customer ID or chat text is stored.

| Check | How it is measured | Rule | Why this rule |
|---|---|---|---|
| Error rate | Share of logged requests with status 400 or higher, since the log began | Yellow above 0.05, red above 0.10 | Judgement values for a demo service |
| Latency p95 | 95th percentile of request latency in the window | Yellow above 500 ms | The brief's Appendix B calls anything above 500 ms "weak" |
| Input drift | PSI over the reference's deciles, and a two-sided KS test, for each of the 25 features | Red when any PSI > 0.2 or any KS p-value < 0.05 | PSI 0.2 is a common rule of thumb; 0.05 is the conventional p-value cut-off |
| Prediction drift | The same two tests on the served probabilities, against the model's scores on its own training reference | Same as input drift | A shift in what the model outputs can matter even when no single input moved |
| Invalid request rate (data quality) | Share of prediction requests rejected by input validation (HTTP 422) | Yellow above 0.05, once 20 prediction requests exist | Judgement value |
| Out-of-range share (data quality) | For each feature, the share of logged values outside the training reference's minimum–maximum | Yellow when any feature exceeds 0.01, once 100 predictions exist | Judgement value; it catches values the model has never seen |
| Rule fire rate | Share of non-empty baskets answered by association rules rather than the popularity fallback | Yellow below 0.5, once 20 such baskets exist | The offline rule-firing coverage was 0.84 |
| Request volume | Requests per route in the window | Shown, no alert | Context for every other number |

- **Status.** The overall status is the worst level among the checks.
- **"not_evaluated" is not "green".** A check with too little data is reported as `not_evaluated`. Drift and the minimum-volume rules appear only once their minimum data exists.

## Automated alerts

- **When.** A background task in the API runs every `ALERT_INTERVAL_SECONDS` (60 by default). Each run re-evaluates every check, whether or not anyone has the dashboard open.
- **What triggers an alert.**
  - A check that turns yellow or red raises an alert.
  - If the check stays breached, it does not alert again, unless its level changes (for example yellow to red).
  - When the check recovers, a "resolved" event is sent.
- **Where alerts go.**
  - The structured log, always.
  - The `alert_log` table, in the same database as the prediction log.
  - A webhook, when `ALERT_WEBHOOK_URL` is set. The payload is Slack-compatible (`text`), plus the structured fields.
  - `GET /alerts` lists the active checks and the recent events.
- **Failures are counted, not fatal.** A delivery failure (database or webhook) is counted in `delivery_failures` and logged. It never stops the API, and webhook delivery is not retried.
- **After a restart.** The active set starts empty, so a check that is still breached alerts once more.

## Delayed ground truth: the backfill design

The label is "made a purchase within 90 days of the prediction". So nobody can know whether a prediction was right until 90 days after it was made. Until then, the monitors above are proxies: drift and data quality can say the inputs changed, but not that accuracy fell.

This design is **documented, not implemented**:

1. **Log a join key at prediction time.** Store a pseudonymous key with each prediction: a keyed hash (HMAC) of the customer ID, whose key is a deployment secret. Store it with the timestamp, the probability, the decision at the threshold and the model version.
   - Today the prediction log deliberately stores no identifier. Adding this key is a privacy decision: it needs the Law 25 review in `GOVERNANCE_LOI25.md` and a retention limit first.
2. **Fill in outcomes every day.** Each day, a job takes the predictions whose 90-day window has just closed. It looks them up in the purchase ledger using the same key. Label 1 means at least one positive invoice in [prediction time, prediction time + 90 days); label 0 otherwise. The results go into a `prediction_outcome` table.
3. **Score matured cohorts.** Group predictions into weekly cohorts by prediction date. For each matured cohort, compute PR-AUC, recall at the threshold, Brier score and a reliability table. Compare them with the model card's holdout values, and alert when PR-AUC falls below the recency baseline on the same cohort.
4. **When the first true accuracy signal arrives.**
   - First labels: on day 91 after the first logged prediction (the 90-day window plus the daily job).
   - First complete weekly cohort: day 97 (7 days of predictions plus 90 days).
   - Until then, only proxies are available.

For the historical demo, the labels already exist. The test snapshot (cut-off 2011-09-01) has its 90-day window closed in the workbook, which is why it can serve as the holdout.

## Filling the dashboard before a defence

With the API running:

```bash
.venv/bin/python -m scripts.simulate_traffic --count 120              # real traffic
.venv/bin/python -m scripts.simulate_traffic --count 120 --shifted    # plus deliberate shifts
```

- **Real traffic.**
  - 120 historical feature rows sent to `/predict/batch`.
  - 40 real September–November 2011 baskets from the cleaned workbook. Each sends the first half of its items, as in the offline evaluation.
- **`--shifted`: deliberate, synthetic changes.**
  - Recency is moved 365 days later, which shows input and prediction drift.
  - Half the baskets contain items that do not exist, so the rule fire rate falls.
  - Ten requests carry an impossible recency, so they are rejected and show up as a data-quality problem.
- **Waiting for alerts.** The script waits for the background monitor's next evaluation (up to 75 seconds). It then saves the before and after metrics and the alerts to `reports/current_run/monitoring_demo.json`.
- **A quicker demo.** Start the API with `ALERT_INTERVAL_SECONDS=5`.

## Limitations

- **One process only.** The window and the active alerts live in one API process. Several workers would each see part of the traffic.
- **No multiple-comparison correction.** 25 features × 2 tests at p < 0.05 will flag something by chance. With large samples, KS also flags differences too small to matter.
- **Drift reference.** The reference comes from `artifacts/`. When the API serves a registry champion trained on other data, the reference should come from that version too; that is not implemented yet.
- **No escalation.** There is no paging, escalation or on-call rota, and the webhook is not retried.
