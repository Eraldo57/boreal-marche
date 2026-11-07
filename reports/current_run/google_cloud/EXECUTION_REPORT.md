# Google Cloud execution report

Updated 2026-11-07T01:59:39.124812+00:00. Owner: Eraldo Ibrushi / Eraldo57.

The owner selected Google Cloud in place of the original brief's AWS services. This report
records actual operations; it does not claim that the instructor approved the substitution or
that owner tasks such as the defence, personal reflection and required video are complete.

## Live application

- Streamlit: https://boreal-ui-qaonyx3dda-nn.a.run.app
- API docs: https://boreal-api-qaonyx3dda-nn.a.run.app/docs
- Source: https://github.com/Eraldo57/boreal-marche
- Private MLflow: https://boreal-mlflow-qaonyx3dda-nn.a.run.app (Google IAM required).
- Project: `boreal-marche-eraldo57`; region: `northamerica-northeast1` (Montreal).

The deployed model is the bundled, checksummed version 0.3.0. New managed models are
candidates; no champion promotion or fairness acknowledgement was invented.
The hosted assistant uses regex, with retrieved excerpts and citations. The local application
retains the full provider selector and the recorded three-provider evaluation.

## Deployment and integration evidence

The [merge-triggered deployment](https://github.com/Eraldo57/boreal-marche/actions/runs/42276317729)
succeeded for source `670f0c1ef270ea7765088e929101bcc6c9d3b7ad`. It used Workload Identity Federation rather than
service-account keys. Main is protected by ten required CI checks. CI includes creation of a
real MLflow run in PostgreSQL 16, in addition to image builds and the source/API test suite.

Cloud Run API/interface readiness, exact Git SHA, OpenAPI, real customer 12347 prediction,
recommendations, cited chat, metrics/alerts and Streamlit health passed (`smoke.json`). Both
Cloud SQL databases passed connection and schema-write checks (`database_verification.json`).
Private MLflow run creation, metric logging and run completion passed
(`tracking_write_verification.json`).

Streamlit's official AppTest exercised all rendered tabs and the prediction, recommendation,
assistant and monitoring buttons against the live cloud API without exceptions
(`interface_verification.json`). Browser automation was unavailable; this is functional
interface evidence, not a screenshot or an independent browser visual inspection.

The shifted-cohort demo sent 120 historical feature rows, 20 real historical baskets,
20 deliberately unknown-item baskets and 10 invalid requests. All 10 invalid requests were
rejected. Six active alerts were recorded: error rate, input drift, invalid request rate,
out-of-range share, prediction drift and rule fire rate (`monitoring_traffic.json`). A red
monitoring status after this deliberate perturbation is expected. These are demonstration
observations, not production performance or a causal experiment.

## Managed training and registry

Custom job `7363671538651889664` succeeded on a Spot `n1-standard-4` worker. The worker
ran the same `src.train.run` implementation, chronological matrices and validation/calibration/
decision/test partitions as local training. It performed 12 Optuna trials per selected family.
The candidate was registered as `boreal-retention` version 1 in the private MLflow registry.

Candidate test AP: 0.7610699516816429; ROC AUC:
0.7875943176330773; Brier:
0.19340104467600638. These are candidate measurements on the reused
historical holdout. They do not replace the serving model's published measurements.
See `vertex_full.json`, `candidate_model_metadata.json` and `candidate_retention_metrics.json`.

The first cloud attempt exposed an MLflow/PostgreSQL driver incompatibility. The corrected
server uses psycopg2; CI now reproduces creation of a PostgreSQL tracking run. No failed job
was counted as successful training evidence.

## Bayesian tuning and local comparison

Managed job `2180591317501870080`: **JOB_STATE_SUCCEEDED**.
The request is Bayesian (Vertex default), uses Spot workers, six trials and at most two
parallel workers. It optimises validation average precision, leaving the final test out of HPO.

Fresh local Optuna comparison: six trials, the same RF objective/ranges and matrices,
seed 42. Local results are in `local_optuna_comparison.json`. Run:

```bash
python -m scripts.vertex_comparison local
python -m scripts.vertex_comparison compare
```

Comparison completion: completed; see comparison_optuna_vs_vertex.json.

Six local trials took 1.1725907502695918 seconds and achieved validation AP 0.6759671049458035. Six completed managed trials took 1367.2262 seconds including queue time, with best validation AP 0.6773866544138769. The estimate for that managed pilot is USD 0.09908826887871598; it excludes the interrupted attempts and other project services. Its illustrative crossover is 227.87103333333334 local seconds per trial at the observed cloud overhead.
Larger twelve-trial attempts were interrupted after two and eight successful trials; their raw records are preserved separately and are not labelled completed studies. The six-trial pilot matches the local comparison budget. The local and managed optimisers/hardware differ. The crossover calculation is an illustrative
fixed-overhead model for this small run, not an extrapolated speedup claim.

## Temporary endpoint

Model `5794597802193453056` version 1 was uploaded to Vertex Model Registry.
Endpoint invocation: **recorded in vertex_endpoint_invocation.json**.
Endpoint teardown: **recorded; remaining endpoints: 0**.
The serving adapter downloads the candidate artifacts and invokes the same validated model
contract used in the local SageMaker-compatible adapter. This is actual Google Cloud
execution, not an AWS endpoint claim.

## Weekly retraining

[Workflow verification](https://github.com/Eraldo57/boreal-marche/actions/runs/42278425160):
**success**.
The Monday workflow rebuilds data/features/rules, trains a bounded Vertex candidate,
evaluates the existing promotion gates, and opens a metrics PR only when they pass.
It never promotes a model, merges a PR or deploys a candidate. Main remains protected. The verified workflow opened candidate PR #4:
https://github.com/Eraldo57/boreal-marche/pull/4.
The repository allows Actions to create PRs; default workflow permissions stay read-only,
with explicit writes only in the retraining workflow. The workflow explicitly dispatches CI on its generated candidate branch, because token-created PRs do not automatically run CI without an approval step. No model is approved or merged by this dispatch.

## Budget, costs and teardown

A CAD 10 monthly gross-spend alert was created before billed compute, excluding credits,
with 50%, 80% and 100% notifications (`budget_alert.json`). This is an alert, not a hard cap.
Cloud Run services have zero minimum instances and one maximum instance per revision.
The API uses instance-based CPU so automatic monitoring continues between requests; it
scales to zero when idle. Open Streamlit sessions keep requests active.

Cloud SQL is the small `db-f1-micro` PostgreSQL 16 instance with a 10 GB HDD and daily backup.
It has continuing compute/storage charges. Private artifacts and registry images also incur
storage charges. Full-training workers have 600-second execution limits and tuning workers allow 1800 seconds; launchers cancel full
jobs after 25 minutes and tuning after one hour, including queue time. Spot queue time does
not use Vertex's Flex-start-only `maxWaitDuration` field.

`pricing_catalog.json` captures the official Google Cloud Billing Catalog API's Montreal N1
training CPU/RAM and management fee SKUs. Their on-demand sum for 4 vCPUs/15 GiB is
USD 0.2719184 per worker-hour. The comparison uses this rate and trial elapsed time as a
conservative compute estimate, with explicit exclusions. Actual worker billable time, Spot
discounts and final billed project totals were not exported; no estimate is labelled actual
billed cost, and no claim that the full project met the brief's spending target is made yet.

After assessment, disable future scheduled jobs with
`gh variable set ENABLE_GCP_RETRAINING --body false`. To stop database compute while
preserving data, use `gcloud sql instances patch boreal-db --activation-policy=NEVER
--project=boreal-marche-eraldo57`. Storage charges remain. See `docs/GOOGLE_CLOUD.md`.

## Remaining owner and scientific limits

The required recording, personal reflection approval, actual defence, independent fresh-machine
reproduction and final billed-cost review remain distinct from deployment. Statistical fairness
gaps, reused historical test data, assumed coupon economics and assistant limitations remain.
The system demonstrates historical public data and fictional policies; it does not implement
verified customer accounts, enforceable deletion or a production privacy certification.
