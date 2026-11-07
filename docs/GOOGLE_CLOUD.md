# Google Cloud deployment

This implementation maps the capstone's cloud requirements to Google Cloud at the project
owner's request. The original assignment specifies AWS/SageMaker and App Runner or Render;
this provider substitution should be disclosed to the instructor.

| Capstone component | Google Cloud implementation |
|---|---|
| SageMaker training and Bayesian managed-spot tuning | Vertex AI custom job and hyperparameter tuning job, Spot VMs, bounded execution and launcher deadlines |
| Temporary real-time endpoint | Vertex AI Model Registry and a prediction endpoint invoked once, then removed in a `finally` block |
| Lasting API and Streamlit links | Separate Cloud Run services; public application and OpenAPI documentation |
| ECR and S3 | Artifact Registry and a private Cloud Storage bucket |
| AWS OIDC authentication from GitHub | Workload Identity Federation restricted to this repository's numeric ID, owner ID and `main` branch |
| Persistent prediction, alert and review records | Cloud SQL for PostgreSQL |
| MLflow tracking and gated model registry | Private Cloud Run MLflow server, PostgreSQL backend and Cloud Storage artifact proxy |
| CD smoke test | `/ready`, `/version` matching the exact source SHA, prediction, recommendation, assistant, monitoring and UI health |
| Budget evidence | CAD 10 alert before billed compute, scale-to-zero services, one instance per revision, short training jobs, endpoint teardown |

## Runbook

The dedicated project is `boreal-marche-eraldo57`, in `northamerica-northeast1` (Montréal).
Resource identifiers and execution results are kept in `reports/current_run/google_cloud/`.
Passwords are stored in Secret Manager. No service-account key is created.

```bash
python -m scripts.gcp build
python -m scripts.gcp deploy
python -m scripts.vertex_jobs train
python -m scripts.vertex_jobs tune --trials 12
python -m scripts.vertex_jobs endpoint-demo
python -m scripts.gcp smoke
```

The first build runs on the development machine and publishes a checksummed runtime archive.
Subsequent CI builds restore that archive inside Cloud Build. It includes the trained artifact,
customer feature store, Chroma index and cached multilingual embedding weights. Raw transaction
data, local environments and credentials are excluded from the cloud build.

The serving API tries the private MLflow registry's champion alias and falls back to the
checksummed bundled model if no champion exists. Training registers candidates; it does not
promote them or silently acknowledge fairness flags. Existing named-approver promotion and
rollback commands remain in `src.registry` and `governance/PROMOTION_POLICY.md`.

The hosted interface fixes its API destination. Provider choices are explicitly configured
for the hosted deployment; the regex provider needs no paid language-model credentials.
The local Compose stack still includes Ollama and the full provider selector.

## Cost and teardown

Cloud Run services use zero minimum instances and one maximum instance per revision.
The API uses instance-based CPU allocation so its automated alert loop runs between requests;
it still scales to zero after its idle period. MLflow and the interface use request-based billing.
Open Streamlit WebSocket sessions count as active requests. Close the demo tabs when finished.
The small PostgreSQL instance and stored images have ongoing charges even when the web
services scale to zero. A budget alert is a notification, not an absolute spending guarantee.
Full training is limited to 600 seconds per worker; tuning trials allow 1800 seconds and
at most four failed trials. The launcher cancels a full
job after 25 minutes or a tuning job after one hour, including queue time. Vertex's
`maxWaitDuration` field applies to Flex-start scheduling and is omitted for Spot workers.

The endpoint demo records creation, prediction and teardown evidence. If interrupted by the
operator or machine, inspect Vertex AI endpoints before leaving the account unattended.

To stop database compute after assessment while preserving its data:

```bash
gcloud sql instances patch boreal-db --activation-policy=NEVER --project=boreal-marche-eraldo57
```

Persistent disk and artifact storage charges continue until those resources are deleted.
Delete this dedicated project's Cloud Run services and database when the portfolio/assessment
period ends. Review the Cloud Billing report for actual charges; cost reporting can lag usage.
Never substitute an estimate for an actual billed-cost record.

## Verification status

See the [execution report](../reports/current_run/google_cloud/EXECUTION_REPORT.md) for live
links, completed checks, managed-job results, cost evidence and remaining owner tasks.
The JSON files beside it preserve actual cloud responses. A configuration file alone is
not evidence of a successful job, endpoint invocation or teardown.
