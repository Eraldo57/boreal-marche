"""Short-lived identity tokens for private Google Cloud Run services."""

import os


def configure_tracking_auth() -> None:
    """Refresh MLflow's bearer token through the workload's Google identity."""
    if os.getenv("MLFLOW_GOOGLE_IAM_AUTH", "").lower() != "true":
        return
    from google.auth.transport.requests import Request
    from google.oauth2.id_token import fetch_id_token

    audience = os.environ["MLFLOW_TRACKING_URI"].rstrip("/")
    os.environ["MLFLOW_TRACKING_TOKEN"] = fetch_id_token(Request(), audience)
