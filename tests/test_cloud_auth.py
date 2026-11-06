"""Cloud identity authentication stays opt-in and never uses stored access keys."""

from src.cloud_auth import configure_tracking_auth


def test_local_tracking_does_not_request_a_cloud_token(monkeypatch):
    monkeypatch.delenv("MLFLOW_GOOGLE_IAM_AUTH", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)
    configure_tracking_auth()
    import os

    assert "MLFLOW_TRACKING_TOKEN" not in os.environ


def test_cloud_tracking_uses_service_url_as_identity_audience(monkeypatch):
    from google.oauth2 import id_token

    seen = []
    monkeypatch.setenv("MLFLOW_GOOGLE_IAM_AUTH", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://tracking.example.run.app/")
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "")
    monkeypatch.setattr(
        id_token, "fetch_id_token", lambda request, audience: seen.append(audience) or "ephemeral"
    )
    configure_tracking_auth()
    import os

    assert seen == ["https://tracking.example.run.app"]
    assert os.environ["MLFLOW_TRACKING_TOKEN"] == "ephemeral"
