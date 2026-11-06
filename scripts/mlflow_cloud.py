"""Start the private MLflow server with credentials supplied through Secret Manager."""

import os


def main() -> None:
    # MLflow 2 passes string experiment IDs into integer PostgreSQL columns.
    # psycopg2's client-side binding supports this; psycopg3's typed binding does not.
    backend = os.environ["MLFLOW_BACKEND_STORE_URI"].replace(
        "postgresql+psycopg://", "postgresql+psycopg2://", 1
    )
    os.execvp(
        "mlflow",
        [
            "mlflow",
            "server",
            "--host",
            "0.0.0.0",
            "--port",
            os.getenv("PORT", "8080"),
            "--workers",
            "1",
            "--serve-artifacts",
            "--backend-store-uri",
            backend,
            "--artifacts-destination",
            os.environ["MLFLOW_ARTIFACTS_DESTINATION"],
        ],
    )


if __name__ == "__main__":
    main()
