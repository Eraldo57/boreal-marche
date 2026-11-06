"""Start the private MLflow server with credentials supplied through Secret Manager."""

import os


def main() -> None:
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
            os.environ["MLFLOW_BACKEND_STORE_URI"],
            "--artifacts-destination",
            os.environ["MLFLOW_ARTIFACTS_DESTINATION"],
        ],
    )


if __name__ == "__main__":
    main()
