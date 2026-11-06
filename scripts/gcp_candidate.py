"""Fetch a completed managed candidate for the existing local promotion-gate checker."""

import json
from datetime import UTC, datetime

from scripts.gcp import EVIDENCE, ROOT, gcloud


def main() -> None:
    cfg = json.loads((EVIDENCE / "settings.json").read_text())
    destination = ROOT / "artifacts/candidates" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination.mkdir(parents=True, exist_ok=True)
    gcloud(
        cfg, "storage", "cp", "--recursive", cfg["trained_artifact_uri"] + "/*", str(destination)
    )
    if not (destination / "retention_model_metadata.json").exists():
        raise RuntimeError("Managed candidate metadata was not downloaded")
    print(f"Candidate evidence: {destination}")


if __name__ == "__main__":
    main()
