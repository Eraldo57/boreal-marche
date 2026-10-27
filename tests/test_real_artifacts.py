"""Optional integration checks: never substitute mock artifacts for real evidence."""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.contracts import FEATURE_NAMES, METADATA_PATH, PROCESSED, PROJECT_ROOT
from src.ingest import _sha256
from src.rag.index import RagIndex

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not METADATA_PATH.exists() or not (PROJECT_ROOT / "data/chromadb/manifest.json").exists(),
        reason="Run the real local pipeline to generate model and RAG artifacts",
    ),
]


def test_data_quality_real_manifest():
    manifest = json.loads((PROCESSED / "manifest.json").read_text())
    transactions = pd.read_parquet(PROCESSED / "transactions.parquet")
    assert len(transactions) == manifest["row_count"]
    assert _sha256(PROCESSED / "transactions.parquet") == manifest["processed_sha256"]
    assert transactions["IsPurchase"].sum() == manifest["purchase_rows"]
    assert transactions["IsReturn"].sum() > 0
    assert not transactions.loc[transactions["IsPurchase"], "IsReturn"].any()
    assert list(pd.read_parquet(PROCESSED / "X.parquet").columns) == FEATURE_NAMES


def test_multilingual_retrieval_real_index():
    hits = RagIndex().query("Quel est le délai pour retourner un produit non périssable ?", top_k=5)
    assert any(h["source"] == "return_policy.md" and h["language"] == "en" for h in hits)


def test_startup_and_predictions_real_artifacts():
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").json()["ready"] is True
        record = client.get("/customers/12347/features").json()
        payload = {"customer_id": "12347", "features": record["features"]}
        single = client.post("/predict", json=payload)
        assert single.status_code == 200
        batch = client.post("/predict/batch", json={"customers": [payload]}).json()
        assert batch["predictions"][0]["probability"] == single.json()["probability"]
        assert 0 <= single.json()["probability"] <= 1
        recommendation = client.post("/recommend", json={"basket": []}).json()
        assert recommendation["strategy"] == "empty_basket"
        assert recommendation["recommendations"]
        fallback = client.post("/recommend", json={"basket": ["never-seen-item"]}).json()
        assert fallback["strategy"] == "popularity_fallback"
        assert fallback["recommendations"] == recommendation["recommendations"]
