"""The basket picker must expose all loaded items, including fallback-only products."""

import pytest

from src.recommend import RecommendationEngine


def test_details_include_all_products_and_preserve_multi_item_rules():
    rules = [
        {
            "antecedent": ["a", "b"],
            "consequent": ["c"],
            "support": 0.1,
            "confidence": 0.8,
            "lift": 2.0,
        }
    ]
    engine = RecommendationEngine.from_records(rules, ["a", "fallback only"])
    details = engine.describe()
    assert details["items"] == ["a", "b", "c", "fallback only"]
    assert set(details["rules"][0]["antecedent"]) == {"a", "b"}
    assert details["rules"][0]["consequent"] == ["c"]
    assert engine.recommend(["a"])["strategy"] == "popularity_fallback"
    assert engine.recommend(["a", "b"])["recommendations"] == ["c"]


@pytest.mark.asyncio
async def test_details_endpoint_lists_loaded_rules_and_items(client):
    response = await client.get("/recommend/details")
    assert response.status_code == 200
    details = response.json()
    assert details["items"] == sorted(set(details["items"]))
    assert "organic maple syrup" in details["items"]
    assert details["rules"]
    assert all(
        set(rule["antecedent"] + rule["consequent"]) <= set(details["items"])
        for rule in details["rules"]
    )


@pytest.mark.asyncio
async def test_details_endpoint_refuses_missing_engine(client, monkeypatch):
    from api.main import app

    monkeypatch.setattr(app.state, "rule_engine", None)
    response = await client.get("/recommend/details")
    assert response.status_code == 503
