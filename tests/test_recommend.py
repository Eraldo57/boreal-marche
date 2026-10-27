"""
Unit tests for the recommendation engine (RuleEngine in api/lifespan.py).
"""

from __future__ import annotations

import pytest

from api.lifespan import RuleEngine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
RULES = [
    {
        "antecedent": ["bread"],
        "consequent": ["butter"],
        "confidence": 0.80,
        "lift": 4.0,
        "support": 0.05,
    },
    {
        "antecedent": ["bread"],
        "consequent": ["jam"],
        "confidence": 0.60,
        "lift": 3.0,
        "support": 0.03,
    },
    {
        "antecedent": ["bread", "butter"],
        "consequent": ["milk"],
        "confidence": 0.55,
        "lift": 2.5,
        "support": 0.02,
    },
    {
        "antecedent": ["eggs"],
        "consequent": ["bacon"],
        "confidence": 0.70,
        "lift": 3.5,
        "support": 0.04,
    },
    {
        "antecedent": ["eggs"],
        "consequent": ["bread"],  # bread is already in the basket in some tests
        "confidence": 0.50,
        "lift": 2.0,
        "support": 0.02,
    },
]

POPULARITY = ["coffee", "bananas", "rice", "pasta", "olive oil"]


@pytest.fixture
def engine() -> RuleEngine:
    e = RuleEngine(rules=RULES, popularity_fallback=POPULARITY)
    e.build_index()
    return e


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------
def test_index_built(engine: RuleEngine) -> None:
    """After build_index, the internal index must have entries."""
    assert len(engine._index) > 0


def test_index_is_by_individual_item(engine: RuleEngine) -> None:
    for key in engine._index:
        assert isinstance(key, str)
        assert all(key in rule.antecedent for rule in engine._index[key])


# ---------------------------------------------------------------------------
# Rule-based recommendations
# ---------------------------------------------------------------------------
def test_recommend_returns_matching_items(engine: RuleEngine) -> None:
    recs, rules, strategy = engine.recommend(["bread"], top_k=5)
    assert strategy == "rule_based"
    assert "butter" in recs
    assert "jam" in recs


def test_recommend_respects_top_k(engine: RuleEngine) -> None:
    recs, _, _ = engine.recommend(["bread"], top_k=1)
    assert len(recs) == 1


def test_recommend_sorted_by_lift(engine: RuleEngine) -> None:
    """The first recommendation should come from the highest-lift rule."""
    recs, rules, _ = engine.recommend(["bread"], top_k=5)
    if len(rules) >= 2:
        assert rules[0]["lift"] >= rules[1]["lift"]


# ---------------------------------------------------------------------------
# Deduplication by consequent
# ---------------------------------------------------------------------------
def test_no_duplicate_recommendations(engine: RuleEngine) -> None:
    recs, _, _ = engine.recommend(["bread", "eggs"], top_k=10)
    assert len(recs) == len(set(r.lower() for r in recs))


# ---------------------------------------------------------------------------
# Items already in basket are excluded
# ---------------------------------------------------------------------------
def test_basket_items_excluded(engine: RuleEngine) -> None:
    """If 'bread' is in the basket AND a rule consequent, it must be excluded."""
    recs, _, _ = engine.recommend(["bread", "eggs"], top_k=10)
    assert "bread" not in [r.lower() for r in recs]


def test_basket_items_excluded_case_insensitive(engine: RuleEngine) -> None:
    """Exclusion must be case-insensitive."""
    recs, _, _ = engine.recommend(["BREAD", "Eggs"], top_k=10)
    assert "bread" not in [r.lower() for r in recs]


# ---------------------------------------------------------------------------
# Multi-item antecedent matching
# ---------------------------------------------------------------------------
def test_multi_item_antecedent(engine: RuleEngine) -> None:
    """Rules with multi-item antecedents fire only if all items are present."""
    recs_partial, _, _ = engine.recommend(["butter"], top_k=10)
    recs_full, _, _ = engine.recommend(["bread", "butter"], top_k=10)

    # "milk" should appear only when both bread AND butter are in the basket
    assert "milk" not in recs_partial
    assert "milk" in recs_full


# ---------------------------------------------------------------------------
# Empty basket
# ---------------------------------------------------------------------------
def test_empty_basket_returns_popularity(engine: RuleEngine) -> None:
    recs, rules, strategy = engine.recommend([], top_k=3)
    assert strategy == "empty_basket"
    assert len(recs) == 3
    assert rules == []
    assert recs == POPULARITY[:3]


# ---------------------------------------------------------------------------
# No-match fallback
# ---------------------------------------------------------------------------
def test_no_match_returns_popularity(engine: RuleEngine) -> None:
    recs, rules, strategy = engine.recommend(["dragon fruit"], top_k=3)
    assert strategy == "popularity_fallback"
    assert len(recs) == 3
    assert rules == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_empty_rules_engine() -> None:
    e = RuleEngine(rules=[], popularity_fallback=POPULARITY)
    e.build_index()
    recs, _, strategy = e.recommend(["bread"], top_k=3)
    assert strategy == "popularity_fallback"


def test_popularity_list_shorter_than_top_k() -> None:
    e = RuleEngine(rules=[], popularity_fallback=["coffee"])
    e.build_index()
    recs, _, strategy = e.recommend([], top_k=5)
    assert len(recs) == 1  # only one item in fallback
