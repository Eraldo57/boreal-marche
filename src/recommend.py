"""
Boréal Marché — Recommendation Serving Engine (Head B)
=======================================================

Loads the mined rules from ``artifacts/rules.json``, builds an inverted
index keyed by individual antecedent items for **O(1)** lookup, and
serves top-*k* recommendations for a given basket.

Design
------
* At startup the engine loads the JSON once and pre-builds two indexes:
  1. ``_by_item``: maps each antecedent item to the list of rules that
     contain it.  This allows efficient lookup without scanning every rule.
  2. ``_popularity``: a ranked fallback list of globally popular items
     used when the basket is empty or no rules match.

* At query time the engine:
  1. Finds candidate rules whose antecedent is a subset of the basket.
  2. Among candidates, deduplicates by consequent — keeping the rule
     with the **highest lift** for each recommended item.
  3. Excludes items already in the basket.
  4. Returns the top-*k* items ordered by lift.

* This module is **pure Python + stdlib** at serving time.  No ML
  framework is imported.

Usage::

    from src.recommend import RecommendationEngine

    engine = RecommendationEngine.load()
    recs = engine.recommend(["white hanging heart t-light holder"])
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RULES_PATH = PROJECT_ROOT / "artifacts" / "rules.json"
POPULARITY_PATH = PROJECT_ROOT / "artifacts" / "popularity.json"


class _Rule:
    """Lightweight internal representation of one association rule."""

    __slots__ = ("antecedent", "consequent", "confidence", "lift", "support")

    def __init__(
        self,
        antecedent: frozenset[str],
        consequent: frozenset[str],
        confidence: float,
        lift: float,
        support: float,
    ) -> None:
        self.antecedent = antecedent
        self.consequent = consequent
        self.confidence = confidence
        self.lift = lift
        self.support = support

    def to_dict(self) -> dict[str, Any]:
        return {
            "antecedent": sorted(self.antecedent),
            "consequent": sorted(self.consequent),
            "confidence": self.confidence,
            "lift": self.lift,
            "support": self.support,
        }


class RecommendationEngine:
    """Serve basket-level product recommendations from pre-mined rules.

    Parameters
    ----------
    rules : list[_Rule]
        Parsed association rules.
    popularity : list[str]
        Globally popular items for the fallback strategy.
    default_k : int
        Default number of recommendations to return.
    """

    def __init__(
        self,
        rules: list[_Rule],
        popularity: list[str],
        default_k: int = 5,
    ) -> None:
        self._rules = rules
        self._popularity = popularity
        self._default_k = default_k

        # Build the inverted index: item → list of rules
        self._by_item: dict[str, list[_Rule]] = defaultdict(list)
        for rule in rules:
            for item in rule.antecedent:
                self._by_item[item].append(rule)

        # Precompute set of all antecedent items for quick "item known?" check
        self._known_items: set[str] = set(self._by_item.keys())

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------
    @classmethod
    def load(
        cls,
        rules_path: str | Path = RULES_PATH,
        popularity_path: str | Path = POPULARITY_PATH,
        default_k: int = 5,
    ) -> "RecommendationEngine":
        """Load rules and popularity from their JSON artifacts.

        Parameters
        ----------
        rules_path : path-like
            Path to ``rules.json``.
        popularity_path : path-like
            Path to ``popularity.json``.
        default_k : int
            Default number of recommendations.

        Returns
        -------
        RecommendationEngine

        Raises
        ------
        FileNotFoundError
            If either artifact file is missing.
        """
        with open(rules_path) as f:
            data = json.load(f)

        rules: list[_Rule] = []
        for r in data["rules"]:
            rules.append(_Rule(
                antecedent=frozenset(r["antecedent"]),
                consequent=frozenset(r["consequent"]),
                confidence=r["confidence"],
                lift=r["lift"],
                support=r["support"],
            ))

        popularity: list[str] = []
        pop_path = Path(popularity_path)
        if pop_path.exists():
            with open(pop_path) as f:
                popularity = json.load(f).get("items", [])

        return cls(rules, popularity, default_k)

    # -----------------------------------------------------------------
    # Query
    # -----------------------------------------------------------------
    def recommend(
        self,
        basket: list[str],
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Return recommendations for a basket.

        Parameters
        ----------
        basket : list[str]
            Product descriptions currently in the basket.  Case-insensitive
            matching is applied internally.
        top_k : int or None
            Number of recommendations.  Defaults to ``self._default_k``.

        Returns
        -------
        dict
            Keys: ``recommendations`` (list[str]),
            ``because`` (list[dict] — the firing rules),
            ``strategy`` (str — "rule_based", "popularity_fallback",
            or "empty_basket").
        """
        k = top_k if top_k is not None else self._default_k

        # Normalise basket to lowercase for matching
        basket_set = {item.lower().strip() for item in basket}

        # --- Empty basket -------------------------------------------------
        if not basket_set:
            return {
                "recommendations": self._popularity[:k],
                "because": [],
                "strategy": "empty_basket",
            }

        # --- Find matching rules ------------------------------------------
        # A rule matches if its antecedent is a SUBSET of the basket.
        candidate_rules: list[_Rule] = []
        checked: set[int] = set()  # avoid checking the same rule twice

        for item in basket_set:
            for rule in self._by_item.get(item, []):
                rule_id = id(rule)
                if rule_id in checked:
                    continue
                checked.add(rule_id)
                if rule.antecedent.issubset(basket_set):
                    candidate_rules.append(rule)

        # --- Deduplicate by consequent item, keeping highest lift ---------
        best_by_item: dict[str, _Rule] = {}
        for rule in candidate_rules:
            for cons_item in rule.consequent:
                if cons_item in basket_set:
                    continue  # exclude items already in the basket
                existing = best_by_item.get(cons_item)
                if existing is None or rule.lift > existing.lift:
                    best_by_item[cons_item] = rule

        if not best_by_item:
            # --- No match fallback ----------------------------------------
            # Exclude items already in the basket from popularity
            fallback = [i for i in self._popularity if i.lower().strip() not in basket_set]
            return {
                "recommendations": fallback[:k],
                "because": [],
                "strategy": "popularity_fallback",
            }

        # --- Rank by lift and take top-k ----------------------------------
        ranked = sorted(best_by_item.items(), key=lambda x: -x[1].lift)[:k]

        recommendations = [item for item, _ in ranked]
        because = [rule.to_dict() for _, rule in ranked]

        return {
            "recommendations": recommendations,
            "because": because,
            "strategy": "rule_based",
        }

    @property
    def n_rules(self) -> int:
        """Number of loaded rules."""
        return len(self._rules)

    @property
    def n_indexed_items(self) -> int:
        """Number of distinct antecedent items in the index."""
        return len(self._by_item)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
def main() -> None:
    """Quick smoke test: load the engine and recommend for a sample basket."""
    engine = RecommendationEngine.load()
    print(f"Loaded {engine.n_rules} rules, {engine.n_indexed_items} indexed items\n")

    test_baskets: list[list[str]] = [
        [],  # empty basket
        ["white hanging heart t-light holder"],
        ["red woolly hottie white heart.", "set 7 babushka nesting boxes"],
        ["nonexistent-product-xyz"],
    ]

    for basket in test_baskets:
        result = engine.recommend(basket, top_k=5)
        print(f"Basket: {basket}")
        print(f"  Strategy: {result['strategy']}")
        print(f"  Recommendations: {result['recommendations'][:3]}…")
        print()


if __name__ == "__main__":
    main()
