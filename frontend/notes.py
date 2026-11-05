"""Plain-language wording for what the interface shows, kept apart so tests can check it.

A recommendation must say where it came from. A rule fired because of items in the basket,
or the list is simply the most popular items and has nothing to do with this basket. The
customer and the analyst should be able to tell which, without reading JSON.
"""
# Developed with AI assistance; verification/provenance: docs/AI_USE.md.

from __future__ import annotations

from typing import Any

# (level, text): level names a Streamlit message, "info" or "warning".
STRATEGY_NOTES = {
    "empty_basket": (
        "info",
        "Your basket is empty, so these are the most popular items overall. "
        "They are not personalised.",
    ),
    "popularity_fallback": (
        "warning",
        "No rule matches this basket, so these are the most popular items overall, "
        "not suggestions based on what you added.",
    ),
    "rule_based": (
        "info",
        "These come from association rules mined from past baskets, not from a model of you.",
    ),
}


def _number(value: float) -> str:
    return f"{value:,.0f}" if float(value).is_integer() or abs(value) >= 100 else f"{value:,.1f}"


def effect_lines(prediction: dict[str, Any]) -> list[str]:
    """One plain-language line per feature that moves this customer's own score.

    The effect is the change in the probability of returning when the feature has this
    customer's value instead of the typical (training median) one.
    """
    lines = []
    for item in prediction.get("explanation", []):
        name = item["feature"].replace("_", " ")
        points = abs(item["effect"]) * 100
        direction = "raises" if item["effect"] > 0 else "lowers"
        lines.append(
            f"**{name}**: {_number(item['value'])} for this customer, "
            f"{_number(item['typical_value'])} for a typical one. "
            f"This {direction} the chance of returning by {points:.1f} percentage points."
        )
    return lines


def strategy_note(result: dict[str, Any]) -> tuple[str, str]:
    """The message that says which path produced this list."""
    return STRATEGY_NOTES.get(
        result["strategy"], ("warning", f"Unknown recommendation strategy: {result['strategy']}")
    )


def reason_lines(result: dict[str, Any]) -> list[str]:
    """One line per recommended item: the basket items that triggered it, and how strong."""
    lines = []
    for item, rule in zip(result["recommendations"], result["because"]):
        added = " + ".join(rule["antecedent"])
        lines.append(
            f"**{item}**: because you added {added}. "
            f"{rule['confidence']:.0%} of past baskets with {added} also had it "
            f"(about {rule['lift']:.1f} times as often as in a typical basket)."
        )
    return lines
