"""Recommendation wording: every list says where it came from."""

from __future__ import annotations

from frontend.notes import STRATEGY_NOTES, reason_lines, strategy_note

RULE = {
    "antecedent": ["white hanging heart t-light holder"],
    "consequent": ["red hanging heart t-light holder"],
    "confidence": 0.231,
    "lift": 4.7,
    "support": 0.035,
}


def test_every_strategy_the_api_can_return_has_a_note():
    # The API's three strategies (src/recommend.py); an unknown one must not pass silently.
    assert set(STRATEGY_NOTES) == {"empty_basket", "popularity_fallback", "rule_based"}
    assert strategy_note({"strategy": "something_new"})[0] == "warning"


def test_popularity_lists_are_not_presented_as_personal():
    level, text = strategy_note({"strategy": "popularity_fallback"})
    assert level == "warning" and "most popular" in text and "not suggestions" in text
    assert "not personalised" in strategy_note({"strategy": "empty_basket"})[1]


def test_rule_based_lines_name_the_trigger_and_the_strength():
    result = {
        "strategy": "rule_based",
        "recommendations": ["red hanging heart t-light holder"],
        "because": [RULE],
    }
    (line,) = reason_lines(result)
    assert "because you added white hanging heart t-light holder" in line
    assert "23%" in line and "4.7 times" in line


def test_a_list_without_rules_has_no_reason_lines():
    result = {"strategy": "popularity_fallback", "recommendations": ["a", "b"], "because": []}
    assert reason_lines(result) == []


def test_effect_lines_say_who_is_compared_and_in_which_direction():
    from frontend.notes import effect_lines

    prediction = {
        "explanation": [
            {"feature": "recency_days", "value": 394.0, "typical_value": 54.0, "effect": -0.185},
            {"feature": "avg_basket_value", "value": 12.4, "typical_value": 9.8, "effect": 0.021},
        ]
    }
    first, second = effect_lines(prediction)
    assert "recency days" in first and "394" in first and "54" in first
    assert "lowers the chance of returning by 18.5 percentage points" in first
    assert "12.4" in second and "raises the chance of returning by 2.1 percentage points" in second
    assert effect_lines({}) == [], "no explanation, no lines, no error"
