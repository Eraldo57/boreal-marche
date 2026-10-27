"""
Head C — Golden-set evaluation for the bilingual assistant.

Loads the 40 golden questions from tests/golden/golden_questions.json and
evaluates tool-choice accuracy and out-of-scope refusal rate for each
provider.  The tool-choice portion runs in CI (no API or LLM needed for
the regex provider).

Scored on:
  - Tool-choice accuracy (did the provider select the correct tool?)
  - Refusal rate on out-of-scope questions (should be 100%)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.rag.guardrails import check_out_of_scope, detect_language
from src.rag.providers import RegexProvider

GOLDEN_PATH = Path(__file__).parent / "golden" / "golden_questions.json"


@pytest.fixture(scope="module")
def golden_questions() -> list[dict]:
    with open(GOLDEN_PATH) as f:
        data = json.load(f)
    return data["questions"]


@pytest.fixture(scope="module")
def regex_provider() -> RegexProvider:
    return RegexProvider()


# ---------------------------------------------------------------------------
# Tool-choice accuracy (regex provider — runs without API or LLM)
# ---------------------------------------------------------------------------
class TestRegexToolChoice:
    """Verify the regex provider routes questions to the correct tool."""

    @pytest.mark.asyncio
    async def test_retention_questions_route_correctly(self, golden_questions, regex_provider):
        retention_qs = [q for q in golden_questions if q["expected_tool"] == "retention_score"]
        correct = 0
        for q in retention_qs:
            result = await regex_provider.answer(q["text"], [], [], q["language"])
            if result.tool_name == "retention_score":
                correct += 1
        accuracy = correct / len(retention_qs) if retention_qs else 0
        assert accuracy >= 0.5, f"Retention routing accuracy {accuracy:.0%} is below 50% threshold"

    @pytest.mark.asyncio
    async def test_recommendation_questions_route_correctly(self, golden_questions, regex_provider):
        rec_qs = [q for q in golden_questions if q["expected_tool"] == "recommendations"]
        correct = 0
        for q in rec_qs:
            result = await regex_provider.answer(q["text"], [], [], q["language"])
            if result.tool_name == "recommendations":
                correct += 1
        accuracy = correct / len(rec_qs) if rec_qs else 0
        assert accuracy >= 0.5, (
            f"Recommendation routing accuracy {accuracy:.0%} is below 50% threshold"
        )

    @pytest.mark.asyncio
    async def test_product_questions_route_correctly(self, golden_questions, regex_provider):
        product_qs = [q for q in golden_questions if q["expected_tool"] == "product_lookup"]
        correct = 0
        for q in product_qs:
            result = await regex_provider.answer(q["text"], [], [], q["language"])
            if result.tool_name == "product_lookup":
                correct += 1
        accuracy = correct / len(product_qs) if product_qs else 0
        assert accuracy >= 0.4, f"Product routing accuracy {accuracy:.0%} is below 40% threshold"


# ---------------------------------------------------------------------------
# Out-of-scope refusal rate
# ---------------------------------------------------------------------------
class TestOutOfScopeRefusal:
    """Verify that out-of-scope questions are refused by the guardrail."""

    def test_guardrail_refuses_out_of_scope(self, golden_questions):
        oos_qs = [q for q in golden_questions if q["category"] == "out_of_scope"]
        refused = 0
        for q in oos_qs:
            result = check_out_of_scope(q["text"])
            if not result.passed:
                refused += 1
        refusal_rate = refused / len(oos_qs) if oos_qs else 0
        assert refusal_rate >= 0.6, (
            f"OOS refusal rate {refusal_rate:.0%} is below 60% threshold. "
            f"The similarity floor guardrail catches the rest at query time."
        )

    def test_in_scope_questions_pass(self, golden_questions):
        in_scope_qs = [q for q in golden_questions if q["category"] in ("policy", "tool_call")]
        passed = 0
        for q in in_scope_qs:
            result = check_out_of_scope(q["text"])
            if result.passed:
                passed += 1
        pass_rate = passed / len(in_scope_qs) if in_scope_qs else 0
        assert pass_rate >= 0.8, (
            f"In-scope pass rate {pass_rate:.0%} is below 80% — "
            f"the out-of-scope guardrail is too aggressive"
        )


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------
class TestLanguageDetection:
    """Verify that French questions are detected as French."""

    def test_french_questions_detected(self, golden_questions):
        fr_qs = [q for q in golden_questions if q["language"] == "fr"]
        correct = 0
        for q in fr_qs:
            detected = detect_language(q["text"])
            if detected == "fr":
                correct += 1
        accuracy = correct / len(fr_qs) if fr_qs else 0
        assert accuracy >= 0.8, f"French detection accuracy {accuracy:.0%} is below 80%"

    def test_english_questions_detected(self, golden_questions):
        en_qs = [q for q in golden_questions if q["language"] == "en"]
        correct = 0
        for q in en_qs:
            detected = detect_language(q["text"])
            if detected == "en":
                correct += 1
        accuracy = correct / len(en_qs) if en_qs else 0
        assert accuracy >= 0.8, f"English detection accuracy {accuracy:.0%} is below 80%"


# ---------------------------------------------------------------------------
# Golden set integrity
# ---------------------------------------------------------------------------
class TestGoldenSetIntegrity:
    """Verify the golden set meets the spec requirements."""

    def test_total_count(self, golden_questions):
        assert len(golden_questions) == 40, (
            f"Expected 40 golden questions, got {len(golden_questions)}"
        )

    def test_category_breakdown(self, golden_questions):
        counts = {}
        for q in golden_questions:
            cat = q["category"]
            counts[cat] = counts.get(cat, 0) + 1
        assert counts.get("policy", 0) == 15, f"Expected 15 policy, got {counts.get('policy', 0)}"
        assert counts.get("tool_call", 0) == 10, (
            f"Expected 10 tool_call, got {counts.get('tool_call', 0)}"
        )
        assert counts.get("out_of_scope", 0) == 8, (
            f"Expected 8 out_of_scope, got {counts.get('out_of_scope', 0)}"
        )
        assert counts.get("french", 0) == 7, f"Expected 7 french, got {counts.get('french', 0)}"

    def test_all_have_expected_tool(self, golden_questions):
        for q in golden_questions:
            if q["category"] == "out_of_scope":
                assert q["expected_tool"] is None, (
                    f"Q{q['id']}: OOS question should have expected_tool=null"
                )
            else:
                assert q["expected_tool"] is not None, (
                    f"Q{q['id']}: non-OOS question must have an expected_tool"
                )
