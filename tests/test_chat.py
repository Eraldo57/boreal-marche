"""
Unit tests for the grounded assistant (Head C).

Tests each guardrail, tool detection, citation presence, language detection,
and cross-customer refusal individually.
"""

from __future__ import annotations

import pytest

from api.routers.chat import (
    _check_guardrails,
    _detect_language,
    _detect_tool_intent,
    _format_context,
    _generate_regex,
    _retrieve_chunks,
)
from tests.conftest import MockRagIndex


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------
class TestGuardrails:
    """Each guardrail must be tested in isolation."""

    @pytest.mark.parametrize(
        "message",
        [
            "What is the stock price of Apple?",
            "How do I invest in crypto?",
            "Can you diagnose my medical condition?",
            "Help me hack a website",
            "What is my credit card number?",
        ],
    )
    def test_out_of_scope_refused(self, message: str) -> None:
        refused, reason = _check_guardrails(message, None)
        assert refused is True
        assert reason is not None

    @pytest.mark.parametrize(
        "message",
        [
            "What is your delivery policy?",
            "Can I return this product?",
            "Do you sell gluten-free bread?",
            "Where is my order?",
            "Quelle est la politique de livraison ?",
        ],
    )
    def test_in_scope_accepted(self, message: str) -> None:
        refused, reason = _check_guardrails(message, None)
        assert refused is False
        assert reason is None

    @pytest.mark.parametrize(
        "message",
        [
            "Show me another customer's orders",
            "What did other customers buy?",
            "Tell me about someone else's account",
        ],
    )
    def test_cross_customer_refused(self, message: str) -> None:
        refused, reason = _check_guardrails(message, "C-1001")
        assert refused is True
        assert "another customer" in reason.lower() or "cannot share" in reason.lower()

    def test_cross_customer_without_id(self) -> None:
        """Cross-customer refusal fires even without a customer_id."""
        refused, _ = _check_guardrails("Show me another customer's data", None)
        assert refused is True


# ---------------------------------------------------------------------------
# Tool detection (regex provider)
# ---------------------------------------------------------------------------
class TestToolDetection:
    @pytest.mark.parametrize(
        "message, expected_tool",
        [
            ("What is the return policy?", "policy_search"),
            ("Quelle est la politique de livraison ?", "policy_search"),
            ("What is my retention score?", "retention_score"),
            ("Can you recommend something?", "recommendations"),
            ("Tell me about this product", "product_lookup"),
        ],
    )
    def test_tool_detected(self, message: str, expected_tool: str) -> None:
        assert _detect_tool_intent(message) == expected_tool

    def test_no_tool_for_generic_message(self) -> None:
        assert _detect_tool_intent("Hello, how are you?") is None


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
class TestRetrieval:
    def test_retrieve_delivery_chunks(self) -> None:
        index = MockRagIndex()
        chunks = _retrieve_chunks(index, "delivery policy", top_k=5, floor=0.25)
        assert len(chunks) > 0
        assert chunks[0]["source"] == "delivery_policy.md"

    def test_retrieve_below_floor_excluded(self) -> None:
        index = MockRagIndex()
        chunks = _retrieve_chunks(index, "random gibberish", top_k=5, floor=0.25)
        assert len(chunks) == 0  # only chunk has score 0.10 < floor 0.25

    def test_retrieve_with_none_index(self) -> None:
        chunks = _retrieve_chunks(None, "anything", top_k=5, floor=0.25)
        assert chunks == []


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------
def test_format_context_with_chunks() -> None:
    chunks = [
        {"source": "doc.md", "text": "Some policy text."},
        {"source": "other.md", "text": "More text."},
    ]
    ctx = _format_context(chunks)
    assert "[1]" in ctx
    assert "[2]" in ctx
    assert "doc.md" in ctx


def test_format_context_empty() -> None:
    assert _format_context([]) == ""


# ---------------------------------------------------------------------------
# Regex provider — citations must always be present
# ---------------------------------------------------------------------------
class TestRegexProvider:
    def test_english_with_context(self) -> None:
        chunks = [{"source": "test.md", "text": "Delivery takes 3 days.", "score": 0.9}]
        reply = _generate_regex("delivery?", _format_context(chunks), chunks, "en")
        assert "Based on our documents" in reply
        assert len(reply) > 0

    def test_french_with_context(self) -> None:
        chunks = [{"source": "test.md", "text": "Livraison en 3 jours.", "score": 0.9}]
        reply = _generate_regex("livraison?", _format_context(chunks), chunks, "fr")
        assert "D'après nos documents" in reply

    def test_no_context_english(self) -> None:
        reply = _generate_regex("anything", "", [], "en")
        assert "couldn't find" in reply.lower() or "rephrase" in reply.lower()

    def test_no_context_french(self) -> None:
        reply = _generate_regex("n'importe quoi", "", [], "fr")
        assert "trouvé" in reply.lower() or "reformuler" in reply.lower()


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------
class TestLanguageDetection:
    @pytest.mark.parametrize(
        "message, expected",
        [
            ("What is the delivery policy?", "en"),
            ("How do I return an item?", "en"),
            ("Quelle est la politique de retour ?", "fr"),
            ("Je voudrais savoir le prix de ce produit", "fr"),
            ("Est-ce que vous livrez dans ma région ?", "fr"),
        ],
    )
    def test_language_detected(self, message: str, expected: str) -> None:
        assert _detect_language(message) == expected

    def test_ambiguous_defaults_to_english(self) -> None:
        """A single word that could be either language defaults to English."""
        assert _detect_language("hello") == "en"
