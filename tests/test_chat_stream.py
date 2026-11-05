"""
/chat/stream: the same guardrails and evidence as /chat, delivered as NDJSON events.
Language models are replaced by labelled stand-ins; the policy text comes from the mock
index in conftest.py (synthetic, not the real corpus).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import api.routers.chat as chat_router
from scripts.evaluate_assistant import token_cost
from src.rag.providers import AssistantResult, OpenAIProvider, answer_messages, ollama_piece


async def _events(client, payload):
    async with client.stream("POST", "/chat/stream", json=payload) as response:
        assert response.status_code == 200
        return [json.loads(line) async for line in response.aiter_lines() if line]


def _text(events):
    return "".join(e["text"] for e in events if e["type"] == "token")


class _StandInLLM:
    """Routes to policy_search like a model would, and writes from the excerpts."""

    def __init__(self, pieces, fail_after=None):
        self.pieces, self.fail_after = pieces, fail_after

    async def answer(self, message, context_chunks, tools, language):
        return AssistantResult(tool_name="policy_search", tool_args={"question": message})

    async def stream_answer(self, question, excerpts, language):
        for i, piece in enumerate(self.pieces):
            if self.fail_after is not None and i == self.fail_after:
                raise ConnectionError("model went away")
            yield piece


@pytest.mark.asyncio
async def test_regex_stream_matches_the_non_streamed_answer(client):
    payload = {"message": "What is the delivery policy?", "provider": "regex"}
    complete = (await client.post("/chat", json=payload)).json()
    events = await _events(client, payload)
    assert events[0]["type"] == "meta" and events[-1]["type"] == "done"
    assert _text(events) == complete["reply"] == events[-1]["reply"]
    assert events[-1]["citations"] == complete["citations"]
    assert events[-1]["answer_written_by"] == "source text"


@pytest.mark.asyncio
async def test_refusals_stream_too(client):
    events = await _events(client, {"message": "How do I invest in crypto?"})
    assert events[-1]["refused"] is True
    assert _text(events) == events[-1]["reply"] and events[-1]["tool_calls"] == []


@pytest.mark.asyncio
async def test_llm_writes_the_policy_answer_from_cited_excerpts(client, monkeypatch):
    llm = _StandInLLM(["La livraison ", "prend de 2 à 5 jours ouvrables."])
    monkeypatch.setattr(chat_router, "get_provider", lambda *a, **k: llm)
    payload = {
        "message": "Quel est le délai de livraison ?",
        "provider": "ollama",
        "language": "fr",
    }
    events = await _events(client, payload)
    done = events[-1]
    assert done["reply"] == "La livraison prend de 2 à 5 jours ouvrables." == _text(events)
    assert done["answer_written_by"] == "ollama"
    assert done["citations"] and done["citations"][0]["source"] == "delivery_policy.md"


@pytest.mark.asyncio
async def test_failed_generation_falls_back_to_the_source_text(client, monkeypatch):
    llm = _StandInLLM(["Partial ", "answer"], fail_after=1)
    monkeypatch.setattr(chat_router, "get_provider", lambda *a, **k: llm)
    payload = {"message": "What is the delivery policy?", "provider": "ollama"}
    events = await _events(client, payload)
    assert any(e["type"] == "fallback" for e in events)
    done = events[-1]
    assert done["answer_written_by"] == "source text"
    assert done["reply"].startswith("Based on our documents:")


def test_answer_prompt_carries_language_sources_and_the_question():
    messages = answer_messages(
        "Délai ?", [{"source": "delivery_policy.md", "text": "2-5 days"}], "fr"
    )
    assert "French" in messages[0]["content"] and "ONLY" in messages[0]["content"]
    assert "(delivery_policy.md) 2-5 days" in messages[1]["content"]
    assert messages[1]["content"].endswith("Délai ?")


def test_figure_check_flags_invented_and_omitted_figures():
    excerpts = [
        {"text": "Returns are accepted within 30 days of delivery."},
        {"text": "Report damage within 48 hours."},
    ]
    faithful = chat_router.figure_check("Vous avez 30 jours pour retourner un article.", excerpts)
    assert faithful == {"figures_not_in_sources": [], "top_source_figures_missing": []}
    # The real failure seen live: the model said there was no deadline at all.
    omitted = chat_router.figure_check("Il n'y a pas de délai spécifique.", excerpts)
    assert omitted["top_source_figures_missing"] == ["30"]
    invented = chat_router.figure_check("You have 60 days, or 48 hours for damage.", excerpts)
    assert invented["figures_not_in_sources"] == ["60"]


@pytest.mark.asyncio
async def test_written_answers_carry_the_figure_check(client, monkeypatch):
    llm = _StandInLLM(["Deliveries take ", "2 to 5 business days."])
    monkeypatch.setattr(chat_router, "get_provider", lambda *a, **k: llm)
    events = await _events(client, {"message": "Delivery policy?", "provider": "ollama"})
    assert "figure_check" in events[-1]


def test_ollama_stream_lines_are_parsed():
    assert ollama_piece('{"message": {"content": "Bon"}, "done": false}') == ("Bon", False)
    assert ollama_piece('{"done": true}') == ("", True)


@pytest.mark.asyncio
async def test_written_answers_report_the_tokens_the_model_spent(client, monkeypatch):
    llm = _StandInLLM(["Deliveries take ", "2 to 5 business days."])
    llm.last_usage = {"prompt_tokens": 900, "completion_tokens": 40}
    monkeypatch.setattr(chat_router, "get_provider", lambda *a, **k: llm)
    events = await _events(client, {"message": "Delivery policy?", "provider": "ollama"})
    assert events[-1]["generation_token_usage"] == {"prompt_tokens": 900, "completion_tokens": 40}


@pytest.mark.asyncio
async def test_openai_stream_records_the_usage_chunk(monkeypatch):
    """OpenAI sends usage in a final chunk with no choices; the text chunks come before it."""
    import openai

    def chunk(text=None, usage=None):
        delta = SimpleNamespace(content=text)
        return SimpleNamespace(choices=[SimpleNamespace(delta=delta)] if text else [], usage=usage)

    async def stream():
        yield chunk("Bon")
        yield chunk("jour")
        yield chunk(usage=SimpleNamespace(prompt_tokens=700, completion_tokens=12))

    requested = {}

    class FakeCompletions:
        async def create(self, **kwargs):
            requested.update(kwargs)
            return stream()

    class FakeClient:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeClient)
    provider = OpenAIProvider(api_key="test-key")  # a placeholder: nothing leaves the process
    pieces = [
        p async for p in provider.stream_answer("Q?", [{"source": "s.md", "text": "t"}], "fr")
    ]
    assert "".join(pieces) == "Bonjour"
    assert requested["stream_options"] == {"include_usage": True}
    assert provider.last_usage == {"prompt_tokens": 700, "completion_tokens": 12}


def test_token_cost_prices_measured_tokens_at_the_configured_rate():
    price = {"input": 0.15, "output": 0.60}  # the shape of configs/pipeline.yaml
    assert token_cost({"prompt_tokens": 1_000_000, "completion_tokens": 0}, price) == 0.15
    assert token_cost({"prompt_tokens": 1000, "completion_tokens": 100}, price) == pytest.approx(
        0.00021
    )
    assert token_cost(None, price) == 0.0  # a call that never reached the model
