"""
/chat/stream: the same guardrails and evidence as /chat, delivered as NDJSON events.
Language models are replaced by labelled stand-ins; the policy text comes from the mock
index in conftest.py (synthetic, not the real corpus).
"""

from __future__ import annotations

import json

import pytest

import api.routers.chat as chat_router
from src.rag.providers import AssistantResult, answer_messages, ollama_piece


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


def test_ollama_stream_lines_are_parsed():
    assert ollama_piece('{"message": {"content": "Bon"}, "done": false}') == ("Bon", False)
    assert ollama_piece('{"done": true}') == ("", True)
