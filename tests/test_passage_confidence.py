"""A weakly matching passage is shown with a caution, not presented as the answer."""

from __future__ import annotations

import json

import pytest

from src.rag.guardrails import UNCERTAIN_NOTICE


class _Index:
    """A retrieval stand-in whose best passage scores `score`."""

    def __init__(self, score: float) -> None:
        self.score = score

    def query(self, text: str, top_k: int = 5):
        return [
            {
                "source": "delivery_policy.md",
                "text": "Delivery takes 2-5 days.",
                "score": self.score,
            }
        ]


async def _chat(client, score: float, language: str = "en"):
    client._transport.app.state.rag_index = _Index(score)
    response = await client.post(
        "/chat", json={"message": "What is the delivery policy?", "language": language}
    )
    assert response.status_code == 200
    return response.json()


@pytest.mark.asyncio
async def test_a_confident_passage_is_quoted_without_a_caution(client):
    data = await _chat(client, 0.60)
    assert data["passage_confidence"] == "confident" and not data["refused"]
    assert UNCERTAIN_NOTICE["en"] not in data["reply"]


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["en", "fr"])
async def test_a_weak_passage_comes_with_a_caution_and_the_handoff(client, language):
    data = await _chat(client, 0.35, language)
    assert data["passage_confidence"] == "uncertain" and not data["refused"]
    assert data["reply"].startswith(UNCERTAIN_NOTICE[language])
    assert "support@borealmarche.ca" in data["reply"]
    assert data["citations"], "the passage is still cited"


@pytest.mark.asyncio
async def test_below_the_floor_it_still_declines(client):
    data = await _chat(client, 0.20)
    assert data["refused"] is True and data["passage_confidence"] is None
    assert "support@borealmarche.ca" in data["reply"]


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["en", "fr"])
async def test_streamed_low_similarity_refusal_has_a_human_handoff(client, language):
    client._transport.app.state.rag_index = _Index(0.10)
    response = await client.post(
        "/chat/stream",
        json={
            "message": "What is the delivery policy?",
            "language": language,
        },
    )
    done = [json.loads(line) for line in response.text.splitlines()][-1]
    assert done["refused"]
    assert "support@borealmarche.ca" in done["reply"]
    assert "fictional" in done["reply"] if language == "en" else "fictives" in done["reply"]


@pytest.mark.asyncio
async def test_the_streamed_answer_carries_the_same_flag(client):
    client._transport.app.state.rag_index = _Index(0.35)
    async with client.stream(
        "POST", "/chat/stream", json={"message": "What is the delivery policy?"}
    ) as response:
        events = [json.loads(line) async for line in response.aiter_lines() if line]
    assert events[-1]["passage_confidence"] == "uncertain"
