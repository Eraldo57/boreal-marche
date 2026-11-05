"""One grounded orchestration path for regex, OpenAI and Ollama providers.

POST /chat         the complete answer at once (policy answers quote the source text)
POST /chat/stream  the same answer as newline-delimited JSON events, word by word. For a
                   policy question with OpenAI or Ollama, the model writes the answer from
                   the cited excerpts, in the requested language; if it fails, the stream
                   falls back to the source text. Events: meta, token..., [fallback], done.

Guardrails, retrieval and the typed tools are shared, so both endpoints refuse the same
questions and cite the same evidence. Scores, recommendations and product data always come
from the tools as templates: no model writes a probability or a product identifier.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from api.schemas import ChatRequest, ChatResponse, Citation, ToolCall
from src.ingest import load_config
from src.rag.guardrails import (
    HANDOFF_NOTICE,
    UNCERTAIN_NOTICE,
    check_no_cross_customer,
    check_out_of_scope,
    check_require_citations,
    detect_language,
)
from src.rag.providers import AssistantProvider, RegexProvider, get_provider
from src.rag.tools import TOOL_REGISTRY

router = APIRouter(tags=["assistant"])
GENERATION_EXCERPTS = 3  # policy excerpts an LLM may write from (all of them are cited)


def _check_guardrails(message: str, customer_id: str | None) -> tuple[bool, str | None]:
    for outcome in [
        check_out_of_scope(message),
        check_no_cross_customer(message, current_customer_id=customer_id),
    ]:
        if not outcome.passed:
            return True, outcome.reason
    return False, None


_detect_language = detect_language


def _detect_tool_intent(message: str) -> str | None:
    """Synchronous compatibility helper for unit tests; same provider logic."""
    return asyncio.run(RegexProvider().answer(message, [], [], "en")).tool_name


def _retrieve_chunks(
    index: Any, query: str, top_k: int = 5, floor: float = 0.25
) -> list[dict[str, Any]]:
    if index is None:
        return []
    return [r for r in index.query(query, top_k=top_k) if r["score"] >= floor]


def _format_context(chunks: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{i}] (source: {c['source']}) {c['text']}" for i, c in enumerate(chunks, 1))


def _generate_regex(message: str, context: str, chunks: list[dict[str, Any]], language: str) -> str:
    """Extractive answer: source wording is preserved, never presented as translation."""
    if not chunks:
        return (
            "Je n'ai pas trouvé de source pertinente."
            if language == "fr"
            else "I couldn't find relevant information."
        )
    intro = (
        "D'après nos documents (extrait dans la langue source) : "
        if language == "fr"
        else "Based on our documents: "
    )
    return intro + str(chunks[0]["text"])


@dataclass
class _Outcome:
    """The finished response, plus what a language model may rewrite when streaming."""

    response: ChatResponse
    question: str = ""
    excerpts: list[dict[str, Any]] = field(default_factory=list)
    provider: AssistantProvider | None = None


async def _orchestrate(req: ChatRequest, request: Request) -> _Outcome:
    language, provider_name = req.language, req.provider.value

    def refusal(reason: str) -> _Outcome:
        return _Outcome(
            ChatResponse(
                reply=reason,
                provider_used=provider_name,
                refused=True,
                refusal_reason=reason,
                language=language,
            )
        )

    denied, reason = _check_guardrails(req.message, req.customer_id)
    if denied:
        return refusal(reason or "")
    cfg = load_config()["assistant"]
    index = request.app.state.rag_index
    chunks = await asyncio.to_thread(
        _retrieve_chunks, index, req.message, cfg["max_chunks_per_query"], cfg["similarity_floor"]
    )
    if provider_name == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            503, "OpenAI provider requires OPENAI_API_KEY; regex is available locally"
        )
    provider = get_provider(
        provider_name,
        openai_model=cfg["openai"]["model"],
        ollama_model=os.getenv("OLLAMA_MODEL", cfg["ollama"]["model"]),
        ollama_host=os.getenv("OLLAMA_HOST", cfg["ollama"]["host"]),
    )
    start = time.perf_counter()
    try:
        decision = await provider.answer(
            req.message, [c["text"] for c in chunks], TOOL_REGISTRY, language
        )
    except Exception as exc:
        raise HTTPException(
            503,
            f"{provider_name} provider unavailable ({type(exc).__name__}); verify provider setup",
        ) from exc
    latency = (time.perf_counter() - start) * 1000
    registry = {t.name: t for t in TOOL_REGISTRY}
    tool = registry.get(decision.tool_name or "")
    if tool is None:
        return refusal(
            "No supported, grounded answer was found. / Aucune réponse fondée disponible."
        )
    arguments = dict(decision.tool_args)
    if tool.name == "retention_score":
        cid = arguments.get("customer_id") or req.customer_id
        # Request identity is demonstration context, NOT authenticated identity.
        if (
            not req.customer_id
            or not cid
            or cid.removeprefix("BM-") != req.customer_id.removeprefix("BM-")
        ):
            return refusal(
                "Provide your matching demo customer ID; another customer's score cannot "
                "be requested. / Indiquez votre identifiant de démonstration correspondant."
            )
        arguments["customer_id"] = cid.removeprefix("BM-")
    if tool.name == "policy_search":
        if not chunks:
            return refusal(HANDOFF_NOTICE[language])
        arguments["question"] = req.message
    try:
        validated = tool.input_model.model_validate(arguments)
        output = await tool.execute(validated, app=request.app)
    except (ValidationError, RuntimeError) as exc:
        return refusal(
            f"Tool request could not be validated: {type(exc).__name__}. / Requête non validée."
        )
    data = output.model_dump()
    if data.get("error"):
        return refusal(
            "The requested data is unavailable; no score or result was invented. / "
            "Données indisponibles."
        )
    calls = [ToolCall(tool_name=tool.name, arguments=arguments, result=data)]
    citations = []
    grounded: list[dict[str, Any]] = []
    uncertain = False
    if tool.name == "policy_search":
        grounded = [c for c in data["chunks"] if c["score"] >= cfg["similarity_floor"]]
        if not grounded:
            return refusal(HANDOFF_NOTICE[language])
        uncertain = grounded[0]["score"] < cfg["confident_score"]
        reply = _generate_regex(req.message, _format_context(grounded), grounded, language)
        if uncertain:
            reply = UNCERTAIN_NOTICE["fr" if language == "fr" else "en"] + reply
        citations = [
            Citation(
                source=grounded[0]["source"], chunk=grounded[0]["text"], score=grounded[0]["score"]
            )
        ]
        if not check_require_citations(reply, [c.model_dump() for c in citations]).passed:
            return refusal("Missing policy citation. / Citation absente.")
    elif tool.name == "retention_score":
        reply = (
            f"Probabilité d'achat dans les 90 jours : {data['probability']:.1%}. "
            f"Risque de non-retour : {data['risk_label']}."
            if language == "fr"
            else f"Probability of purchase within 90 days: {data['probability']:.1%}. "
            f"Non-return risk: {data['risk_label']}."
        )
        reply += " Historical public-data demo / démonstration historique."
    elif tool.name == "recommendations":
        reply = (
            "Suggestions issues des transactions UCI : "
            if language == "fr"
            else "Suggestions from UCI transaction history: "
        ) + ", ".join(data["recommendations"])
        reply += f" ({data['strategy']})"
    else:
        fr = language == "fr"
        reply = (
            "Catalogue Boréal Marché (export des achats UCI, prix unitaire médian en GBP) : "
            if fr
            else "Boréal Marché catalogue (export of UCI purchases, median unit price in GBP): "
        )
        if data["corrections"]:
            pairs = ", ".join(f"'{k}' → '{v}'" for k, v in data["corrections"].items())
            reply += f"(interprété : {pairs}) " if fr else f"(read as: {pairs}) "
        if data["products"] and not data["all_terms_matched"]:
            words = ", ".join(data["search_terms"])
            reply += (
                f"aucun produit ne contient tous ces mots ({words}) ; articles les plus proches : "
                if fr
                else f"no product matches all of ({words}); closest items: "
            )
        reply += "; ".join(
            f"{p['stock_code']} — {p['description']}: £{p['median_unit_price_gbp']:.2f}"
            for p in data["products"]
        )
        if not data["products"]:
            reply += "aucun produit correspondant." if fr else "no matching products."
        citations = [
            Citation(
                source="artifacts/product_catalogue.csv",
                chunk="Exported from cleaned UCI Online Retail II purchases; one row per StockCode",
                score=1.0,
            )
        ]
    response = ChatResponse(
        reply=reply,
        citations=citations,
        tool_calls=calls,
        provider_used=provider_name,
        passage_confidence=(("uncertain" if uncertain else "confident") if grounded else None),
        language=language,
        provider_latency_ms=latency,
        token_usage=decision.token_usage,
        # Local providers have no per-question charge; OpenAI's depends on a price list that
        # this repository does not store, so it stays unknown rather than estimated.
        cost_usd=0.0 if provider_name in {"regex", "ollama"} else None,
    )
    return _Outcome(response, req.message, grounded[:GENERATION_EXCERPTS], provider)


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    return (await _orchestrate(req, request)).response


def _line(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


# A figure with its unit ("30 days", "48 h", "15 %", "$5"): what a policy answer must not alter.
_FIGURE = re.compile(
    r"(?:[$£]\s?)?(\d+(?:[.,]\d+)?)\s*(?:days?|jours?|hours?|heures?|h\b|%|\$|CAD|GBP|£)",
    re.IGNORECASE,
)


def figure_check(answer: str, excerpts: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Figures a written answer adds that no cited excerpt states, and figures the top
    excerpt states that the answer leaves out. A signal for the reader, not a proof."""
    stated = set(_FIGURE.findall(answer))
    sources = set(_FIGURE.findall(" ".join(e["text"] for e in excerpts)))
    top = set(_FIGURE.findall(excerpts[0]["text"])) if excerpts else set()
    return {
        "figures_not_in_sources": sorted(stated - sources),
        "top_source_figures_missing": sorted(top - stated),
    }


def _words(text: str) -> list[str]:
    return re.findall(r"\S+\s*", text)


async def _events(outcome: _Outcome) -> AsyncIterator[str]:
    response = outcome.response
    tool = response.tool_calls[0].tool_name if response.tool_calls else None
    yield _line({"type": "meta", "provider": response.provider_used, "tool": tool})
    reply, written_by, citations = response.reply, "template", response.citations
    if tool == "policy_search":
        written_by = "source text"
    stream_answer = getattr(outcome.provider, "stream_answer", None)
    generated = False
    if outcome.excerpts and stream_answer is not None:
        pieces: list[str] = []
        try:
            async for piece in stream_answer(outcome.question, outcome.excerpts, response.language):
                pieces.append(piece)
                yield _line({"type": "token", "text": piece})
            if not "".join(pieces).strip():
                raise ValueError("the model returned no text")
            reply, written_by, generated = "".join(pieces).strip(), response.provider_used, True
            citations = [
                Citation(source=c["source"], chunk=c["text"], score=c["score"])
                for c in outcome.excerpts
            ]
        except Exception as exc:  # any provider problem: show the grounded source text instead
            yield _line(
                {
                    "type": "fallback",
                    "reason": f"{type(exc).__name__}: the model could not write the answer; "
                    "showing the source text",
                }
            )
            if pieces:
                yield _line({"type": "token", "text": "\n\n"})
    if not generated:
        for word in _words(reply):
            yield _line({"type": "token", "text": word})
    final = response.model_dump(mode="json")
    final.update(
        type="done",
        reply=reply,
        answer_written_by=written_by,
        citations=[c.model_dump() for c in citations],
    )
    if generated:
        final["figure_check"] = figure_check(reply, outcome.excerpts)
        # Tokens the model spent writing this answer, when the provider reports them; the
        # routing call's tokens are already in token_usage.
        final["generation_token_usage"] = getattr(outcome.provider, "last_usage", None)
    yield _line(final)


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
    """The /chat answer as newline-delimited JSON events, for word-by-word display."""
    outcome = await _orchestrate(req, request)
    return StreamingResponse(_events(outcome), media_type="application/x-ndjson")
