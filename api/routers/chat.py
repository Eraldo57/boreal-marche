"""One grounded orchestration path for regex, OpenAI and Ollama providers."""

from __future__ import annotations

import asyncio
import os
import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from api.schemas import ChatRequest, ChatResponse, Citation, ToolCall
from src.ingest import load_config
from src.rag.guardrails import (
    check_no_cross_customer,
    check_out_of_scope,
    check_require_citations,
    detect_language,
)
from src.rag.providers import RegexProvider, get_provider
from src.rag.tools import TOOL_REGISTRY

router = APIRouter(tags=["assistant"])


def _check_guardrails(message, customer_id):
    for outcome in [
        check_out_of_scope(message),
        check_no_cross_customer(message, current_customer_id=customer_id),
    ]:
        if not outcome.passed:
            return True, outcome.reason
    return False, None


_detect_language = detect_language


def _detect_tool_intent(message):
    """Synchronous compatibility helper for unit tests; same provider logic."""
    return asyncio.run(RegexProvider().answer(message, [], [], "en")).tool_name


def _retrieve_chunks(index, query, top_k=5, floor=0.25):
    if index is None:
        return []
    return [r for r in index.query(query, top_k=top_k) if r["score"] >= floor]


def _format_context(chunks):
    return "\n".join(f"[{i}] (source: {c['source']}) {c['text']}" for i, c in enumerate(chunks, 1))


def _generate_regex(message, context, chunks, language):
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
    return intro + chunks[0]["text"]


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    language, provider_name = req.language, req.provider.value

    def refusal(reason):
        return ChatResponse(
            reply=reason,
            provider_used=provider_name,
            refused=True,
            refusal_reason=reason,
            language=language,
        )

    denied, reason = _check_guardrails(req.message, req.customer_id)
    if denied:
        return refusal(reason)
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
        ollama_model=cfg["ollama"]["model"],
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
    tool = registry.get(decision.tool_name)
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
            return refusal(
                "No policy source meets the relevance threshold. / "
                "Aucune source suffisamment pertinente."
            )
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
    if tool.name == "policy_search":
        grounded = [c for c in data["chunks"] if c["score"] >= cfg["similarity_floor"]]
        if not grounded:
            return refusal("No relevant policy evidence found. / Aucune source pertinente.")
        reply = _generate_regex(req.message, _format_context(grounded), grounded, language)
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
    return ChatResponse(
        reply=reply,
        citations=citations,
        tool_calls=calls,
        provider_used=provider_name,
        language=language,
        provider_latency_ms=latency,
        token_usage=decision.token_usage,
        cost_usd=0.0 if provider_name == "regex" else None,
    )
