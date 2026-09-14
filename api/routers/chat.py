"""
Head C — Grounded bilingual assistant endpoint.

POST /chat  Ask a question; the assistant answers using retrieval,
            tool calls, and guardrails.
"""

from __future__ import annotations

import re
from typing import Any

import structlog
from fastapi import APIRouter, Request

from api.schemas import ChatRequest, ChatResponse, Citation, Provider, ToolCall

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["assistant"])

# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------
_OUT_OF_SCOPE_PATTERNS = re.compile(
    r"(stock\s*price|invest|lottery|medical\s*diagnosis|lawsuit|"
    r"hack|exploit|password|credit\s*card\s*number)",
    re.IGNORECASE,
)

_CROSS_CUSTOMER_PATTERN = re.compile(
    r"(another\s*customer|other\s*customer'?s?|someone\s*else'?s?\s*(order|account|data))",
    re.IGNORECASE,
)


def _check_guardrails(
    message: str, customer_id: str | None
) -> tuple[bool, str | None]:
    """Return (refused, reason).  refused=False means the message is safe."""
    if _OUT_OF_SCOPE_PATTERNS.search(message):
        return True, (
            "This question is outside Boréal Marché's scope. "
            "I can help with delivery, returns, allergens, loyalty, and product questions. "
            "For anything else, please contact a human agent."
        )
    if _CROSS_CUSTOMER_PATTERN.search(message):
        return True, (
            "I cannot share information about another customer's account. "
            "I can only help with your own orders and account."
        )
    return False, None


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------
def _retrieve_chunks(
    rag_index: Any, query: str, top_k: int = 5, floor: float = 0.25
) -> list[dict[str, Any]]:
    """Query the RAG index and return chunks above the similarity floor."""
    if rag_index is None:
        return []

    try:
        results = rag_index.query(query, top_k=top_k)
    except Exception as exc:
        logger.warning("rag_query_failed", error=str(exc))
        return []

    return [r for r in results if r.get("score", 0) >= floor]


def _format_context(chunks: list[dict[str, Any]]) -> str:
    """Turn retrieved chunks into a prompt context block."""
    if not chunks:
        return ""
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] (source: {c.get('source', 'unknown')}) {c.get('text', '')}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------
def _detect_tool_intent(message: str) -> str | None:
    """Simple keyword-based tool detection (regex provider baseline)."""
    lower = message.lower()

    if any(w in lower for w in ("policy", "return", "delivery", "allergen",
                                 "privacy", "politique", "livraison",
                                 "retour", "allergène")):
        return "policy_search"

    if any(w in lower for w in ("retention", "churn", "will.*return",
                                 "rétention", "prédiction")):
        return "retention_score"

    if any(w in lower for w in ("recommend", "suggest", "also like",
                                 "recommand", "suggér")):
        return "recommendations"

    if any(w in lower for w in ("product", "item", "sku", "produit",
                                 "catalogue", "price", "prix")):
        return "product_lookup"

    return None


def _execute_tool(
    tool_name: str, message: str, request: Request, customer_id: str | None
) -> ToolCall:
    """Execute a tool and return a ToolCall record."""
    if tool_name == "retention_score" and customer_id:
        # Call our own /predict endpoint logic
        return ToolCall(
            tool_name="retention_score",
            arguments={"customer_id": customer_id},
            result={"note": "Retention score requires feature vector — not available in chat context."},
        )

    if tool_name == "recommendations":
        # Parse a basket from the message — very basic regex extraction
        return ToolCall(
            tool_name="recommendations",
            arguments={"message": message},
            result={"note": "Use /recommend endpoint with an explicit basket."},
        )

    if tool_name == "product_lookup":
        return ToolCall(
            tool_name="product_lookup",
            arguments={"query": message},
            result={"note": "Product catalogue lookup."},
        )

    # Default: policy_search
    return ToolCall(
        tool_name="policy_search",
        arguments={"query": message},
        result={"note": "Searching policy documents."},
    )


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------
def _generate_regex(
    message: str,
    context: str,
    chunks: list[dict[str, Any]],
    language: str,
) -> str:
    """Keyword-matching baseline — no LLM required."""
    if not context:
        if language == "fr":
            return "Je n'ai pas trouvé d'information pertinente dans nos documents. Pourriez-vous reformuler votre question ?"
        return "I couldn't find relevant information in our documents. Could you rephrase your question?"

    # Return the best chunk as the answer
    best = chunks[0] if chunks else {}
    text = best.get("text", context.split("\n")[0] if context else "")

    if language == "fr":
        return f"D'après nos documents : {text}"
    return f"Based on our documents: {text}"


def _generate_openai(
    message: str, context: str, language: str
) -> str:
    """OpenAI provider — calls the API."""
    try:
        from openai import OpenAI
        import os

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        lang_instruction = "Respond in French." if language == "fr" else "Respond in English."
        system_prompt = (
            f"You are a customer-support assistant for Boréal Marché, a Québec online grocery retailer. "
            f"Answer ONLY using the provided context. If the context does not contain the answer, say so. "
            f"Always cite your source. {lang_instruction}\n\nContext:\n{context}"
        )
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
        )
        return r.choices[0].message.content or ""
    except Exception as exc:
        logger.error("openai_call_failed", error=str(exc))
        return f"OpenAI provider error: {exc}"


def _generate_ollama(
    message: str, context: str, language: str
) -> str:
    """Ollama provider — calls a local LLM."""
    try:
        import httpx

        lang_instruction = "Respond in French." if language == "fr" else "Respond in English."
        system_prompt = (
            f"You are a customer-support assistant for Boréal Marché, a Québec online grocery retailer. "
            f"Answer ONLY using the provided context. If the context does not contain the answer, say so. "
            f"Always cite your source. {lang_instruction}\n\nContext:\n{context}"
        )
        host = "http://localhost:11434"
        r = httpx.post(
            f"{host}/api/chat",
            timeout=90,
            json={
                "model": "llama3.1",
                "stream": False,
                "options": {"temperature": 0},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message},
                ],
            },
        )
        r.raise_for_status()
        return r.json()["message"]["content"]
    except Exception as exc:
        logger.error("ollama_call_failed", error=str(exc))
        return f"Ollama provider error: {exc}"


def _detect_language(message: str) -> str:
    """Heuristic language detection — French vs English."""
    french_markers = {"je", "le", "la", "les", "de", "du", "des", "un", "une",
                      "est", "et", "en", "que", "qui", "pour", "dans", "sur",
                      "mon", "ma", "mes", "votre", "notre", "avec", "pas",
                      "cette", "quel", "quelle", "comment", "pourquoi", "quand"}
    words = set(re.findall(r"\b\w+\b", message.lower()))
    fr_count = len(words & french_markers)
    return "fr" if fr_count >= 2 else "en"


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------
@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    """Answer a customer question with grounded, cited information."""
    # Step 0 — detect language if client didn't specify explicitly
    language = req.language or _detect_language(req.message)

    # Step 1 — guardrails
    refused, reason = _check_guardrails(req.message, req.customer_id)
    if refused:
        return ChatResponse(
            reply=reason or "",
            provider_used=req.provider.value,
            refused=True,
            refusal_reason=reason,
            language=language,
        )

    # Step 2 — tool detection
    tool_calls: list[ToolCall] = []
    tool_name = _detect_tool_intent(req.message)
    if tool_name:
        tc = _execute_tool(tool_name, req.message, request, req.customer_id)
        tool_calls.append(tc)

    # Step 3 — retrieval
    rag_index = request.app.state.rag_index
    chunks = _retrieve_chunks(rag_index, req.message, top_k=5, floor=0.25)
    context = _format_context(chunks)

    # If no context and no tool calls, return a handoff message
    if not context and not tool_calls:
        no_info_msg = (
            "Je n'ai pas assez d'information pour répondre. Veuillez contacter notre service client."
            if language == "fr"
            else "I don't have enough information to answer. Please contact our customer service."
        )
        return ChatResponse(
            reply=no_info_msg,
            provider_used=req.provider.value,
            refused=False,
            language=language,
        )

    # Step 4 — generate answer
    if req.provider == Provider.OPENAI:
        reply = _generate_openai(req.message, context, language)
    elif req.provider == Provider.OLLAMA:
        reply = _generate_ollama(req.message, context, language)
    else:
        reply = _generate_regex(req.message, context, chunks, language)

    # Step 5 — build citations
    citations = [
        Citation(
            source=c.get("source", "unknown"),
            chunk=c.get("text", "")[:200],
            score=round(c.get("score", 0.0), 4),
        )
        for c in chunks
    ]

    logger.info(
        "chat_served",
        provider=req.provider.value,
        tool_calls=[tc.tool_name for tc in tool_calls],
        n_citations=len(citations),
        language=language,
        refused=False,
    )

    return ChatResponse(
        reply=reply,
        citations=citations,
        tool_calls=tool_calls,
        provider_used=req.provider.value,
        refused=False,
        language=language,
    )
