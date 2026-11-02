"""
Three interchangeable providers for the Boréal Marché assistant.

Every provider implements :class:`AssistantProvider`:

    async def answer(message, context_chunks, tools, language) -> AssistantResult

The provider's ONLY job is to choose which tool to call (if any) and extract
the arguments.  It must NEVER produce a retention probability, a product
identifier, or a recommendation from its own weights — those come from the
tools, which call the real models behind the API.

Providers
---------
1. **RegexProvider** — rule/keyword baseline.  Zero external dependencies.
2. **OpenAIProvider** — OpenAI chat completions with JSON mode.
3. **OllamaProvider** — local Ollama server, same JSON approach.
"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from src.rag.tools import ToolDef

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------


@dataclass
class AssistantResult:
    """Intermediate result from a provider before the orchestrator formats it."""

    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""  # free-form text the provider wants shown if no tool fires
    language: str = "en"
    token_usage: dict[str, int] | None = None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class AssistantProvider(ABC):
    """Interface every provider must implement."""

    @abstractmethod
    async def answer(
        self,
        message: str,
        context_chunks: list[str],
        tools: list[ToolDef],
        language: str,
    ) -> AssistantResult:
        """Decide which tool to call and extract arguments.

        Parameters
        ----------
        message : str
            The customer's question.
        context_chunks : list[str]
            Retrieved policy chunks (may be empty).
        tools : list[ToolDef]
            Available tool definitions.
        language : str
            Detected language of *message* (``"en"`` or ``"fr"``).

        Returns
        -------
        AssistantResult
            Must set *tool_name* and *tool_args* when a tool should fire,
            or *raw_text* for a direct answer assembled from context chunks.
        """


# ============================================================================
# 1. RegexProvider — no LLM, pure pattern matching
# ============================================================================


class RegexProvider(AssistantProvider):
    """Rule-based baseline that pattern-matches the user's question.

    This is deliberately simple.  It demonstrates how far you can get without
    a language model and gives the evaluation a fair "no-LLM" data-point.
    """

    # --- Patterns that route to a tool ---------------------------------- #

    _RETENTION_PAT = re.compile(
        r"(?:retention|churn|rétention|risk|risque|score|probability|"
        r"probabilit|will\s+(?:they|this customer)\s+(?:come back|return|buy)|"
        r"va-t-il\s+revenir|chance|likelihood|will\s+customer|will\s+return)",
        re.IGNORECASE,
    )
    _CUSTOMER_ID_PAT = re.compile(
        r"(?:customer|client|compte|account)?\s*(?:#|id|num[ée]ro)?\s*"
        r"[:\s]*([A-Z]{0,3}-?\d{3,8})\b",
        re.IGNORECASE,
    )

    _RECOMMEND_PAT = re.compile(
        r"(?:recommend|suggest|pair|goes?\s+with|complément|proposer|sugg[ée]r|"
        r"what\s+(?:else|should|could)|que\s+(?:suggér|propos|ajouter)|"
        r"aussi\s+aimer|you\s+might\s+(?:also\s+)?like|"
        r"based\s+on\s+(?:my\s+)?(?:basket|cart|panier))",
        re.IGNORECASE,
    )
    _BASKET_ITEM_PAT = re.compile(
        r"(?:basket|cart|panier)\s*(?:contains?|has|avec|contient)?\s*[:\s]*(.*)",
        re.IGNORECASE,
    )

    _PRODUCT_PAT = re.compile(
        r"(?:product|produit|item|article|sku|price|prix|"
        r"do\s+you\s+(?:sell|carry|have)|avez-vous|vendez|cherche|"
        r"looking\s+for|find|trouver|catalogue|catégorie|category|"
        r"gluten.?free|vegan|organic|bio|halal|kosher|casher|"
        r"frozen|congel[ée]|dairy|lait|meat|viande|seafood|fruit|"
        r"bakery|boulangerie|beverage|boisson|household|ménag)",
        re.IGNORECASE,
    )

    async def answer(
        self,
        message: str,
        context_chunks: list[str],
        tools: list[ToolDef],
        language: str,
    ) -> AssistantResult:
        result = AssistantResult(language=language)

        # --- 1. Retention score ----------------------------------------- #
        if self._RETENTION_PAT.search(message):
            cid_match = self._CUSTOMER_ID_PAT.search(message)
            result.tool_name = "retention_score"
            result.tool_args = {"customer_id": cid_match.group(1)} if cid_match else {}
            return result

        # --- 2. Recommendations ----------------------------------------- #
        if self._RECOMMEND_PAT.search(message):
            basket_items: list[str] = []
            bm = re.search(r"(?:have|got)\s+(.+?)\s+in\s+my\s+(?:basket|cart)", message, re.I)
            if bm is None:
                bm = re.search(
                    r"(?:basket|cart|panier)\s*(?:contains|has|with|avec|contient)\s+([^.!?,]+)",
                    message,
                    re.I,
                )
            if bm is None:
                bm = re.search(r"(?:go\s+with|avec)\s+(?:a\s+)?([^.!?]+)", message, re.I)
            if bm:
                raw = bm.group(1)
                basket_items = [
                    item.strip() for item in re.split(r"[,;]|\band\b|\bet\b", raw) if item.strip()
                ]
            if not basket_items:
                # Try to extract quoted items.
                basket_items = re.findall(r'"([^"]+)"', message)
            if not basket_items:
                # Fall back: grab nouns after common verbs.
                basket_items = re.findall(
                    r"(?:have|got|bought|acheté|ajouté|added)\s+(.+?)(?:[.,;]|$)",
                    message,
                    re.IGNORECASE,
                )
            result.tool_name = "recommendations"
            result.tool_args = {"basket": basket_items}
            return result

        # Policy terms take precedence over generic words such as products/items.
        if re.search(
            r"polic|politiq|deliver|livraison|ship|return|retour|refund|rembours|loyalty|"
            r"fidélité|point|allergen|allergène|privacy|personal data|"
            r"delete.*(?:account|data)|supprimer|cancel|cold.chain",
            message,
            re.I,
        ):
            return AssistantResult(
                tool_name="policy_search", tool_args={"question": message}, language=language
            )

        # --- 3. Product lookup ------------------------------------------ #
        if self._PRODUCT_PAT.search(message):
            # Use the whole message as the search query — the product_lookup
            # tool does keyword matching internally.
            result.tool_name = "product_lookup"
            result.tool_args = {"query": message}
            return result

        # --- 4. Policy search (default) --------------------------------- #
        if context_chunks:
            result.tool_name = "policy_search"
            result.tool_args = {"question": message}
        else:
            result.raw_text = message  # nothing matched; let guardrails handle it
        return result


# ============================================================================
# 2. OpenAIProvider — uses OpenAI chat completions with JSON mode
# ============================================================================

_SYSTEM_PROMPT = """\
You are the Boréal Marché customer assistant's decision layer.
Your ONLY job is to decide which tool to call and extract its arguments.
You NEVER answer the customer directly. You NEVER invent data.

Available tools:
{tool_descriptions}

Given the customer's message and the retrieved policy context, return EXACTLY
one JSON object:
{{
  "tool": "<tool_name or null>",
  "args": {{ ... }}     // must match the tool's input schema
}}

If no tool is appropriate, return:
{{ "tool": null, "args": {{}}, "direct_answer": "<answer using ONLY the context>" }}

Rules:
- NEVER guess a probability, a price, or a product ID. Use the tools.
- If the customer asks about retention or churn, use "retention_score".
- If the customer asks for product suggestions, use "recommendations".
- If the customer asks about a specific product, use "product_lookup".
- For policy questions, use "policy_search".
- Respond in {language}.
"""


_ANSWER_PROMPT = """\
You are Boréal Marché's customer assistant. Answer the customer's question using ONLY the
policy excerpts provided. If they do not contain the answer, say that you cannot answer
from the policies and suggest contacting customer service. Never add facts, prices,
product codes or probabilities that are not in the excerpts. Answer in {language}, in at
most four sentences, and name the source document you relied on.
"""


def answer_messages(
    question: str, excerpts: list[dict[str, Any]], language: str
) -> list[dict[str, str]]:
    """Chat messages for writing a grounded answer from retrieved policy excerpts."""
    context = "\n\n".join(f"[{i}] ({e['source']}) {e['text']}" for i, e in enumerate(excerpts, 1))
    return [
        {
            "role": "system",
            "content": _ANSWER_PROMPT.format(language="French" if language == "fr" else "English"),
        },
        {
            "role": "user",
            "content": f"Policy excerpts:\n{context}\n\nCustomer question: {question}",
        },
    ]


def ollama_piece(line: str) -> tuple[str, bool]:
    """Text and done flag from one line of Ollama's streamed /api/chat response."""
    data = json.loads(line)
    return data.get("message", {}).get("content", ""), bool(data.get("done"))


def _build_tool_desc(tools: list[ToolDef]) -> str:
    """Format tool descriptions for the system prompt."""
    lines: list[str] = []
    for t in tools:
        schema = t.input_model.model_json_schema()
        props = schema.get("properties", {})
        args_desc = ", ".join(f"{k}: {v.get('type', '?')}" for k, v in props.items())
        lines.append(f"- {t.name}({args_desc}): {t.description}")
    return "\n".join(lines)


class OpenAIProvider(AssistantProvider):
    """Uses OpenAI chat completions with JSON mode for tool dispatch."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

    async def answer(
        self,
        message: str,
        context_chunks: list[str],
        tools: list[ToolDef],
        language: str,
    ) -> AssistantResult:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key)
        system = _SYSTEM_PROMPT.format(
            tool_descriptions=_build_tool_desc(tools),
            language="French" if language == "fr" else "English",
        )

        context_text = "\n---\n".join(context_chunks) if context_chunks else "(no context)"
        user_content = f"Customer message: {message}\n\nRetrieved policy context:\n{context_text}"

        resp = await client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        )

        raw = resp.choices[0].message.content or "{}"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("OpenAI returned invalid JSON: %s", raw[:200])
            return AssistantResult(raw_text=raw, language=language)

        result = AssistantResult(
            tool_name=parsed.get("tool"),
            tool_args=parsed.get("args", {}),
            raw_text=parsed.get("direct_answer", ""),
            language=language,
            token_usage={
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            }
            if resp.usage
            else None,
        )
        # Normalise null/"null"/None.
        if result.tool_name in (None, "null", ""):
            result.tool_name = None
        return result

    async def stream_answer(
        self, question: str, excerpts: list[dict[str, Any]], language: str
    ) -> AsyncIterator[str]:
        """Write a grounded policy answer token by token (used only by /chat/stream)."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key)
        stream: Any = await client.chat.completions.create(  # an AsyncStream when stream=True
            model=self.model,
            temperature=0,
            stream=True,
            messages=answer_messages(question, excerpts, language),  # type: ignore[arg-type]
        )
        async for chunk in stream:
            piece = chunk.choices[0].delta.content if chunk.choices else None
            if piece:
                yield piece


# ============================================================================
# 3. OllamaProvider — same approach, local model
# ============================================================================


class OllamaProvider(AssistantProvider):
    """Uses a local Ollama instance for tool dispatch."""

    def __init__(
        self,
        model: str = "llama3.2:3b",
        host: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")

    async def answer(
        self,
        message: str,
        context_chunks: list[str],
        tools: list[ToolDef],
        language: str,
    ) -> AssistantResult:
        import httpx

        system = _SYSTEM_PROMPT.format(
            tool_descriptions=_build_tool_desc(tools),
            language="French" if language == "fr" else "English",
        )
        context_text = "\n---\n".join(context_chunks) if context_chunks else "(no context)"
        user_content = f"Customer message: {message}\n\nRetrieved policy context:\n{context_text}"

        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_content},
                    ],
                },
            )
            resp.raise_for_status()
            raw = resp.json()["message"]["content"]

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Ollama returned invalid JSON: %s", raw[:200])
            return AssistantResult(raw_text=raw, language=language)

        result = AssistantResult(
            tool_name=parsed.get("tool"),
            tool_args=parsed.get("args", {}),
            raw_text=parsed.get("direct_answer", ""),
            language=language,
        )
        if result.tool_name in (None, "null", ""):
            result.tool_name = None
        return result

    async def stream_answer(
        self, question: str, excerpts: list[dict[str, Any]], language: str
    ) -> AsyncIterator[str]:
        """Write a grounded policy answer token by token (used only by /chat/stream)."""
        import httpx

        body = {
            "model": self.model,
            "stream": True,
            "options": {"temperature": 0},
            "messages": answer_messages(question, excerpts, language),
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", f"{self.host}/api/chat", json=body) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    piece, done = ollama_piece(line)
                    if piece:
                        yield piece
                    if done:
                        break


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_provider(
    name: str,
    *,
    openai_model: str = "gpt-4o-mini",
    openai_api_key: str | None = None,
    ollama_model: str = "llama3.2:3b",
    ollama_host: str = "http://localhost:11434",
) -> AssistantProvider:
    """Return the provider instance for *name* (``regex``, ``openai``, ``ollama``)."""
    name = name.lower().strip()
    if name == "regex":
        return RegexProvider()
    if name == "openai":
        return OpenAIProvider(model=openai_model, api_key=openai_api_key)
    if name == "ollama":
        return OllamaProvider(model=ollama_model, host=ollama_host)
    raise ValueError(f"Unknown provider: {name!r}. Choose regex, openai, or ollama.")
