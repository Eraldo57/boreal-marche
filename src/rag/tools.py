"""
Typed tools for the Boréal Marché assistant.

Each tool has:
  - A Pydantic *Input* model  — validates what the provider sends.
  - A Pydantic *Output* model — validates what the tool returns.
  - An async ``execute`` function that does the work.

Tools that need predictions or recommendations call the **local API** via httpx.
The language model NEVER produces a probability or a product identifier from its
own weights.
"""

from __future__ import annotations

import asyncio
import csv
import difflib
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from src.contracts import ARTIFACTS
from src.rag.index import query_index

# The API base URL for internal tool calls (same process or Docker Compose).
API_BASE = "http://localhost:8000"


class ToolModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


def api_client(app=None):
    """In-process HTTP follows the same strict routes without a loopback deadlock."""
    return httpx.AsyncClient(
        base_url=os.getenv("API_BASE_URL", API_BASE),
        timeout=30,
        transport=httpx.ASGITransport(app=app) if app else None,
    )


# ============================================================================
# 1. policy_search — retrieves relevant policy chunks
# ============================================================================


class PolicySearchInput(ToolModel):
    """Input: a natural-language question about Boréal Marché policies."""

    question: str = Field(..., min_length=1, max_length=1000)
    n_results: int = Field(default=5, ge=1, le=10)


class PolicyChunk(BaseModel):
    """One retrieved chunk with its provenance."""

    text: str
    source: str
    score: float
    language: str


class PolicySearchOutput(BaseModel):
    """Output: the top matching policy chunks."""

    chunks: list[PolicyChunk]
    query: str


async def policy_search(inp: PolicySearchInput, *, app=None) -> PolicySearchOutput:
    """Retrieve policy chunks semantically similar to *inp.question*."""
    if app is not None:
        if app.state.rag_index is None:
            raise RuntimeError("RAG index not loaded")
        hits = await asyncio.to_thread(app.state.rag_index.query, inp.question, inp.n_results)
    else:
        hits = await asyncio.to_thread(query_index, inp.question, n_results=inp.n_results)
    chunks = [
        PolicyChunk(
            text=h["text"],
            source=h["source"],
            score=h["score"],
            language=h.get("language", "en"),
        )
        for h in hits
    ]
    return PolicySearchOutput(chunks=chunks, query=inp.question)


# ============================================================================
# 2. retention_score — calls /predict to get a customer's retention probability
# ============================================================================


class RetentionScoreInput(ToolModel):
    """Input: the customer whose retention probability we want."""

    customer_id: str = Field(..., min_length=1, max_length=100)


class RetentionScoreOutput(BaseModel):
    """Output: retention probability and risk label from Head A."""

    customer_id: str
    probability: float | None = None
    will_return: bool | None = None
    risk_label: str
    threshold: float | None = None
    error: str | None = None


async def retention_score(inp: RetentionScoreInput, *, app=None) -> RetentionScoreOutput:
    """Call the /predict endpoint and return the retention score.

    This function sends a request to the local API.  The features are looked
    up server-side; the assistant only needs the customer id.
    """
    try:
        async with api_client(app) as client:
            features = await client.get(f"/customers/{inp.customer_id}/features")
            features.raise_for_status()
            resp = await client.post(
                "/predict",
                json={
                    "customer_id": inp.customer_id,
                    "features": features.json()["features"],
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                return RetentionScoreOutput(
                    customer_id=data["customer_id"],
                    probability=data["probability"],
                    will_return=data["will_return"],
                    risk_label=data["risk_label"],
                    threshold=data["threshold"],
                )
            else:
                return RetentionScoreOutput(
                    customer_id=inp.customer_id,
                    risk_label="unknown",
                    error=f"API returned {resp.status_code}: {resp.text[:200]}",
                )
    except httpx.HTTPError as exc:
        return RetentionScoreOutput(
            customer_id=inp.customer_id,
            risk_label="unknown",
            error=f"Could not reach the prediction API: {exc}",
        )


# ============================================================================
# 3. recommendations — calls /recommend for basket suggestions
# ============================================================================


class RecommendationsInput(ToolModel):
    """Input: the basket items the customer currently has."""

    basket: list[str] = Field(
        ...,
        min_length=0,
        description="Product descriptions currently in the basket",
    )
    top_k: int = Field(default=5, ge=1, le=20)


class RecommendationsOutput(BaseModel):
    """Output: recommended products and the rules that produced them."""

    recommendations: list[str]
    strategy: str
    rules: list[dict[str, Any]] = []
    error: str | None = None


async def recommendations(inp: RecommendationsInput, *, app=None) -> RecommendationsOutput:
    """Call the /recommend endpoint and return suggestions."""
    try:
        async with api_client(app) as client:
            resp = await client.post(
                "/recommend",
                json={"basket": inp.basket, "top_k": inp.top_k},
            )
            if resp.status_code == 200:
                data = resp.json()
                return RecommendationsOutput(
                    recommendations=data["recommendations"],
                    strategy=data["strategy"],
                    rules=[r for r in data.get("because", [])],
                )
            else:
                return RecommendationsOutput(
                    recommendations=[],
                    strategy="error",
                    error=f"API returned {resp.status_code}: {resp.text[:200]}",
                )
    except httpx.HTTPError as exc:
        return RecommendationsOutput(
            recommendations=[],
            strategy="error",
            error=f"Could not reach the recommendation API: {exc}",
        )


# ============================================================================
# 4. product_lookup — searches the product catalogue
# ============================================================================


class ProductLookupInput(ToolModel):
    """Input: a search term (product description words or StockCode)."""

    query: str = Field(..., min_length=1, max_length=200)
    max_results: int = Field(default=10, ge=1, le=50)


class ProductHit(BaseModel):
    stock_code: str
    description: str
    median_unit_price_gbp: float
    invoice_count: int
    first_seen: str
    last_seen: str


class ProductLookupOutput(BaseModel):
    """Output: matching products from the catalogue."""

    products: list[ProductHit]
    query: str
    total_matches: int
    search_terms: list[str]
    all_terms_matched: bool
    corrections: dict[str, str]


@dataclass(frozen=True)
class _Catalogue:
    rows: list[dict[str, str]]
    searchable: list[str]
    corpus: str
    vocabulary: list[str]


_CATALOGUE_PATH = ARTIFACTS / "product_catalogue.csv"
_CATALOGUE_CACHE: _Catalogue | None = None
_WORD = re.compile(r"\w+(?:-\w+)*")
# Measured on the real vocabulary: 0.8 fixes "haert"/"hagning"/"chese"; words under
# four letters stay exact so "red" never becomes "bed".
_FUZZY_CUTOFF = 0.8
_FUZZY_MIN_LENGTH = 4


def _load_catalogue() -> _Catalogue:
    """Read the catalogue exported from real purchases by src.catalogue (cached)."""
    global _CATALOGUE_CACHE
    if _CATALOGUE_CACHE is None:
        if not _CATALOGUE_PATH.exists():
            raise RuntimeError("Product catalogue missing; run python -m src.catalogue")
        with _CATALOGUE_PATH.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        searchable = [
            " ".join([r["description"], r["other_descriptions"], r["stock_code"]]).lower()
            for r in rows
        ]
        vocabulary = sorted({w for text in searchable for w in _WORD.findall(text)})
        _CATALOGUE_CACHE = _Catalogue(rows, searchable, "\n".join(searchable), vocabulary)
    return _CATALOGUE_CACHE


def _correct(term: str, catalogue: _Catalogue) -> str | None:
    """Closest catalogue word for a likely typo; None when the term needs no correction."""
    if term in catalogue.corpus or len(term) < _FUZZY_MIN_LENGTH:
        return None
    close = difflib.get_close_matches(term, catalogue.vocabulary, n=1, cutoff=_FUZZY_CUTOFF)
    return close[0] if close else None


def _rank(forms: list[str], catalogue: _Catalogue) -> tuple[int, list[dict[str, str]]]:
    """Products matching the most search forms, most purchased first."""
    scores = [sum(form in text for form in forms) for text in catalogue.searchable]
    best = max(scores, default=0)
    if best == 0:
        return 0, []
    return best, [row for row, score in zip(catalogue.rows, scores) if score == best]


async def product_lookup(inp: ProductLookupInput, *, app=None) -> ProductLookupOutput:
    """Search the exported catalogue by words in the description or by StockCode."""
    catalogue = _load_catalogue()
    query_lower = inp.query.lower()
    stop = {
        "do",
        "you",
        "sell",
        "carry",
        "have",
        "any",
        "what",
        "is",
        "the",
        "of",
        "a",
        "an",
        "me",
        "show",
        "tell",
        "about",
        "product",
        "products",
        "options",
        "des",
        "de",
        "du",
        "le",
        "la",
        "les",
        "vos",
        "vous",
        "avez",
        "produit",
        "produits",
        "quel",
        "prix",
    }
    terms = list(dict.fromkeys(t for t in _WORD.findall(query_lower) if t not in stop))
    corrections = {t: c for t in terms if (c := _correct(t, catalogue))}
    best, hits = _rank([corrections.get(t, t) for t in terms], catalogue)
    if corrections and best < len(terms):
        # A correction that yields no full match is a guess too far: keep the user's words.
        corrections = {}
        best, hits = _rank(terms, catalogue)

    products = [
        ProductHit(
            stock_code=p["stock_code"],
            description=p["description"],
            median_unit_price_gbp=float(p["median_unit_price_gbp"]),
            invoice_count=int(p["invoice_count"]),
            first_seen=p["first_seen"],
            last_seen=p["last_seen"],
        )
        for p in hits[: inp.max_results]
    ]
    return ProductLookupOutput(
        products=products,
        query=inp.query,
        total_matches=len(hits),
        search_terms=terms,
        all_terms_matched=bool(terms) and best == len(terms),
        corrections=corrections,
    )


# ============================================================================
# Registry — all tools in one place
# ============================================================================


@dataclass(frozen=True)
class ToolDef:
    """Metadata the providers use to describe a tool to the LLM."""

    name: str
    description: str
    input_model: type[BaseModel]
    execute: Any  # the async callable


TOOL_REGISTRY: list[ToolDef] = [
    ToolDef(
        name="policy_search",
        description=(
            "Search Boréal Marché policy documents (delivery, returns, allergens, "
            "loyalty, privacy) for information relevant to the customer's question."
        ),
        input_model=PolicySearchInput,
        execute=policy_search,
    ),
    ToolDef(
        name="retention_score",
        description=(
            "Look up a specific customer's retention probability and risk level. "
            "Requires a customer_id."
        ),
        input_model=RetentionScoreInput,
        execute=retention_score,
    ),
    ToolDef(
        name="recommendations",
        description=(
            "Given items currently in a customer's basket, suggest additional "
            "products they might like. Provide the basket contents as a list of "
            "product descriptions."
        ),
        input_model=RecommendationsInput,
        execute=recommendations,
    ),
    ToolDef(
        name="product_lookup",
        description=(
            "Search the Boréal Marché product catalogue (exported from real purchase "
            "history) by words in the product description or by StockCode, tolerating "
            "small typos. It has no categories, allergens or dietary tags."
        ),
        input_model=ProductLookupInput,
        execute=product_lookup,
    ),
]
