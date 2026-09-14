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

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from src.rag.index import query_index

# The API base URL for internal tool calls (same process or Docker Compose).
API_BASE = "http://localhost:8000"


# ============================================================================
# 1. policy_search — retrieves relevant policy chunks
# ============================================================================

class PolicySearchInput(BaseModel):
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


async def policy_search(inp: PolicySearchInput) -> PolicySearchOutput:
    """Retrieve policy chunks semantically similar to *inp.question*."""
    hits = query_index(inp.question, n_results=inp.n_results)
    chunks = [
        PolicyChunk(
            text=h["text"],
            source=h["source"],
            score=h["score"],
            language=h["language"],
        )
        for h in hits
    ]
    return PolicySearchOutput(chunks=chunks, query=inp.question)


# ============================================================================
# 2. retention_score — calls /predict to get a customer's retention probability
# ============================================================================

class RetentionScoreInput(BaseModel):
    """Input: the customer whose retention probability we want."""
    customer_id: str = Field(..., min_length=1, max_length=100)


class RetentionScoreOutput(BaseModel):
    """Output: retention probability and risk label from Head A."""
    customer_id: str
    probability: float
    will_return: bool
    risk_label: str
    threshold: float
    error: str | None = None


async def retention_score(inp: RetentionScoreInput) -> RetentionScoreOutput:
    """Call the /predict endpoint and return the retention score.

    This function sends a request to the local API.  The features are looked
    up server-side; the assistant only needs the customer id.
    """
    try:
        async with httpx.AsyncClient(base_url=API_BASE, timeout=10.0) as client:
            resp = await client.post(
                "/predict",
                json={
                    "customer_id": inp.customer_id,
                    # An empty features dict tells the API to look up features
                    # from the feature store.  When the feature store is not
                    # available, the API returns a 422 and the error is relayed.
                    "features": {},
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
                    probability=0.0,
                    will_return=False,
                    risk_label="unknown",
                    threshold=0.5,
                    error=f"API returned {resp.status_code}: {resp.text[:200]}",
                )
    except httpx.HTTPError as exc:
        return RetentionScoreOutput(
            customer_id=inp.customer_id,
            probability=0.0,
            will_return=False,
            risk_label="unknown",
            threshold=0.5,
            error=f"Could not reach the prediction API: {exc}",
        )


# ============================================================================
# 3. recommendations — calls /recommend for basket suggestions
# ============================================================================

class RecommendationsInput(BaseModel):
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


async def recommendations(inp: RecommendationsInput) -> RecommendationsOutput:
    """Call the /recommend endpoint and return suggestions."""
    try:
        async with httpx.AsyncClient(base_url=API_BASE, timeout=10.0) as client:
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

class ProductLookupInput(BaseModel):
    """Input: a search term (product name, category, tag, or SKU)."""
    query: str = Field(..., min_length=1, max_length=200)
    max_results: int = Field(default=10, ge=1, le=50)


class ProductHit(BaseModel):
    sku: str
    name: str
    category: str
    price: float
    unit: str
    temperature: str
    allergens: str
    tags: str


class ProductLookupOutput(BaseModel):
    """Output: matching products from the catalogue."""
    products: list[ProductHit]
    query: str
    total_matches: int


# Module-level cache for the parsed catalogue.
_CATALOGUE_CACHE: list[dict[str, str]] | None = None


def _load_catalogue() -> list[dict[str, str]]:
    """Parse the product_catalogue.md into a list of dicts (cached)."""
    global _CATALOGUE_CACHE
    if _CATALOGUE_CACHE is not None:
        return _CATALOGUE_CACHE

    cat_path = Path(__file__).resolve().parents[2] / "docs" / "corpus" / "product_catalogue.md"
    text = cat_path.read_text(encoding="utf-8")

    products: list[dict[str, str]] = []
    # Match Markdown table rows that start with a SKU like BM-XX-NNN.
    row_pattern = re.compile(
        r"\|\s*(BM-\S+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|"
        r"\s*([\d.]+)\s*\|\s*(.+?)\s*\|\s*(\S+)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|"
    )
    for m in row_pattern.finditer(text):
        products.append({
            "sku": m.group(1).strip(),
            "name": m.group(2).strip(),
            "category": m.group(3).strip(),
            "sub_category": m.group(4).strip(),
            "price": m.group(5).strip(),
            "unit": m.group(6).strip(),
            "temperature": m.group(7).strip(),
            "allergens": m.group(8).strip(),
            "tags": m.group(9).strip(),
        })

    _CATALOGUE_CACHE = products
    return products


async def product_lookup(inp: ProductLookupInput) -> ProductLookupOutput:
    """Search the product catalogue by keyword in name, category, or tags."""
    catalogue = _load_catalogue()
    query_lower = inp.query.lower()
    terms = query_lower.split()

    scored: list[tuple[int, dict[str, str]]] = []
    for prod in catalogue:
        searchable = " ".join([
            prod["name"], prod["category"], prod["sub_category"],
            prod["allergens"], prod["tags"], prod["sku"],
        ]).lower()
        # Score = number of query terms that appear in the searchable text.
        score = sum(1 for t in terms if t in searchable)
        if score > 0:
            scored.append((score, prod))

    scored.sort(key=lambda x: x[0], reverse=True)
    total = len(scored)
    top = scored[: inp.max_results]

    products = [
        ProductHit(
            sku=p["sku"],
            name=p["name"],
            category=p["category"],
            price=float(p["price"]),
            unit=p["unit"],
            temperature=p["temperature"],
            allergens=p["allergens"],
            tags=p["tags"],
        )
        for _, p in top
    ]
    return ProductLookupOutput(products=products, query=inp.query, total_matches=total)


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
            "Search the Boréal Marché product catalogue by product name, category, "
            "allergen, dietary tag (vegan, gluten-free, organic, halal, kosher), or SKU."
        ),
        input_model=ProductLookupInput,
        execute=product_lookup,
    ),
]
