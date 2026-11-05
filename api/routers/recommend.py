"""
Head B — Recommendation endpoint.

POST /recommend  Given a basket, suggest what to add next.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Request

from api.schemas import (
    FiringRule,
    RecommendationDetailsResponse,
    RecommendationRequest,
    RecommendationResponse,
)

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["recommendation"])


@router.get("/recommend/details", response_model=RecommendationDetailsResponse)
async def recommendation_details(request: Request) -> dict:
    """All public training item descriptions, loaded rules and mining provenance."""
    engine = request.app.state.rule_engine
    if engine is None:
        raise HTTPException(503, "Recommendation artifacts missing; run python -m src.rules")
    # The compatibility adapter used by existing fixtures exposes the same real engine.
    return dict(getattr(engine, "engine", engine).describe())


@router.post("/recommend", response_model=RecommendationResponse)
async def recommend(req: RecommendationRequest, request: Request) -> RecommendationResponse:
    """Return product recommendations for the given basket."""
    rule_engine = request.app.state.rule_engine

    if rule_engine is None:
        raise HTTPException(503, "Recommendation artifacts missing; run python -m src.rules")
    result = rule_engine.recommend(
        basket=req.basket,
        top_k=req.top_k,
    )
    if isinstance(result, tuple):  # legacy adapter used only by existing unit fixtures
        recommendations, firing_rules, strategy = result
    else:
        recommendations, firing_rules, strategy = (
            result["recommendations"],
            result["because"],
            result["strategy"],
        )

    because = [
        FiringRule(
            antecedent=r["antecedent"],
            consequent=r["consequent"],
            confidence=r["confidence"],
            lift=r["lift"],
            support=r["support"],
        )
        for r in firing_rules
    ]

    logger.info(
        "recommendation_served",
        basket_size=len(req.basket),
        n_recommendations=len(recommendations),
        strategy=strategy,
        customer_id=req.customer_id,
    )
    # Logged by the middleware for the rule fire-rate monitor; no item names or IDs.
    request.state.recommendation = {
        "strategy": strategy,
        "basket_size": len(req.basket),
        "rules_fired": len(firing_rules),
    }

    return RecommendationResponse(
        recommendations=recommendations,
        because=because,
        strategy=strategy,
    )
