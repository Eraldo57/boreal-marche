"""
Head B — Recommendation endpoint.

POST /recommend  Given a basket, suggest what to add next.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Request

from api.schemas import FiringRule, RecommendationRequest, RecommendationResponse

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["recommendation"])


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

    return RecommendationResponse(
        recommendations=recommendations,
        because=because,
        strategy=strategy,
    )
