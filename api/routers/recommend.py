"""
Head B — Recommendation endpoint.

POST /recommend  Given a basket, suggest what to add next.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Request

from api.schemas import FiringRule, RecommendationRequest, RecommendationResponse

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["recommendation"])


@router.post("/recommend", response_model=RecommendationResponse)
async def recommend(req: RecommendationRequest, request: Request) -> RecommendationResponse:
    """Return product recommendations for the given basket."""
    rule_engine = request.app.state.rule_engine

    recommendations, firing_rules, strategy = rule_engine.recommend(
        basket=req.basket,
        top_k=req.top_k,
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
