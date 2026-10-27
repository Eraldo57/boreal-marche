"""
Guardrails for the Boréal Marché assistant.

Each guardrail is a pure function:

    (query, context, ...) -> GuardrailResult(passed, reason)

The orchestrator runs them in order before and after tool execution.
A failed guardrail short-circuits the response with a polite refusal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GuardrailResult:
    """Outcome of a single guardrail check."""

    passed: bool
    reason: str | None = None


# ---------------------------------------------------------------------------
# 1. out_of_scope — reject questions unrelated to Boréal Marché
# ---------------------------------------------------------------------------

# Topics we CAN answer about.
_IN_SCOPE_SIGNALS = re.compile(
    r"bor[ée]al|march[ée]|deliver|livraison|ship|return|retour|refund|"
    r"rembours|allerg|loyalty|fid[ée]lit|point|tier|palier|order|commande|"
    r"product|produit|price|prix|coupon|discount|rabais|privacy|confidential|"
    r"loi\s*25|law\s*25|account|compte|subscription|abonnement|basket|panier|"
    r"recommend|suggest|cold.?chain|frozen|congel|refriger|shipping|exp[ée]di|"
    r"recall|rappel|cancel|annul|sku|bm-|cuisine\s*bor|policy|politiq|"
    r"gluten|vegan|organic|bio|halal|kosher|casher|review|avis|gift.?card|"
    r"carte.?cadeau|customer|client|support|chat|email|courriel|phone|"
    r"t[ée]l[ée]phone|pick.?up|click.?collect|warehouse|entrep[oô]t",
    re.IGNORECASE,
)

# Strong signals the question is entirely off-topic.
_OUT_OF_SCOPE_SIGNALS = re.compile(
    r"\b(weather|m[ée]t[ée]o|stock\s*market|bourse|bitcoin|crypto|"
    r"homework|devoir|essay|dissertation|translate\s+this|"
    r"write\s+(?:me\s+)?(?:a\s+)?(?:poem|song|story|code|script)|"
    r"who\s+is\s+the\s+president|capital\s+of|"
    r"solve|equation|integral|derivative|"
    r"(?:what|how)\s+(?:is|are)\s+(?:the\s+)?(?:meaning|definition)\s+of)\b",
    re.IGNORECASE,
)

_UNSAFE = re.compile(
    r"stock\s*(?:price|market)|invest|medical|diagnos|hack|exploit|password|credit\s*card\s*number|ignore.*(?:instruction|rule)|(?:system|developer)\s*prompt|write.*(?:python|script|poem|code)|translate.*(?:paragraph|spanish)",
    re.I,
)


def check_out_of_scope(query: str, **_: Any) -> GuardrailResult:
    """Refuse questions clearly outside Boréal Marché's domain."""
    if _UNSAFE.search(query) or _OUT_OF_SCOPE_SIGNALS.search(query):
        return GuardrailResult(
            passed=False,
            reason=(
                "This request is outside my scope. I'm the Boréal Marché assistant — "
                "I can help with delivery, "
                "returns, allergens, loyalty points, product questions, and our "
                "privacy policy. For anything outside that, please contact the "
                "appropriate service. / Je suis l'assistant Boréal Marché — je "
                "peux vous aider avec la livraison, les retours, les allergènes, "
                "les points de fidélité, les questions sur les produits et notre "
                "politique de confidentialité."
            ),
        )
    # Ambiguous — let it through; the similarity floor will catch truly
    # unrelated questions when no policy chunk matches.
    return GuardrailResult(passed=True)


# ---------------------------------------------------------------------------
# 2. similarity_floor — refuse when retrieval found nothing relevant
# ---------------------------------------------------------------------------


def check_similarity_floor(
    query: str,
    context_chunks: list[dict[str, Any]],
    *,
    threshold: float = 0.25,
    **_: Any,
) -> GuardrailResult:
    """Refuse to answer when no retrieved chunk scores above *threshold*.

    This prevents hallucinated answers to questions that look in-scope but
    are not covered by any policy document.
    """
    if not context_chunks:
        return GuardrailResult(
            passed=False,
            reason=(
                "I couldn't find any information about that in our policies. "
                "Please contact our support team for help: "
                "support@borealmarche.ca / 1-877-BOREAL-1. / "
                "Je n'ai trouvé aucune information à ce sujet dans nos politiques. "
                "Veuillez contacter notre équipe de soutien : "
                "support@borealmarche.ca / 1-877-BOREAL-1."
            ),
        )

    best_score = max(c.get("score", 0.0) for c in context_chunks)
    if best_score < threshold:
        return GuardrailResult(
            passed=False,
            reason=(
                "I'm not confident I have the right answer for that. "
                "Let me hand you off to a human agent who can help: "
                "support@borealmarche.ca / 1-877-BOREAL-1. / "
                "Je ne suis pas certain d'avoir la bonne réponse. "
                "Laissez-moi vous transférer à un agent qui pourra vous aider : "
                "support@borealmarche.ca / 1-877-BOREAL-1."
            ),
        )

    return GuardrailResult(passed=True)


# ---------------------------------------------------------------------------
# 3. no_cross_customer — refuse to disclose another customer's data
# ---------------------------------------------------------------------------

_CROSS_CUSTOMER_PATTERN = re.compile(
    r"(?:customer|client|account|compte|user|utilisateur)\s*"
    r"(?:#|num[ée]ro|number|id)?\s*"
    r"(?:of|de|for|pour)?\s*"
    r"['\"]?([A-Za-z\s]+)['\"]?\s*"
    r"(?:'s|de)?\s*"
    r"(?:info|data|order|commande|score|probability|point|address|adresse)",
    re.IGNORECASE,
)

_DIRECT_CROSS = re.compile(
    r"\b(?:show|give|tell|display|montrer|donner|afficher)\s+(?:me\s+)?"
    r"(?:another|other|someone|quelqu'?un|autre)\s*(?:customer|client|'s)?\s*"
    r"(?:info|data|order|commande|score|detail|donn[ée]e|renseignement)",
    re.IGNORECASE,
)


def check_no_cross_customer(
    query: str,
    *,
    current_customer_id: str | None = None,
    **_: Any,
) -> GuardrailResult:
    """Refuse to disclose personal data belonging to another customer."""
    if _DIRECT_CROSS.search(query) or re.search(
        r"another\s*customer|other\s*customers?|someone\s*else|autre\s*client", query, re.I
    ):
        return GuardrailResult(
            passed=False,
            reason=(
                "I can only access your own account information. I cannot look up "
                "or share another customer's data. / Je ne peux accéder qu'à vos "
                "propres informations. Je ne peux ni consulter ni partager les "
                "données d'un autre client."
            ),
        )
    # If the query mentions a specific customer id that differs from the
    # current user, block it.
    if current_customer_id:
        mentioned_ids = re.findall(
            r"(?:customer|client|account|compte)\s*(?:#|id)?\s*(?:BM-)?(\d{4,8})\b", query, re.I
        )
        for mid in mentioned_ids:
            normalised = mid.lstrip("0")
            current_normalised = current_customer_id.removeprefix("BM-").lstrip("0")
            if normalised and normalised != current_normalised:
                return GuardrailResult(
                    passed=False,
                    reason=(
                        "I can only access your own account. I am not able to "
                        "look up information for a different customer. / Je ne "
                        "peux accéder qu'à votre propre compte."
                    ),
                )

    return GuardrailResult(passed=True)


# ---------------------------------------------------------------------------
# 4. require_citations — every grounded answer must cite its sources
# ---------------------------------------------------------------------------


def check_require_citations(
    reply: str,
    citations: list[dict[str, Any]],
    **_: Any,
) -> GuardrailResult:
    """Ensure the reply includes at least one citation when grounded on policy.

    This is checked AFTER generation, not before.  If citations are missing
    the orchestrator appends them automatically rather than refusing entirely.
    """
    if not citations:
        return GuardrailResult(
            passed=False,
            reason="no_citations",  # internal signal; the orchestrator handles it
        )
    return GuardrailResult(passed=True)


# ---------------------------------------------------------------------------
# 5. language_match — respond in the same language the user wrote in
# ---------------------------------------------------------------------------

_FR_STOP = {
    "le",
    "la",
    "les",
    "de",
    "du",
    "des",
    "un",
    "une",
    "et",
    "est",
    "en",
    "dans",
    "pour",
    "sur",
    "avec",
    "par",
    "au",
    "aux",
    "ce",
    "cette",
    "nous",
    "vous",
    "sont",
    "qui",
    "que",
    "pas",
    "mon",
    "ma",
    "mes",
    "ton",
    "ta",
    "tes",
    "je",
    "tu",
    "il",
    "elle",
    "quoi",
    "comment",
    "pourquoi",
    "combien",
    "quel",
    "quelle",
}


def detect_language(text: str) -> str:
    """Simple French vs English detection based on stop-word count."""
    words = set(re.findall(r"\b[a-zàâéèêëïîôùûüç]+\b", text.lower()))
    score = len(words & _FR_STOP)
    return "fr" if score >= 3 else "en"


def check_language_match(
    query: str,
    reply: str,
    **_: Any,
) -> GuardrailResult:
    """Verify the reply language matches the query language.

    Returns a *recommendation* — the orchestrator may re-generate or translate.
    """
    q_lang = detect_language(query)
    r_lang = detect_language(reply)
    if q_lang != r_lang:
        return GuardrailResult(
            passed=False,
            reason=f"language_mismatch:expected={q_lang},got={r_lang}",
        )
    return GuardrailResult(passed=True)


# ---------------------------------------------------------------------------
# Convenience: run all pre-generation guardrails
# ---------------------------------------------------------------------------


def run_pre_guardrails(
    query: str,
    context_chunks: list[dict[str, Any]],
    *,
    similarity_threshold: float = 0.25,
    current_customer_id: str | None = None,
) -> GuardrailResult | None:
    """Run all guardrails that apply BEFORE generation.

    Returns the first failing result, or ``None`` if all pass.
    """
    for check_fn, kwargs in [
        (check_out_of_scope, {"query": query}),
        (
            check_no_cross_customer,
            {
                "query": query,
                "current_customer_id": current_customer_id,
            },
        ),
        (
            check_similarity_floor,
            {
                "query": query,
                "context_chunks": context_chunks,
                "threshold": similarity_threshold,
            },
        ),
    ]:
        result = check_fn(**kwargs)
        if not result.passed:
            return result
    return None
