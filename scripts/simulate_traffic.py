#!/usr/bin/env python3
"""
Boréal Marché — Traffic Simulation Script

Generates realistic API traffic including a deliberately shifted cohort
to trigger drift alerts on the monitoring dashboard.

Usage:
    python scripts/simulate_traffic.py --base-url http://localhost:8000 --duration 120 --rps 5
    python scripts/simulate_traffic.py --shifted   # include a shifted-distribution cohort
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass

import httpx

# ---------------------------------------------------------------------------
# Realistic feature distributions (from UCI Online Retail II statistics)
# ---------------------------------------------------------------------------
NORMAL_FEATURES = {
    "recency_days":                {"mean": 45,   "std": 35,  "min": 0,   "max": 365},
    "frequency":                   {"mean": 12,   "std": 18,  "min": 1,   "max": 200},
    "monetary_total":              {"mean": 1800, "std": 2500, "min": 1,  "max": 50000},
    "monetary_mean":               {"mean": 180,  "std": 150, "min": 1,   "max": 5000},
    "monetary_std":                {"mean": 90,   "std": 80,  "min": 0,   "max": 2000},
    "avg_basket_size":             {"mean": 15,   "std": 12,  "min": 1,   "max": 100},
    "avg_basket_value":            {"mean": 280,  "std": 200, "min": 1,   "max": 5000},
    "max_basket_value":            {"mean": 600,  "std": 500, "min": 1,   "max": 10000},
    "avg_unique_items_per_basket": {"mean": 10,   "std": 8,   "min": 1,   "max": 80},
    "single_item_basket_ratio":    {"mean": 0.12, "std": 0.15, "min": 0,  "max": 1},
    "days_since_first_purchase":   {"mean": 200,  "std": 140, "min": 1,   "max": 730},
    "avg_days_between_orders":     {"mean": 35,   "std": 30,  "min": 1,   "max": 365},
    "std_days_between_orders":     {"mean": 20,   "std": 15,  "min": 0,   "max": 200},
    "is_one_time_buyer":           {"mean": 0.15, "std": 0.0, "min": 0,   "max": 1},
    "n_unique_categories":         {"mean": 8,    "std": 5,   "min": 1,   "max": 50},
    "category_entropy":            {"mean": 1.8,  "std": 0.6, "min": 0,   "max": 4},
    "top_category_share":          {"mean": 0.35, "std": 0.15, "min": 0,  "max": 1},
    "has_ever_cancelled":          {"mean": 0.2,  "std": 0.0, "min": 0,   "max": 1},
    "cancellation_rate":           {"mean": 0.03, "std": 0.05, "min": 0,  "max": 1},
    "return_rate":                 {"mean": 0.02, "std": 0.03, "min": 0,  "max": 1},
    "weekend_purchase_ratio":      {"mean": 0.22, "std": 0.1, "min": 0,   "max": 1},
    "morning_purchase_ratio":      {"mean": 0.3,  "std": 0.12, "min": 0,  "max": 1},
    "country_code":                {"mean": 0,    "std": 0,   "min": 0,   "max": 5},
    "is_domestic":                 {"mean": 0.85, "std": 0.0, "min": 0,   "max": 1},
}

# Shifted cohort: younger, lower-spend, international customers
SHIFTED_FEATURES = {
    **NORMAL_FEATURES,
    "recency_days":              {"mean": 120,  "std": 50,  "min": 0,   "max": 365},
    "frequency":                 {"mean": 3,    "std": 2,   "min": 1,   "max": 20},
    "monetary_total":            {"mean": 300,  "std": 200, "min": 1,   "max": 5000},
    "days_since_first_purchase": {"mean": 40,   "std": 20,  "min": 1,   "max": 200},
    "is_domestic":               {"mean": 0.3,  "std": 0.0, "min": 0,   "max": 1},
    "n_unique_categories":       {"mean": 2,    "std": 1,   "min": 1,   "max": 10},
}

SAMPLE_PRODUCTS = [
    "WHITE HANGING HEART T-LIGHT HOLDER",
    "REGENCY CAKESTAND 3 TIER",
    "JUMBO BAG RED RETROSPOT",
    "PARTY BUNTING",
    "LUNCH BAG RED RETROSPOT",
    "ASSORTED COLOUR BIRD ORNAMENT",
    "PACK OF 72 RETROSPOT CAKE CASES",
    "SET OF 3 CAKE TINS PANTRY DESIGN",
    "NATURAL SLATE HEART CHALKBOARD",
    "WOODEN FRAME ANTIQUE WHITE",
    "RED WOOLLY HOTTIE WHITE HEART",
    "STRAWBERRY CERAMIC TRINKET BOX",
    "PAPER CHAIN KIT 50'S CHRISTMAS",
    "ALARM CLOCK BAKELIKE GREEN",
    "POPCORN HOLDER",
]

SAMPLE_CHAT_EN = [
    "What is your return policy?",
    "How long does delivery take to Montreal?",
    "Do you ship to Ontario?",
    "Are your products allergen-free?",
    "I want to cancel my order",
    "What are your business hours?",
    "Tell me about your loyalty program",
    "How can I track my order?",
    "What payment methods do you accept?",
    "Is my data protected under Law 25?",
]

SAMPLE_CHAT_FR = [
    "Quelle est votre politique de retour?",
    "Combien de temps la livraison prend-elle à Montréal?",
    "Livrez-vous en Ontario?",
    "Vos produits sont-ils sans allergènes?",
    "Je veux annuler ma commande",
    "Quelles sont vos heures d'ouverture?",
    "Parlez-moi de votre programme de fidélité",
    "Comment puis-je suivre ma commande?",
    "Quels modes de paiement acceptez-vous?",
    "Mes données sont-elles protégées par la Loi 25?",
]


def sample_features(dist: dict) -> dict[str, float]:
    """Generate a random feature vector from a distribution."""
    features = {}
    for name, params in dist.items():
        if params["std"] == 0:
            # Binary or constant — use mean as probability for Bernoulli
            val = 1.0 if random.random() < params["mean"] else 0.0
        else:
            val = random.gauss(params["mean"], params["std"])
        val = max(params["min"], min(params["max"], val))
        features[name] = round(val, 4)
    return features


@dataclass
class Stats:
    predict_ok: int = 0
    predict_err: int = 0
    recommend_ok: int = 0
    recommend_err: int = 0
    chat_ok: int = 0
    chat_err: int = 0
    total_latency_ms: float = 0.0
    requests: int = 0


def send_predict(client: httpx.Client, stats: Stats, shifted: bool = False) -> None:
    dist = SHIFTED_FEATURES if shifted else NORMAL_FEATURES
    features = sample_features(dist)
    cid = f"SIM-{'SHIFT' if shifted else 'NORM'}-{random.randint(1000, 9999)}"
    payload = {"customer_id": cid, "features": features}

    t0 = time.monotonic()
    try:
        r = client.post("/predict", json=payload, timeout=10)
        elapsed = (time.monotonic() - t0) * 1000
        stats.total_latency_ms += elapsed
        stats.requests += 1
        if r.status_code == 200:
            stats.predict_ok += 1
            prob = r.json().get("probability", "?")
            tag = "SHIFTED" if shifted else "normal"
            print(f"  ✅ predict [{tag}] {cid} → {prob} ({elapsed:.0f}ms)")
        else:
            stats.predict_err += 1
            print(f"  ❌ predict [{cid}] → {r.status_code}")
    except Exception as e:
        stats.predict_err += 1
        stats.requests += 1
        print(f"  ❌ predict error: {e}")


def send_recommend(client: httpx.Client, stats: Stats) -> None:
    basket = random.sample(SAMPLE_PRODUCTS, k=random.randint(1, 4))
    payload = {"basket": basket, "top_k": 5}

    t0 = time.monotonic()
    try:
        r = client.post("/recommend", json=payload, timeout=10)
        elapsed = (time.monotonic() - t0) * 1000
        stats.total_latency_ms += elapsed
        stats.requests += 1
        if r.status_code == 200:
            stats.recommend_ok += 1
            recs = r.json().get("recommendations", [])
            print(f"  ✅ recommend ({len(basket)} items) → {len(recs)} recs ({elapsed:.0f}ms)")
        else:
            stats.recommend_err += 1
            print(f"  ❌ recommend → {r.status_code}")
    except Exception as e:
        stats.recommend_err += 1
        stats.requests += 1
        print(f"  ❌ recommend error: {e}")


def send_chat(client: httpx.Client, stats: Stats) -> None:
    lang = random.choice(["en", "fr"])
    questions = SAMPLE_CHAT_EN if lang == "en" else SAMPLE_CHAT_FR
    message = random.choice(questions)
    payload = {"message": message, "provider": "regex", "language": lang}

    t0 = time.monotonic()
    try:
        r = client.post("/chat", json=payload, timeout=15)
        elapsed = (time.monotonic() - t0) * 1000
        stats.total_latency_ms += elapsed
        stats.requests += 1
        if r.status_code == 200:
            stats.chat_ok += 1
            refused = r.json().get("refused", False)
            tag = "refused" if refused else "ok"
            print(f"  ✅ chat [{lang}] [{tag}] ({elapsed:.0f}ms): {message[:50]}…")
        else:
            stats.chat_err += 1
            print(f"  ❌ chat → {r.status_code}")
    except Exception as e:
        stats.chat_err += 1
        stats.requests += 1
        print(f"  ❌ chat error: {e}")


def run(base_url: str, duration: int, rps: float, shifted: bool) -> None:
    """Main simulation loop."""
    print(f"🚀 Starting traffic simulation")
    print(f"   Target: {base_url}")
    print(f"   Duration: {duration}s | Rate: {rps} req/s | Shifted cohort: {shifted}")
    print()

    stats = Stats()
    interval = 1.0 / rps if rps > 0 else 1.0
    end_time = time.monotonic() + duration

    with httpx.Client(base_url=base_url) as client:
        # Verify API is reachable
        try:
            health = client.get("/health", timeout=5)
            health.raise_for_status()
            print(f"✅ API is healthy: {health.json()}")
        except Exception as e:
            print(f"❌ Cannot reach API at {base_url}: {e}")
            sys.exit(1)

        print()
        iteration = 0
        while time.monotonic() < end_time:
            iteration += 1
            # Weighted random endpoint selection
            r = random.random()
            if r < 0.50:
                # 50% predictions (with some shifted)
                use_shifted = shifted and random.random() < 0.3
                send_predict(client, stats, shifted=use_shifted)
            elif r < 0.80:
                # 30% recommendations
                send_recommend(client, stats)
            else:
                # 20% chat
                send_chat(client, stats)

            # Sleep to maintain target rate
            time.sleep(interval)

    # Report
    print()
    print("=" * 60)
    print("📊 SIMULATION RESULTS")
    print("=" * 60)
    total = stats.predict_ok + stats.predict_err + stats.recommend_ok + stats.recommend_err + stats.chat_ok + stats.chat_err
    errors = stats.predict_err + stats.recommend_err + stats.chat_err
    avg_latency = stats.total_latency_ms / stats.requests if stats.requests else 0

    print(f"  Total requests:      {total}")
    print(f"  Predictions:         {stats.predict_ok} ok / {stats.predict_err} err")
    print(f"  Recommendations:     {stats.recommend_ok} ok / {stats.recommend_err} err")
    print(f"  Chat:                {stats.chat_ok} ok / {stats.chat_err} err")
    print(f"  Error rate:          {errors / total:.1%}" if total else "  Error rate: N/A")
    print(f"  Avg latency:         {avg_latency:.0f} ms")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Boréal Marché traffic simulator")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--rps", type=float, default=2.0, help="Requests per second")
    parser.add_argument("--shifted", action="store_true",
                        help="Include a shifted-distribution cohort (30%% of predictions) to trigger drift alerts")
    args = parser.parse_args()

    run(args.base_url, args.duration, args.rps, args.shifted)


if __name__ == "__main__":
    main()
