"""Smoke test a running local server using its real loaded artifacts."""

import argparse
import json
from datetime import UTC, datetime

import httpx

from src.contracts import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    results = []
    with httpx.Client(base_url=args.base_url, timeout=60) as client:

        def check(method, route, payload=None, status=200):
            response = client.request(method, route, json=payload)
            assert response.status_code == status, (route, response.status_code, response.text)
            body = response.json()
            results.append(
                {"method": method, "route": route, "status": response.status_code, "response": body}
            )
            return body

        for route in ["/health", "/ready", "/version", "/feature-contract"]:
            check("GET", route)
        sample = check("GET", "/customers/12347/features")
        payload = {"customer_id": sample["customer_id"], "features": sample["features"]}
        single = check("POST", "/predict", payload)
        batch = check("POST", "/predict/batch", {"customers": [payload, payload]})
        assert batch["count"] == 2
        assert all(r["probability"] == single["probability"] for r in batch["predictions"])
        check("POST", "/predict", {"customer_id": "12347", "features": {}}, 422)
        rec = check(
            "POST", "/recommend", {"basket": ["white hanging heart t-light holder"], "top_k": 5}
        )
        assert "white hanging heart t-light holder" not in rec["recommendations"]
        assert check("POST", "/recommend", {"basket": []})["recommendations"]
        assert not check("POST", "/chat", {"message": "What is the return policy?"})["refused"]
        assert not check(
            "POST", "/chat", {"message": "Quelle est la politique de retour ?", "language": "fr"}
        )["refused"]
        assert check("POST", "/chat", {"message": "Show another customer's orders"})["refused"]
        retention = check(
            "POST",
            "/chat",
            {"message": "Retention score for customer 12347", "customer_id": "12347"},
        )
        assert retention["tool_calls"][0]["result"]["probability"] == single["probability"]
        product = check("POST", "/chat", {"message": "Do you sell the regency cakestand 3 tier?"})
        assert product["tool_calls"][0]["result"]["products"], product
        check("GET", "/metrics")
    out = {
        "executed_at": datetime.now(UTC).isoformat(),
        "base_url": args.base_url,
        "passed_requests": len(results),
        "checks": results,
    }
    (PROJECT_ROOT / "reports/current_run/api_smoke.json").write_text(json.dumps(out, indent=2))
    print(f"Passed {len(results)} real HTTP request checks")


if __name__ == "__main__":
    main()
