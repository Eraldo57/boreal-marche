"""Exercise Streamlit widgets against the running real FastAPI backend.

The app reads the API address from API_BASE_URL, as it does when served.
"""

import argparse
import json
from datetime import UTC, datetime

import httpx
from streamlit.testing.v1 import AppTest

from src.contracts import FEATURE_NAMES, PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streamlit-url", default="http://127.0.0.1:8501")
    args = parser.parse_args()
    health = httpx.get(f"{args.streamlit_url}/_stcore/health")
    health.raise_for_status()
    app = AppTest.from_file(str(PROJECT_ROOT / "frontend/streamlit_app.py"), default_timeout=60)
    app.run()
    checks = []

    def check(label):
        assert not app.exception, list(app.exception)
        assert not app.error, [e.value for e in app.error]
        checks.append(label)

    def click(label):
        next(b for b in app.button if b.label == label).click().run()
        check(label)

    check("initial render and API readiness")
    click("Load historical customer features")
    assert len(app.number_input) == len(FEATURE_NAMES)
    assert any("Why this score" in h.value for h in app.subheader), "must explain this customer"
    assert any(m.label == "Probability of purchase within 90 days" for m in app.metric)
    before = next(m.value for m in app.metric if m.label.startswith("Probability"))
    app.number_input[0].set_value(app.number_input[0].value + 365).run()  # what-if: recency
    check("live what-if update")
    after = next(m.value for m in app.metric if m.label.startswith("Probability"))
    assert after != before, "changing recency must update the probability without a button"
    click("Get recommendations")
    shown = " ".join(m.value for m in app.markdown)
    assert "because you added" in shown, "a rule-based list must say which item triggered each"
    basket = next(t for t in app.text_area if t.label.startswith("One item per line"))
    if app.multiselect:
        app.multiselect[0].set_value([]).run()
        basket = next(t for t in app.text_area if t.label.startswith("One item per line"))
    basket.set_value("never-seen-item").run()
    click("Get recommendations")
    assert any("No rule matches" in w.value for w in app.warning), "fallback must be announced"
    click("Ask assistant")
    assert len(app.expander) >= 2
    click("Refresh metrics")
    report = {
        "executed_at": datetime.now(UTC).isoformat(),
        "method": "Streamlit AppTest widgets using real localhost FastAPI; HTTP server health",
        "checks": checks,
        "feature_widgets": len(app.number_input),
        "exceptions": 0,
        "limitations": "Not a browser visual or multi-user concurrency test",
    }
    (PROJECT_ROOT / "reports/current_run/ui_smoke.json").write_text(json.dumps(report, indent=2))
    print(f"Passed {len(checks)} Streamlit/API interaction checks")


if __name__ == "__main__":
    main()
