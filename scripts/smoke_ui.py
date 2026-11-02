"""Exercise Streamlit widgets against the running real FastAPI backend."""

import json
from datetime import UTC, datetime

import httpx
from streamlit.testing.v1 import AppTest

from src.contracts import PROJECT_ROOT


def main():
    health = httpx.get("http://127.0.0.1:8501/_stcore/health")
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
    assert len(app.number_input) == 25
    click("Predict")
    assert any(m.label == "Probability of purchase within 90 days" for m in app.metric)
    click("Get recommendations")
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
