"""The retraining report on a throwaway registry with a SYNTHETIC stand-in candidate."""

from __future__ import annotations

import json
import sys

from scripts import retrain_report
from tests.test_registry import _good_model, _register, tracking  # noqa: F401


def test_report_writes_evidence_and_workflow_outputs(tracking, monkeypatch):  # noqa: F811
    version = _register(tracking, _good_model(), "good")
    candidate = tracking / "good"
    (candidate / "reports").mkdir()
    (candidate / "reports" / "retention_metrics.json").write_text('{"final_test": {}}')
    monkeypatch.setattr(retrain_report, "PROJECT_ROOT", tracking)
    outputs = tracking / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(outputs))
    monkeypatch.setattr(sys, "argv", ["retrain_report", "--candidate-dir", str(candidate)])

    retrain_report.main()

    written = dict(line.split("=", 1) for line in outputs.read_text().splitlines())
    assert written["passed"] == "true"
    summary = (tracking / written["summary"]).read_text()
    assert f"version {version}" in summary and "a candidate only" in summary
    assert "python -m src.registry promote --version" in summary
    folder = (tracking / written["summary"]).parent
    assert json.loads((folder / "gates.json").read_text())["passed"] is True
    assert (folder / "retention_metrics.json").exists()
