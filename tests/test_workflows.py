"""
Static checks of the GitHub Actions workflows against the brief (section 3.6): CI on
every pull request, CD on merge to main with OIDC, and a weekly retraining workflow that
opens a pull request but never deploys. Files are parsed, nothing is executed.
"""

from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _load(name: str) -> dict:
    workflow = yaml.safe_load((WORKFLOWS / name).read_text())
    workflow["on"] = workflow.pop(True, workflow.get("on"))  # YAML 1.1 reads "on" as True
    return workflow


def _steps_text(workflow: dict) -> str:
    return yaml.safe_dump(workflow["jobs"]).lower()


def test_ci_runs_every_required_gate_on_pull_requests():
    ci = _load("ci.yml")
    assert "pull_request" in ci["on"]
    jobs = set(ci["jobs"])
    assert {"lint", "typecheck", "test", "data-quality", "docker-build", "secret-scan"} <= jobs
    assert "assistant-tool-choice" in jobs
    assert "--fail-under=80" in _steps_text(ci)


def test_cd_deploys_on_merge_with_oidc_sha_tag_and_version_smoke_test():
    cd = _load("cd.yml")
    assert cd["on"]["push"]["branches"] == ["main"]
    assert cd["permissions"]["id-token"] == "write"
    text = _steps_text(cd)
    assert "role-to-assume" in text and "aws-access-key-id" not in text
    assert "github.sha" in text and "/version" in text
    assert "retrain" not in text, "retraining has its own workflow"


def test_retraining_is_weekly_opens_a_pull_request_and_never_deploys():
    retrain = _load("retrain.yml")
    assert retrain["on"]["schedule"][0]["cron"].split()[-1] == "1"  # weekly, on Mondays
    text = _steps_text(retrain)
    for stage in ("src.ingest", "src.features", "src.rules", "scripts.retrain"):
        assert stage in text, stage
    assert "create-pull-request" in text and "steps.gates.outputs.passed == 'true'" in text
    for deploy in ("apprunner", "amazon-ecr-login", "configure-aws-credentials", "build-push"):
        assert deploy not in text, deploy
    assert "id-token" not in retrain.get("permissions", {})
