"""Summarise a retraining candidate for its pull request: metrics and promotion gates.

Run after ``python -m scripts.retrain`` (the weekly workflow does). Writes
reports/retraining/<date>/: the candidate's retention metrics, the re-mined rule metrics,
the gate report and SUMMARY.md, which becomes the pull request's description. Inside
GitHub Actions it also writes passed, date and summary to $GITHUB_OUTPUT, so the
workflow opens a pull request only when every gate passes. Nothing is promoted.
"""

import argparse
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from src import registry
from src.contracts import ARTIFACTS, PROJECT_ROOT


def newest_candidate() -> Path:
    candidates = sorted((ARTIFACTS / "candidates").glob("*/retention_model_metadata.json"))
    if not candidates:
        raise SystemExit("No candidate found; run python -m scripts.retrain first")
    return candidates[-1].parent


def candidate_version(run_id: str) -> str:
    for entry in registry.status():
        if entry["run_id"] == run_id:
            return entry["version"]
    raise SystemExit(f"Run {run_id} has no registered version")


def summary(version: str, gates: dict, date: str) -> str:
    cand, champ = gates["candidate"], gates["champion"]
    lines = [
        f"## Weekly retraining candidate, {date}",
        "",
        f"Registered as `{registry.settings()['model_name']}` version {version} in "
        f"`{registry._redacted(registry.tracking_uri())}`: a candidate only. Merging this pull "
        "request changes nothing that is served, and nothing is deployed.",
        "",
        f"| Metric ({gates['holdout']}) | Candidate | Champion |",
        "|---|---:|---:|",
    ]
    for name in ("pr_auc", "brier"):
        champion = f"{champ[name]:.4f}" if champ else "none yet"
        lines.append(f"| {name} | {cand[name]:.4f} | {champion} |")
    lines += ["", "| Gate | Passed | Detail |", "|---|---|---|"]
    lines += [
        f"| {g['gate']} | {'yes' if g['passed'] else '**no**'} | {g['detail']} |"
        for g in gates["gates"]
    ]
    flags = gates["flags_needing_acknowledgement"]
    lines += [
        "",
        f"Fairness flags an approver must acknowledge: {', '.join(flags) if flags else 'none'}.",
        "",
        "To serve this candidate after review, a named approver runs:",
        "",
        "```bash",
        f'python -m src.registry promote --version {version} --approver "Name" --reason "why"'
        + (" --acknowledge-fairness-flags" if flags else ""),
        "```",
        "",
        "The UCI workbook is a fixed historical dataset, so a weekly run reproduces the same "
        "candidate unless code or configuration changed. Policy: governance/PROMOTION_POLICY.md.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=None)
    args = parser.parse_args()
    candidate = args.candidate_dir or newest_candidate()
    metadata = json.loads((candidate / "retention_model_metadata.json").read_text())
    version = candidate_version(metadata["mlflow_run_id"])
    gates = registry.evaluate(version)
    date = datetime.now(UTC).strftime("%Y-%m-%d")
    out = PROJECT_ROOT / "reports" / "retraining" / date
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(candidate / "reports" / "retention_metrics.json", out)
    rules = PROJECT_ROOT / "reports" / "current_run" / "recommendation_metrics.json"
    if rules.exists():
        shutil.copy(rules, out)
    (out / "gates.json").write_text(json.dumps(gates, indent=2, default=str))
    (out / "SUMMARY.md").write_text(summary(version, gates, date))
    outputs = {
        "passed": str(gates["passed"]).lower(),
        "date": date,
        "summary": str((out / "SUMMARY.md").relative_to(PROJECT_ROOT)),
    }
    if os.getenv("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as handle:
            handle.writelines(f"{key}={value}\n" for key, value in outputs.items())
    print(json.dumps(outputs))


if __name__ == "__main__":
    main()
