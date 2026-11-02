"""MLflow model registry for the retention model: register, gate, promote, roll back.

Lifecycle (the written policy is governance/PROMOTION_POLICY.md):

1. Every training run registers its model as a new version with provenance tags.
   Registering never changes what is served.
2. ``python -m src.registry evaluate --version N`` scores the candidate and the current
   champion on the same holdout rows and checks the policy's gates.
3. ``python -m src.registry promote --version N --approver NAME --reason TEXT`` re-runs
   the gates and, only if they pass, moves the champion alias. A named person and a
   reason are required; both are stored on the version with the time and gate results.
4. ``python -m src.registry rollback --approver NAME --reason TEXT`` moves the alias back
   to the previous champion.

The API resolves the alias at startup (resolve_champion), so after a promotion or a
rollback a restart serves the new choice: no new image and no redeploy. When the
registry, the alias or the files are unavailable, the API serves artifacts/ and says so
in /version. MLflow is imported inside functions, so the API also runs without it.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from src.contracts import (
    FEATURE_NAMES,
    METADATA_PATH,
    MODEL_PATH,
    PROCESSED,
    PROJECT_ROOT,
    validate_metadata,
)
from src.fairness import assign_groups, audit
from src.ingest import _sha256, load_config

MODEL_FILE, METADATA_FILE = "retention_model.joblib", "retention_model_metadata.json"
MODEL_DIR = "model"  # run artifact folder: MLflow's sklearn model plus the checksummed files
CACHE_DIR = Path(os.getenv("REGISTRY_CACHE_DIR", PROJECT_ROOT / "data" / "registry_cache"))
EVIDENCE_DIR = PROJECT_ROOT / "reports" / "current_run" / "registry"


class PromotionRefusedError(RuntimeError):
    """A gate failed or an approval detail was missing; nothing was changed."""


def settings() -> dict[str, Any]:
    return dict(load_config()["registry"])


def tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")


def _redacted(uri: str) -> str:
    """The URI without any user:password part, safe to show in /version."""
    parts = urlsplit(uri)
    if parts.username is None:
        return uri
    return urlunsplit(parts._replace(netloc=parts.hostname or ""))


def _client() -> Any:
    import mlflow
    from mlflow import MlflowClient

    # MLflow resolves runs:/ and models:/ addresses with its process-wide settings, not
    # the client's, so both are pointed at the same store before anything is resolved.
    uri = tracking_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def load_checked(model_path: Path, metadata_path: Path) -> tuple[Any, dict[str, Any]]:
    """Load a model only if its checksum and serving contract match its metadata."""
    metadata = json.loads(metadata_path.read_text())
    validate_metadata(metadata)
    if _sha256(model_path) != metadata["model_sha256"]:
        raise ValueError("Model checksum does not match metadata")
    model = joblib.load(model_path)  # only from this machine or a trusted registry
    if list(model.feature_names_in_) != FEATURE_NAMES:
        raise ValueError("Serialized estimator feature order mismatch")
    return model, metadata


# ---------------------------------------------------------------------------
# Registering and resolving versions
# ---------------------------------------------------------------------------
def register_run(run_id: str, model_path: Path, metadata_path: Path) -> str:
    """Add the checksummed files to the run's logged model and register a new version.

    The version starts as a candidate: no alias moves, so nothing served changes.
    """
    from mlflow.exceptions import MlflowException

    metadata = json.loads(metadata_path.read_text())
    if metadata.get("mlflow_run_id") != run_id:
        raise ValueError("Metadata belongs to a different MLflow run")
    if _sha256(model_path) != metadata["model_sha256"]:
        raise ValueError("Model checksum does not match metadata")
    client = _client()
    for path in (model_path, metadata_path):
        client.log_artifact(run_id, str(path), MODEL_DIR)
    name = settings()["model_name"]
    try:
        client.get_registered_model(name)
    except MlflowException:
        client.create_registered_model(name, description="Boréal Marché 90-day retention model")
    version = client.create_model_version(
        name=name,
        source=f"runs:/{run_id}/{MODEL_DIR}",
        run_id=run_id,
        tags={
            "status": "candidate",
            "model_sha256": metadata["model_sha256"],
            "contract_version": metadata["contract_version"],
            "model_family": metadata["model_family"],
            "trained_at": metadata["trained_at"],
            "git_commit": metadata["git_commit"],
        },
    )
    return str(version.version)


def download_version(version: str, cache_dir: Path = CACHE_DIR) -> Path:
    """Download one version's files into cache_dir/<name>/<version>; return that folder."""
    import mlflow.artifacts

    name = settings()["model_name"]
    target = cache_dir / name / str(version)
    target.mkdir(parents=True, exist_ok=True)
    source = _client().get_model_version_download_uri(name, str(version))
    folder = mlflow.artifacts.download_artifacts(
        artifact_uri=source, dst_path=str(target), tracking_uri=tracking_uri()
    )
    return Path(folder)


def champion_version() -> str | None:
    from mlflow.exceptions import MlflowException

    cfg = settings()
    try:
        return str(_client().get_model_version_by_alias(cfg["model_name"], cfg["alias"]).version)
    except MlflowException:
        return None


def resolve_champion(cache_dir: Path = CACHE_DIR) -> tuple[Path, Path, dict[str, Any]]:
    """Files of the version the alias points to, plus where they came from."""
    cfg = settings()
    version = _client().get_model_version_by_alias(cfg["model_name"], cfg["alias"])
    folder = download_version(str(version.version), cache_dir)
    source = {
        "source": "registry",
        "registry_uri": _redacted(tracking_uri()),
        "model_name": cfg["model_name"],
        "alias": cfg["alias"],
        "version": str(version.version),
        "run_id": version.run_id,
    }
    return folder / MODEL_FILE, folder / METADATA_FILE, source


# ---------------------------------------------------------------------------
# Promotion gates
# ---------------------------------------------------------------------------
def _scores(
    model: Any, X: pd.DataFrame, y: np.ndarray, threshold: float, groups: pd.DataFrame
) -> dict[str, Any]:
    p = model.predict_proba(X)[:, 1]
    report = audit(y, p, threshold, groups)
    strong = {
        f"{dimension}={row['group']}": row["recall_gap_pp"]
        for dimension, rows in report["groups"].items()
        for row in rows
        if row["evidence"] == "strong"
    }
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "threshold": threshold,
        "strong_fairness_flags": strong,
    }


def check_gates(
    candidate: tuple[Any, dict[str, Any]],
    champion: tuple[Any, dict[str, Any]] | None,
    X: pd.DataFrame,
    y: np.ndarray,
    rules: dict[str, float],
) -> dict[str, Any]:
    """The policy's gates on one shared holdout; each model uses its own threshold."""
    groups = assign_groups(X, candidate[1]["country_map"])
    cand = _scores(candidate[0], X, y, candidate[1]["decision_threshold"], groups)
    gates: list[dict[str, Any]] = []

    def gate(name: str, passed: bool, detail: str) -> None:
        gates.append({"gate": name, "passed": bool(passed), "detail": detail})

    champ = None
    new_flags: list[str] = []
    widened: list[str] = []
    if champion is None:
        heuristic = float(average_precision_score(y, 1 / (1 + X["recency_days"].to_numpy())))
        gate(
            "beats_recency_baseline",
            cand["pr_auc"] > heuristic,
            f"PR-AUC {cand['pr_auc']:.4f} vs recency heuristic {heuristic:.4f} (first champion)",
        )
    else:
        gate(
            "same_feature_encoding",
            candidate[1]["country_map"] == champion[1]["country_map"],
            "country codes must mean the same countries for both models",
        )
        champ = _scores(champion[0], X, y, champion[1]["decision_threshold"], groups)
        gate(
            "pr_auc_not_worse",
            cand["pr_auc"] >= champ["pr_auc"] - rules["max_pr_auc_drop"],
            f"PR-AUC {cand['pr_auc']:.4f} vs champion {champ['pr_auc']:.4f} "
            f"(allowed drop {rules['max_pr_auc_drop']})",
        )
        gate(
            "calibration_not_worse",
            cand["brier"] <= champ["brier"] + rules["max_brier_increase"],
            f"Brier {cand['brier']:.4f} vs champion {champ['brier']:.4f} "
            f"(allowed increase {rules['max_brier_increase']})",
        )
        mine, theirs = cand["strong_fairness_flags"], champ["strong_fairness_flags"]
        new_flags = sorted(set(mine) - set(theirs))
        widened = sorted(
            group
            for group in set(mine) & set(theirs)
            if mine[group] < theirs[group] - rules["max_fairness_gap_widening_pp"]
        )
    gate("no_new_fairness_flags", not new_flags, f"new strong flags: {new_flags or 'none'}")
    gate(
        "no_widened_fairness_gaps",
        not widened,
        f"flags widened by more than {rules['max_fairness_gap_widening_pp']} pp: "
        f"{widened or 'none'}",
    )
    return {
        "passed": all(g["passed"] for g in gates),
        "gates": gates,
        "candidate": cand,
        "champion": champ,
        "flags_needing_acknowledgement": sorted(cand["strong_fairness_flags"]),
    }


def holdout(processed: Path | None = None) -> tuple[pd.DataFrame, np.ndarray, str]:
    """The test snapshot of the processed data on disk: the candidate's untouched holdout."""
    processed = processed or PROCESSED
    X = pd.read_parquet(processed / "X.parquet")[FEATURE_NAMES]
    y = pd.read_parquet(processed / "y.parquet")["will_return"]
    snapshots = pd.read_parquet(processed / "snapshots.parquet")
    test = snapshots.index[snapshots["split"] == "test"]
    cutoff = ", ".join(sorted(snapshots.loc[test, "cutoff"].unique()))
    return X.loc[test], y.loc[test].to_numpy(), cutoff


def evaluate(version: str, cache_dir: Path = CACHE_DIR) -> dict[str, Any]:
    folder = download_version(version, cache_dir)
    candidate = load_checked(folder / MODEL_FILE, folder / METADATA_FILE)
    current = champion_version()
    champion = None
    if current is not None and current != version:
        champ_folder = download_version(current, cache_dir)
        champion = load_checked(champ_folder / MODEL_FILE, champ_folder / METADATA_FILE)
    X, y, cutoff = holdout()
    report = check_gates(candidate, champion, X, y, settings()["promotion"])
    report.update(
        candidate_version=version,
        champion_version=current,
        holdout=f"test snapshot {cutoff}, {len(y):,} rows",
        evaluated_at=datetime.now(UTC).isoformat(),
    )
    return report


# ---------------------------------------------------------------------------
# Human decisions: promote and roll back
# ---------------------------------------------------------------------------
def _require(approver: str, reason: str) -> None:
    if not approver.strip() or not reason.strip():
        raise PromotionRefusedError("A named approver and a reason are required")


def _record(name: str, payload: dict[str, Any]) -> str:
    """Write a decision's evidence file; return its path for the decision record."""
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / name
    path.write_text(json.dumps(payload, indent=2))
    return str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path)


def promote(
    version: str,
    approver: str,
    reason: str,
    acknowledge_fairness_flags: bool = False,
    cache_dir: Path = CACHE_DIR,
) -> dict[str, Any]:
    _require(approver, reason)
    report = evaluate(version, cache_dir)
    if report["champion_version"] == version:
        raise PromotionRefusedError(f"Version {version} is already the champion")
    failed = [g["gate"] for g in report["gates"] if not g["passed"]]
    if failed:
        raise PromotionRefusedError(f"Gates failed: {failed}")
    flags = report["flags_needing_acknowledgement"]
    if flags and not acknowledge_fairness_flags:
        raise PromotionRefusedError(
            f"Known fairness flags must be acknowledged (--acknowledge-fairness-flags): {flags}"
        )
    cfg, client = settings(), _client()
    previous = report["champion_version"]
    client.set_registered_model_alias(cfg["model_name"], cfg["alias"], version)
    tags = {
        "status": cfg["alias"],
        "approved_by": approver,
        "approved_at": datetime.now(UTC).isoformat(),
        "approval_reason": reason,
        "previous_champion": previous or "none",
        "acknowledged_fairness_flags": json.dumps(flags),
        "gates_passed": json.dumps([g["gate"] for g in report["gates"]]),
    }
    for key, value in tags.items():
        client.set_model_version_tag(cfg["model_name"], version, key, value)
    if previous:
        client.set_model_version_tag(cfg["model_name"], previous, "status", "retired")
    report["approval"] = tags
    report["evidence_file"] = _record(f"promotion_v{version}.json", report)
    return report


def rollback(approver: str, reason: str) -> dict[str, Any]:
    """Point the alias back at the previous champion. No gates: this is the emergency path."""
    _require(approver, reason)
    cfg, client = settings(), _client()
    current = client.get_model_version_by_alias(cfg["model_name"], cfg["alias"])
    previous = current.tags.get("previous_champion", "none")
    if previous == "none":
        raise PromotionRefusedError("There is no previous champion to roll back to")
    client.set_registered_model_alias(cfg["model_name"], cfg["alias"], previous)
    now = datetime.now(UTC).isoformat()
    for key, value in {
        "status": "rolled_back",
        "rolled_back_by": approver,
        "rolled_back_at": now,
        "rollback_reason": reason,
    }.items():
        client.set_model_version_tag(cfg["model_name"], current.version, key, value)
    client.set_model_version_tag(cfg["model_name"], previous, "status", cfg["alias"])
    record = {
        "from_version": str(current.version),
        "to_version": previous,
        "approver": approver,
        "reason": reason,
        "at": now,
    }
    record["evidence_file"] = _record(f"rollback_v{current.version}_to_v{previous}.json", record)
    return record


def status() -> list[dict[str, Any]]:
    cfg, client = settings(), _client()
    versions = client.search_model_versions(f"name='{cfg['model_name']}'")
    return [
        {"version": str(v.version), "aliases": list(v.aliases), "run_id": v.run_id, **v.tags}
        for v in sorted(versions, key=lambda v: int(v.version))
    ]


def register_current() -> str:
    """Register the model in artifacts/ under the MLflow run that trained it."""
    metadata = json.loads(METADATA_PATH.read_text())
    _client().get_run(metadata["mlflow_run_id"])  # fails clearly if the run is not here
    return register_run(metadata["mlflow_run_id"], MODEL_PATH, METADATA_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="list versions, aliases and approval tags")
    commands.add_parser("register-current", help="register the model in artifacts/")
    evaluate_cmd = commands.add_parser("evaluate", help="check the gates for a version")
    evaluate_cmd.add_argument("--version", required=True)
    for name in ("promote", "rollback"):
        command = commands.add_parser(name)
        command.add_argument("--approver", required=True, help="the person taking the decision")
        command.add_argument("--reason", required=True)
        if name == "promote":
            command.add_argument("--version", required=True)
            command.add_argument("--acknowledge-fairness-flags", action="store_true")
    args = parser.parse_args()
    if args.command == "status":
        result: Any = status()
    elif args.command == "register-current":
        result = {"registered_version": register_current()}
    elif args.command == "evaluate":
        result = evaluate(args.version)
    elif args.command == "promote":
        result = promote(args.version, args.approver, args.reason, args.acknowledge_fairness_flags)
    else:
        result = rollback(args.approver, args.reason)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
