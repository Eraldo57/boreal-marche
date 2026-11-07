"""Build and verify the complete submission using only the Python standard library."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "Boreal_Marche_Complete_Submission"
DATE = "2026-11-06"
SKIP_DIRS = {
    "outputs",
    ".cloud-runtime",
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "node_modules",
    ".locks",
    "lo-profile",
    ".idea",
    ".vscode",
}
SKIP_FILES = {".DS_Store", "Thumbs.db", ".coverage", "coverage.xml", "coverage.json"}
REQUIRED = [
    "deliverables/Boreal_Marche_Defence.pptx",
    "deliverables/Boreal_Marche_Defence.pdf",
    "deliverables/Boreal_Marche_Technical_Report.docx",
    "frontend/streamlit_app.py",
    "frontend/about.py",
    "frontend/basket.py",
    "frontend/coupon.py",
    "frontend/evaluation.py",
    "frontend/notes.py",
    "api/main.py",
    "artifacts/retention_model.joblib",
    "artifacts/retention_model_metadata.json",
    "artifacts/rules.json",
    "artifacts/product_catalogue.csv",
    "data/raw/online_retail_II.xlsx",
    "data/processed/X.parquet",
    "data/chromadb/chroma.sqlite3",
    "data/chromadb/sections.json",
    "docs/SAGEMAKER.md",
    "scripts/sagemaker_jobs.py",
    "scripts/sagemaker_entry.py",
    "api/sagemaker_serve.py",
    "docker/Dockerfile",
    "docker/compose.yml",
    ".github/workflows/cd.yml",
    ".env.example",
    "SUBMISSION.md",
]
TOKEN_PATTERNS = {
    "AWS access key": re.compile(rb"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    "OpenAI key": re.compile(rb"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{40,}"),
    "GitHub token": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})"),
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def excluded(path: Path) -> str | None:
    parts = path.parts
    if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in parts):
        return "environment, tool cache or temporary profile"
    if parts[:2] in {(".dvc", "cache"), (".dvc", "tmp")}:
        return "redundant DVC cache or runtime state"
    if path.name.startswith("gha-creds-") and path.suffix == ".json":
        return "short-lived Google authentication configuration"
    if path.name in SKIP_FILES or path.suffix in {".pyc", ".pyo", ".swp"}:
        return "temporary file"
    if path.suffix == ".lock" and path.name != "dvc.lock":
        return "runtime lock file"
    if path.name.startswith(".env") and path.name != ".env.example":
        return "private environment configuration"
    if path.name in {"config.local", "secrets.toml", "credentials", "credentials.json"}:
        return "private local configuration"
    if path.suffix.lower() in {".pem", ".key"}:
        return "private key material"
    if path.name.endswith(("-wal", "-shm", "-journal")):
        return "SQLite journal covered by consistent database backup"
    if path.name in {"compose.override.yml", "docker-compose.override.yml"}:
        return "private local Docker override"
    return None


def private_values() -> list[bytes]:
    values = []
    source = ROOT / ".env"
    if source.exists():
        for line in source.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and re.search(r"KEY|TOKEN|SECRET|PASSWORD|WEBHOOK", key, re.I):
                value = value.strip().strip("\"'")
                if len(value) >= 12 and value != "boreal-local-only":
                    values.append(value.encode())
    return values


def scan(path: Path, known: list[bytes]) -> None:
    # Overlapping chunks cover token boundaries without loading model weights into RAM.
    previous = b""
    findings = set()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            data = previous + chunk
            findings.update(
                label for label, pattern in TOKEN_PATTERNS.items() if pattern.search(data)
            )
            if any(value in data for value in known):
                findings.add("private configured credential")
            previous = data[-8192:]
    if findings:
        raise ValueError(f"Credential scan stopped at {path}: {', '.join(sorted(findings))}")
    if path.suffix.lower() in {".zip", ".pptx", ".docx", ".xlsx"}:
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                # Model/office packages can contain compressed source text.
                if entry.file_size > 16 * 1024 * 1024 or entry.is_dir():
                    continue
                data = archive.read(entry)
                if any(p.search(data) for p in TOKEN_PATTERNS.values()) or any(
                    v in data for v in known
                ):
                    raise ValueError(
                        f"Credential scan stopped inside {path.name}: {entry.filename}"
                    )


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as stream:
        is_sqlite = stream.read(16) == b"SQLite format 3\x00"
    if is_sqlite:
        with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as original:
            with sqlite3.connect(target) as backup:
                original.backup(backup)
                result = backup.execute("PRAGMA integrity_check").fetchone()
                if result != ("ok",):
                    raise ValueError(f"Invalid SQLite backup: {source.name}")
    else:
        # Materialize symlinks so Windows ZIP extraction works too.
        shutil.copyfile(source, target)


def build(output: Path) -> None:
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing submission files: {missing}")
    output = output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Choose an output directory outside the source repository")
    bundle = output / NAME
    if bundle.exists() or (output / f"{NAME}.zip").exists():
        raise FileExistsError(
            f"Submission already exists in {output}; use a new --output directory"
        )
    bundle.mkdir(parents=True)
    known = private_values()
    skipped = []
    materialized = []
    # The snapshot names become regular files. Their underlying blobs are duplicates.
    blob_targets = {
        p.resolve()
        for p in (ROOT / "data/model_cache").rglob("*")
        if p.is_symlink() and p.is_file()
    }
    for current, directories, filenames in os.walk(ROOT):
        directory = Path(current)
        keep = []
        for name in sorted(directories):
            relative = (directory / name).relative_to(ROOT)
            reason = excluded(relative)
            if reason:
                skipped.append({"path": str(relative) + "/", "reason": reason})
            else:
                keep.append(name)
        directories[:] = keep
        for name in sorted(filenames):
            source = directory / name
            relative = source.relative_to(ROOT)
            reason = excluded(relative)
            if not source.is_symlink() and source in blob_targets:
                reason = "model blob materialized at its snapshot path"
            if reason:
                skipped.append({"path": str(relative), "reason": reason})
                continue
            if source.is_symlink():
                if not source.resolve().is_relative_to(ROOT):
                    raise ValueError(f"Symlink points outside the project: {relative}")
                materialized.append(str(relative))
            target = bundle / "project" / relative
            copy_file(source, target)
            scan(target, known)
    for source, subfolder in [
        (ROOT.parent / "Version_1", "assignment_brief"),
        (
            ROOT.parent / "legacy/boreal-marche_archive_2026-10-21/reports/progress_presentation",
            "historical_presentations",
        ),
    ]:
        if source.exists():
            for item in sorted(source.iterdir()):
                if item.is_file() and item.suffix.lower() in {
                    ".pdf",
                    ".html",
                    ".pptx",
                    ".ppt",
                    ".md",
                }:
                    target = bundle / subfolder / item.name
                    copy_file(item, target)
                    scan(target, known)
    shutil.copyfile(ROOT / "SUBMISSION.md", bundle / "START_HERE.md")
    (bundle / "GOOGLE_CLOUD_READ_ME.md").write_text(
        "# Google Cloud deployment\n\n"
        "Streamlit: https://boreal-ui-qaonyx3dda-nn.a.run.app\n\n"
        "API docs: https://boreal-api-qaonyx3dda-nn.a.run.app/docs\n\n"
        "- Runbook: project/docs/GOOGLE_CLOUD.md\n"
        "- Execution report and JSON evidence: project/reports/current_run/google_cloud/\n"
        "- Provision/build/deploy: project/scripts/gcp.py\n"
        "- Managed training, tuning and temporary endpoint: project/scripts/vertex_jobs.py\n"
        "- Deployment/retraining: project/.github/workflows/gcp-*.yml\n\n"
        "Google Cloud replaces the brief's AWS services at the owner's request. "
        "Consult the execution report for completed operations and remaining limitations. "
        "Credentials and service-account keys are excluded.\n",
        encoding="utf-8",
    )
    (bundle / "AWS_READ_ME.md").write_text(
        "# AWS / SageMaker materials\n\n"
        "Status: implementation and local evidence included; no live AWS execution recorded.\n\n"
        "- Run instructions: project/docs/SAGEMAKER.md\n"
        "- AWS launcher and cost comparison: project/scripts/sagemaker_jobs.py\n"
        "- Container entry point: project/scripts/sagemaker_entry.py\n"
        "- Endpoint: project/api/sagemaker_serve.py\n"
        "- Training code: project/src/train.py\n"
        "- Images: project/docker/Dockerfile and sagemaker-requirements.txt\n"
        "- CI/CD and OIDC configuration: project/.github/workflows/\n"
        "- Configuration template: project/.env.example\n"
        "- Request-contract tests: project/tests/test_sagemaker.py\n"
        "- Local SageMaker evidence: project/reports/current_run/sagemaker_local.json\n"
        "- Verification: project/reports/current_run/COMPLETION_VERIFICATION_20261105.md\n\n"
        "The dated SageMaker document records an earlier model. Current model evidence is in "
        "the completion report. Credentials and account secrets are excluded. No cloud resource "
        "is created by extracting this package.\n",
        encoding="utf-8",
    )
    write_json(
        bundle / "EXCLUSIONS.json", {"excluded": skipped, "materialized_symlinks": materialized}
    )
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    write_json(
        bundle / "BUILD_INFO.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "source_commit": revision,
            "source_had_uncommitted_changes": bool(status.strip()),
            "repository": "https://github.com/Eraldo57/boreal-marche",
            "aws_execution": "not performed; source and local evidence included",
            "credential_scan": "passed for copied files and small embedded office/archive members",
        },
    )
    entries = [
        {"path": str(p.relative_to(bundle)), "bytes": p.stat().st_size, "sha256": digest(p)}
        for p in sorted(bundle.rglob("*"))
        if p.is_file()
    ]
    write_json(bundle / "MANIFEST.json", entries)
    entries.append({"path": "MANIFEST.json", "sha256": digest(bundle / "MANIFEST.json")})
    (bundle / "SHA256SUMS.txt").write_text(
        "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in entries),
        encoding="utf-8",
    )
    archive_path = output / f"{NAME}.zip"
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                archive.write(path, str(path.relative_to(output)))
    with zipfile.ZipFile(archive_path) as archive:
        for entry in entries:
            with archive.open(f"{NAME}/{entry['path']}") as stream:
                h = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest() != entry["sha256"]:
                raise ValueError(f"ZIP checksum mismatch: {entry['path']}")
        expected = {f"{NAME}/{e['path']}" for e in entries} | {f"{NAME}/SHA256SUMS.txt"}
        if set(archive.namelist()) != expected:
            raise ValueError("ZIP membership does not match the manifest")
        if archive.read(f"{NAME}/SHA256SUMS.txt") != (bundle / "SHA256SUMS.txt").read_bytes():
            raise ValueError("Checksum file differs inside the ZIP")
    checksum = digest(archive_path)
    archive_path.with_suffix(".zip.sha256").write_text(f"{checksum}  {archive_path.name}\n")
    write_json(
        output / "VERIFICATION.json",
        {
            "archive": archive_path.name,
            "bytes": archive_path.stat().st_size,
            "sha256": checksum,
            "files": len(entries) + 1,
            "zip_member_checksums": "all passed",
            "required_files": len(REQUIRED),
            "credential_scan": "passed",
            "sqlite_backups": "integrity checked",
        },
    )
    print(
        json.dumps(
            {
                "archive": str(archive_path),
                "bytes": archive_path.stat().st_size,
                "files": len(entries) + 1,
                "sha256": checksum,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "submissions" / DATE)
    build(parser.parse_args().output)
