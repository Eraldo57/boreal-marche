"""Provision and deploy the capstone on Google Cloud using short-lived credentials.

No passwords, access tokens or service-account keys are written into the repository.
Commands save resource identifiers and execution evidence in reports/current_run/google_cloud.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import secrets
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reports/current_run/google_cloud"
DEFAULT_PROJECT = "boreal-marche-eraldo57"
DEFAULT_REGION = "northamerica-northeast1"
GITHUB_REPOSITORY_ID = "1447634288"
GITHUB_OWNER_ID = "271045992"
GCLOUD = shutil.which("gcloud") or "/opt/homebrew/share/google-cloud-sdk/bin/gcloud"


def settings(project: str = DEFAULT_PROJECT, region: str = DEFAULT_REGION) -> dict:
    return {
        "project": project,
        "region": region,
        "bucket": f"{project}-artifacts",
        "repository": "boreal",
        "database": "boreal-db",
        "accounts": {
            name: f"boreal-{name}@{project}.iam.gserviceaccount.com"
            for name in ("api", "mlflow", "train", "build", "deploy")
        },
    }


def save(name: str, value: object) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / f"{name}.json").write_text(json.dumps(value, indent=2, default=str) + "\n")


def run(args: list[str], *, stdin: str | None = None) -> str:
    result = subprocess.run(args, input=stdin, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def gcloud(cfg: dict, *args: str, stdin: str | None = None, as_json: bool = False):
    command = [
        GCLOUD,
        f"--project={cfg['project']}",
        f"--billing-project={cfg['project']}",
        "--quiet",
        *args,
    ]
    if as_json:
        command += ["--format=json"]
    result = run(command, stdin=stdin)
    return json.loads(result) if as_json and result else result


def api(cfg: dict, method: str, url: str, body: dict | None = None) -> dict:
    # Capture the token in memory, never in tool output, command arguments or evidence.
    token = gcloud(cfg, "auth", "print-access-token")
    response = requests.request(
        method,
        url,
        json=body,
        timeout=120,
        headers={"Authorization": f"Bearer {token}", "x-goog-user-project": cfg["project"]},
    )
    if not response.ok:
        raise RuntimeError(f"Google API {response.status_code}: {response.text[:2500]}")
    return response.json() if response.content else {}


def service_url(cfg: dict, name: str) -> str:
    result = gcloud(
        cfg, "run", "services", "describe", name, f"--region={cfg['region']}", as_json=True
    )
    return result["status"]["url"]


def grant_project(cfg: dict, account: str, role: str) -> None:
    gcloud(
        cfg,
        "projects",
        "add-iam-policy-binding",
        cfg["project"],
        f"--member=serviceAccount:{account}",
        f"--role={role}",
        "--condition=None",
    )


def secret_value(cfg: dict, name: str, value: str, account: str) -> None:
    base = f"https://secretmanager.googleapis.com/v1/projects/{cfg['project']}"
    api(cfg, "POST", f"{base}/secrets?secretId={name}", {"replication": {"automatic": {}}})
    api(
        cfg,
        "POST",
        f"{base}/secrets/{name}:addVersion",
        {"payload": {"data": base64.b64encode(value.encode()).decode()}},
    )
    gcloud(
        cfg,
        "secrets",
        "add-iam-policy-binding",
        name,
        f"--member=serviceAccount:{account}",
        "--role=roles/secretmanager.secretAccessor",
    )


def bootstrap(cfg: dict, billing_account: str) -> None:
    number = gcloud(cfg, "projects", "describe", cfg["project"], as_json=True)["projectNumber"]
    cfg["number"] = number
    budget = api(
        cfg,
        "POST",
        f"https://billingbudgets.googleapis.com/v1/billingAccounts/{billing_account}/budgets",
        {
            "displayName": "Boreal capstone: 10 budget alert",
            "budgetFilter": {
                "projects": [f"projects/{number}"],
                "calendarPeriod": "MONTH",
                "creditTypesTreatment": "EXCLUDE_ALL_CREDITS",
            },
            "amount": {"specifiedAmount": {"units": "10"}},
            "thresholdRules": [{"thresholdPercent": p} for p in (0.5, 0.8, 1.0)],
        },
    )
    save("budget_alert", budget)
    print("Budget alert created before starting billed compute.", flush=True)
    gcloud(
        cfg,
        "storage",
        "buckets",
        "create",
        f"gs://{cfg['bucket']}",
        f"--location={cfg['region']}",
        "--uniform-bucket-level-access",
        "--public-access-prevention",
    )
    gcloud(
        cfg,
        "artifacts",
        "repositories",
        "create",
        cfg["repository"],
        "--repository-format=docker",
        f"--location={cfg['region']}",
    )
    for name, account in cfg["accounts"].items():
        gcloud(
            cfg,
            "iam",
            "service-accounts",
            "create",
            f"boreal-{name}",
            f"--display-name=Boreal {name}",
        )
    roles = {
        "api": ["roles/cloudsql.client"],
        "mlflow": ["roles/cloudsql.client"],
        "build": ["roles/artifactregistry.writer", "roles/logging.logWriter"],
        "deploy": [
            "roles/run.admin",
            "roles/cloudbuild.builds.editor",
            "roles/artifactregistry.writer",
            "roles/aiplatform.user",
        ],
    }
    for name, names in roles.items():
        for role in names:
            grant_project(cfg, cfg["accounts"][name], role)
    for name in ("train", "mlflow", "build", "deploy"):
        gcloud(
            cfg,
            "storage",
            "buckets",
            "add-iam-policy-binding",
            f"gs://{cfg['bucket']}",
            f"--member=serviceAccount:{cfg['accounts'][name]}",
            "--role=roles/storage.objectAdmin",
        )
    for name in ("api", "mlflow", "train", "build"):
        gcloud(
            cfg,
            "iam",
            "service-accounts",
            "add-iam-policy-binding",
            cfg["accounts"][name],
            f"--member=serviceAccount:{cfg['accounts']['deploy']}",
            "--role=roles/iam.serviceAccountUser",
        )
    gcloud(
        cfg,
        "iam",
        "workload-identity-pools",
        "create",
        "github-pipeline",
        "--location=global",
        "--display-name=Boreal GitHub deployment",
    )
    condition = (
        f"assertion.repository_owner_id == '{GITHUB_OWNER_ID}' && "
        f"assertion.repository_id == '{GITHUB_REPOSITORY_ID}' && "
        "assertion.ref == 'refs/heads/main'"
    )
    gcloud(
        cfg,
        "iam",
        "workload-identity-pools",
        "providers",
        "create-oidc",
        "github-main",
        "--location=global",
        "--workload-identity-pool=github-pipeline",
        "--issuer-uri=https://token.actions.githubusercontent.com",
        "--attribute-mapping=google.subject=assertion.sub,attribute.repository_id=assertion.repository_id",
        f"--attribute-condition={condition}",
    )
    cfg["workload_identity_provider"] = (
        f"projects/{number}/locations/global/workloadIdentityPools/github-pipeline/providers/github-main"
    )
    member = (
        f"principalSet://iam.googleapis.com/projects/{number}/locations/global/"
        f"workloadIdentityPools/github-pipeline/attribute.repository_id/{GITHUB_REPOSITORY_ID}"
    )
    gcloud(
        cfg,
        "iam",
        "service-accounts",
        "add-iam-policy-binding",
        cfg["accounts"]["deploy"],
        f"--member={member}",
        "--role=roles/iam.workloadIdentityUser",
    )
    sql = gcloud(
        cfg,
        "sql",
        "instances",
        "create",
        cfg["database"],
        "--database-version=POSTGRES_16",
        "--edition=ENTERPRISE",
        "--tier=db-f1-micro",
        f"--region={cfg['region']}",
        "--storage-size=10",
        "--storage-type=HDD",
        "--no-storage-auto-increase",
        "--backup-start-time=08:00",
        "--async",
        as_json=True,
    )
    cfg["sql_connection"] = f"{cfg['project']}:{cfg['region']}:{cfg['database']}"
    save("settings", cfg)
    save("sql_create", sql)
    print("Identity federation, storage, registry and PostgreSQL provisioning started.", flush=True)


def finish_database(cfg: dict) -> None:
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        state = gcloud(cfg, "sql", "instances", "describe", cfg["database"], as_json=True)
        if state["state"] == "RUNNABLE":
            break
        print(f"PostgreSQL state: {state['state']}", flush=True)
        time.sleep(15)
    else:
        raise TimeoutError("PostgreSQL did not become ready within 20 minutes")
    sql_base = f"https://sqladmin.googleapis.com/sql/v1beta4/projects/{cfg['project']}/instances/{cfg['database']}"
    for name, secret in (
        ("boreal", "boreal-database-url"),
        ("mlflow", "boreal-mlflow-database-url"),
    ):
        password = secrets.token_urlsafe(32)
        gcloud(cfg, "sql", "databases", "create", name, f"--instance={cfg['database']}")
        operation = api(cfg, "POST", f"{sql_base}/users", {"name": name, "password": password})
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            result = api(cfg, "GET", operation["selfLink"])
            if result["status"] == "DONE":
                if "error" in result:
                    raise RuntimeError(result["error"])
                break
            time.sleep(3)
        else:
            raise TimeoutError("Database user creation timed out")
        dsn = (
            f"postgresql+psycopg://{name}:{quote(password, safe='')}@/{name}"
            f"?host=/cloudsql/{cfg['sql_connection']}"
        )
        secret_value(cfg, secret, dsn, cfg["accounts"]["api" if name == "boreal" else "mlflow"])
    print("Database credentials stored in Secret Manager.", flush=True)


def prepare_runtime(destination: Path) -> None:
    from scripts.package_submission import copy_file

    source_cache = ROOT / "data/model_cache"
    blobs = {p.resolve() for p in source_cache.rglob("*") if p.is_symlink() and p.is_file()}
    for folder in ("chromadb", "model_cache"):
        source = ROOT / "data" / folder
        for path in source.rglob("*"):
            if not path.is_file() or ".locks" in path.parts or path.name.endswith(("-wal", "-shm")):
                continue
            if not path.is_symlink() and path in blobs:
                continue
            if not path.resolve().is_relative_to(ROOT):
                raise ValueError("Runtime asset symlink leaves the project")
            copy_file(path, destination / "data" / folder / path.relative_to(source))
    feature = ROOT / "data/processed/customer_features.parquet"
    copy_file(feature, destination / "data/processed" / feature.name)
    model = ROOT / "artifacts/retention_model.joblib"
    metadata = json.loads((model.parent / "retention_model_metadata.json").read_text())
    if hashlib.sha256(model.read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError("Model checksum mismatch")


def build(cfg: dict, *, publish_runtime: bool = False) -> dict:
    sha = run(["git", "rev-parse", "HEAD"])
    prefix = f"{cfg['region']}-docker.pkg.dev/{cfg['project']}/{cfg['repository']}"
    images = {name: f"{prefix}/{name}:{sha}" for name in ("api", "frontend", "vertex", "mlflow")}
    with tempfile.TemporaryDirectory(prefix="boreal-cloud-build-") as temporary:
        context = Path(temporary)
        for folder in (
            "api",
            "src",
            "frontend",
            "configs",
            "docs/corpus",
            "governance",
            "docker",
            "scripts",
        ):
            shutil.copytree(
                ROOT / folder, context / folder, ignore=shutil.ignore_patterns("__pycache__")
            )
        for name in (
            "pyproject.toml",
            ".dockerignore",
            "MODEL_CARD.md",
            "DATA_CARD.md",
            "ARCHITECTURE.md",
        ):
            shutil.copyfile(ROOT / name, context / name)
        if (ROOT / "artifacts").exists():
            shutil.copytree(ROOT / "artifacts", context / "artifacts")
        reports = context / "reports/current_run"
        reports.mkdir(parents=True)
        for name in (
            "retention_metrics.json",
            "retention_diagnostics.json",
            "recommendation_metrics.json",
            "rules_scaling.json",
        ):
            shutil.copyfile(ROOT / "reports/current_run" / name, reports / name)
        shutil.copytree(ROOT / "reports/current_run/figures", reports / "figures")
        if publish_runtime and (ROOT / "artifacts/retention_model.joblib").exists():
            prepare_runtime(context / ".cloud-runtime")
            bundle = context / "runtime.tar.gz"
            with tarfile.open(bundle, "w:gz", dereference=True) as archive:
                archive.add(context / ".cloud-runtime", arcname=".cloud-runtime")
                archive.add(context / "artifacts", arcname="artifacts")
            gcloud(
                cfg, "storage", "cp", str(bundle), f"gs://{cfg['bucket']}/deployment/runtime.tar.gz"
            )
            cfg["runtime_sha256"] = hashlib.sha256(bundle.read_bytes()).hexdigest()
            save("settings", cfg)
            bundle.unlink()
            shutil.rmtree(context / ".cloud-runtime")
        if not cfg.get("runtime_sha256"):
            raise RuntimeError(
                "Publish the verified runtime bundle from the development machine first"
            )
        steps = [
            {
                "name": "gcr.io/google.com/cloudsdktool/google-cloud-cli:slim",
                "entrypoint": "python3",
                "args": [
                    "scripts/cloud_build_runtime.py",
                    f"gs://{cfg['bucket']}/deployment/runtime.tar.gz",
                    cfg["runtime_sha256"],
                ],
            }
        ]
        for name, target in (
            ("api", "cloudrun-api"),
            ("frontend", "frontend"),
            ("vertex", "vertex"),
            ("mlflow", "mlflow"),
        ):
            steps.append(
                {
                    "name": "gcr.io/cloud-builders/docker",
                    "args": [
                        "build",
                        "-f",
                        "docker/Dockerfile",
                        "--target",
                        target,
                        "--build-arg",
                        f"GIT_SHA={sha}",
                        "-t",
                        images[name],
                        ".",
                    ],
                }
            )
        config = {
            "steps": steps,
            "images": list(images.values()),
            "timeout": "1800s",
            "logsBucket": f"gs://{cfg['bucket']}/build-logs",
            "options": {"logging": "GCS_ONLY"},
            "serviceAccount": (
                f"projects/{cfg['project']}/serviceAccounts/{cfg['accounts']['build']}"
            ),
        }
        file = context / "cloudbuild.yaml"
        file.write_text(yaml.safe_dump(config, sort_keys=False))
        print("Building and pushing four SHA-tagged images.", flush=True)
        result = gcloud(
            cfg,
            "builds",
            "submit",
            str(context),
            f"--config={file}",
            f"--region={cfg['region']}",
            f"--gcs-source-staging-dir=gs://{cfg['bucket']}/build-source",
            as_json=True,
        )
        save("build", result)
    cfg.update({"git_sha": sha, "images": images})
    save("settings", cfg)
    return cfg


def deploy_service(
    cfg: dict,
    name: str,
    image: str,
    account: str,
    port: int,
    memory: str,
    env: dict,
    *,
    public: bool,
    database: bool = False,
    secret: str | None = None,
) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as file:
        yaml.safe_dump({str(k): str(v) for k, v in env.items()}, file)
        file.flush()
        args = [
            "run",
            "deploy",
            name,
            f"--image={image}",
            f"--region={cfg['region']}",
            f"--service-account={account}",
            f"--port={port}",
            f"--memory={memory}",
            "--cpu=1",
            "--min-instances=0",
            "--max-instances=1",
            "--concurrency=8",
            "--timeout=300",
            "--execution-environment=gen2",
            "--cpu-boost",
            f"--env-vars-file={file.name}",
            "--allow-unauthenticated" if public else "--no-allow-unauthenticated",
        ]
        if database:
            args += [f"--add-cloudsql-instances={cfg['sql_connection']}"]
        if name == "boreal-api":
            # Keep the alert loop running between requests, while retaining scale-to-zero.
            args += ["--no-cpu-throttling"]
        if secret:
            args += [f"--set-secrets={secret}"]
        result = gcloud(cfg, *args, as_json=True)
    save(name, result)
    return result["status"]["url"]


def deploy(cfg: dict) -> dict:
    cfg["mlflow_url"] = deploy_service(
        cfg,
        "boreal-mlflow",
        cfg["images"]["mlflow"],
        cfg["accounts"]["mlflow"],
        8080,
        "1Gi",
        {
            "MLFLOW_ARTIFACTS_DESTINATION": f"gs://{cfg['bucket']}/mlflow",
            "MLFLOW_SQLALCHEMYSTORE_POOL_SIZE": "2",
            "MLFLOW_SQLALCHEMYSTORE_MAX_OVERFLOW": "1",
        },
        public=False,
        database=True,
        secret="MLFLOW_BACKEND_STORE_URI=boreal-mlflow-database-url:latest",
    )
    for name in ("api", "train", "deploy"):
        gcloud(
            cfg,
            "run",
            "services",
            "add-iam-policy-binding",
            "boreal-mlflow",
            f"--region={cfg['region']}",
            f"--member=serviceAccount:{cfg['accounts'][name]}",
            "--role=roles/run.invoker",
        )
    cfg["api_url"] = deploy_service(
        cfg,
        "boreal-api",
        cfg["images"]["api"],
        cfg["accounts"]["api"],
        8000,
        "2Gi",
        {
            "GIT_SHA": cfg["git_sha"],
            "MODEL_SOURCE": "registry",
            "BOREAL_NO_DOTENV": "1",
            "MLFLOW_TRACKING_URI": cfg["mlflow_url"],
            "MLFLOW_GOOGLE_IAM_AUTH": "true",
            "MLFLOW_HTTP_REQUEST_TIMEOUT": "60",
            "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "0",
        },
        public=True,
        database=True,
        secret="DATABASE_URL=boreal-database-url:latest",
    )
    cfg["ui_url"] = deploy_service(
        cfg,
        "boreal-ui",
        cfg["images"]["frontend"],
        cfg["accounts"]["api"],
        8501,
        "512Mi",
        {
            "API_BASE_URL": cfg["api_url"],
            "HOSTED_DEMO": "true",
            "ASSISTANT_PROVIDERS": "regex",
        },
        public=True,
    )
    save("settings", cfg)
    gcloud(
        cfg,
        "storage",
        "cp",
        str(EVIDENCE / "settings.json"),
        f"gs://{cfg['bucket']}/deployment/settings.json",
    )
    smoke(cfg)
    return cfg


def smoke(cfg: dict) -> None:
    base = cfg["api_url"]
    session = requests.Session()
    results = {}
    for endpoint in ("health", "ready", "version", "openapi.json", "metrics", "alerts"):
        response = session.get(f"{base}/{endpoint}", timeout=180)
        response.raise_for_status()
        results[endpoint] = response.json()
    if not results["ready"]["ready"] or results["version"]["git_sha"] != cfg["git_sha"]:
        raise RuntimeError("Deployment readiness or Git SHA check failed")
    record = session.get(f"{base}/customers/12347/features", timeout=60).json()
    for endpoint, payload in (
        ("predict", {"customer_id": "12347", "features": record["features"]}),
        ("recommend", {"basket": [], "top_k": 5}),
        (
            "chat",
            {
                "message": "What is the return policy for non-perishable items?",
                "language": "en",
                "provider": "regex",
            },
        ),
    ):
        response = session.post(f"{base}/{endpoint}", json=payload, timeout=180)
        response.raise_for_status()
        results[endpoint] = response.json()
    ui = session.get(f"{cfg['ui_url']}/_stcore/health", timeout=180)
    ui.raise_for_status()
    results["streamlit_health"] = ui.text
    save("smoke", results)
    print(
        json.dumps(
            {"api": base, "ui": cfg["ui_url"], "git_sha": cfg["git_sha"], "smoke": "passed"}
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["bootstrap", "database", "build", "deploy", "smoke", "runtime"]
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--billing-account")
    args = parser.parse_args()
    cfg = settings(args.project, args.region)
    file = EVIDENCE / "settings.json"
    if file.exists():
        cfg.update(json.loads(file.read_text()))
    if args.command == "bootstrap":
        if not args.billing_account:
            parser.error("bootstrap needs --billing-account")
        bootstrap(cfg, args.billing_account)
    elif args.command == "database":
        finish_database(cfg)
    elif args.command == "build":
        build(cfg, publish_runtime=True)
    elif args.command == "deploy":
        deploy(cfg)
    elif args.command == "smoke":
        smoke(cfg)
    else:
        prepare_runtime(ROOT / ".cloud-runtime")


if __name__ == "__main__":
    main()
