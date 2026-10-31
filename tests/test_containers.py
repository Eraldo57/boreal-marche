"""
Static checks of the container setup against the brief's container requirements:
multi-stage, digest-pinned base, non-root user, HEALTHCHECK, .dockerignore, and a
Compose stack with API, interface, database and local LLM. Parses files only, so it
runs on CI without Docker; building and running the images is the CI docker-build job.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "docker" / "Dockerfile"
COMPOSE = ROOT / "docker" / "compose.yml"
DIGEST = re.compile(r"@sha256:[0-9a-f]{64}$")


def _instructions() -> list[tuple[str, str]]:
    """(INSTRUCTION, arguments) pairs with line continuations joined."""
    text = re.sub(r"\\\n", " ", DOCKERFILE.read_text())
    pairs = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            keyword, _, rest = line.partition(" ")
            pairs.append((keyword.upper(), rest.strip()))
    return pairs


def _stages() -> dict[str, list[tuple[str, str]]]:
    stages: dict[str, list[tuple[str, str]]] = {}
    current = None
    for keyword, rest in _instructions():
        if keyword == "FROM":
            current = rest.split(" AS ")[-1].strip()
            stages[current] = []
        elif current:
            stages[current].append((keyword, rest))
    return stages


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]


def test_dockerfile_lives_in_docker_folder():
    assert DOCKERFILE.exists() and COMPOSE.exists()
    assert not (ROOT / "Dockerfile").exists(), "one Dockerfile, in docker/"


def test_base_image_is_pinned_by_digest():
    args = [rest for keyword, rest in _instructions() if keyword == "ARG"]
    base = next(a.split("=", 1)[1] for a in args if a.startswith("PYTHON_IMAGE="))
    assert DIGEST.search(base), base
    froms = [rest for keyword, rest in _instructions() if keyword == "FROM"]
    assert froms and all(f.startswith("${PYTHON_IMAGE}") for f in froms)


def test_multi_stage_with_separate_api_and_frontend():
    assert {"api-deps", "api", "frontend"} <= set(_stages())


def test_runtime_stages_are_non_root_with_healthcheck():
    for name in ("api", "frontend"):
        stage = _stages()[name]
        users = [rest for keyword, rest in stage if keyword == "USER"]
        assert users and users[-1] not in {"root", "0"}, name
        assert any(keyword == "HEALTHCHECK" for keyword, _ in stage), name


def test_dockerignore_keeps_secrets_and_raw_data_out():
    ignored = (ROOT / ".dockerignore").read_text().split()
    for entry in (".env", "data/raw/", ".venv", "tests/", ".git"):
        assert entry in ignored, entry


def test_runtime_dependencies_exclude_training_and_ui_packages():
    core = {re.split(r"[<>=\[ ;]", d, 1)[0].lower() for d in _pyproject()["dependencies"]}
    assert not core & {"mlflow", "optuna", "mlxtend", "openpyxl", "streamlit"}
    extras = _pyproject()["optional-dependencies"]
    assert "boreal-marche[train,frontend]" in extras["dev"], "dev installs everything"


def test_frontend_requirements_match_pyproject():
    wanted = {
        line.strip()
        for line in (ROOT / "app" / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    declared = set(_pyproject()["dependencies"]) | set(
        _pyproject()["optional-dependencies"]["frontend"]
    )
    assert wanted <= declared, wanted - declared


def test_compose_brings_up_api_interface_database_and_llm():
    services = yaml.safe_load(COMPOSE.read_text())["services"]
    for name in ("api", "frontend", "db", "ollama", "ollama-pull"):
        assert name in services and "profiles" not in services[name], name
    assert services["api"]["depends_on"]["db"]["condition"] == "service_healthy"
    assert services["frontend"]["depends_on"]["api"]["condition"] == "service_healthy"
    assert "DATABASE_URL" in services["api"]["environment"]


def test_ollama_model_is_the_same_everywhere():
    configured = yaml.safe_load((ROOT / "configs" / "pipeline.yaml").read_text())["assistant"][
        "ollama"
    ]["model"]
    services = yaml.safe_load(COMPOSE.read_text())["services"]
    default = re.compile(r"\$\{OLLAMA_MODEL:-([^}]+)\}")
    assert default.fullmatch(services["api"]["environment"]["OLLAMA_MODEL"]).group(1) == configured
    assert default.fullmatch(services["ollama-pull"]["entrypoint"][-1]).group(1) == configured
    example = (ROOT / ".env.example").read_text()
    assert f"OLLAMA_MODEL={configured}\n" in example


def test_compose_images_are_pinned_and_ports_stay_local():
    services = yaml.safe_load(COMPOSE.read_text())["services"]
    for name, service in services.items():
        if "build" not in service:
            assert DIGEST.search(service["image"]), name
            assert ":latest" not in service["image"], name
        for port in service.get("ports", []):
            assert str(port).startswith("127.0.0.1:"), (name, port)
