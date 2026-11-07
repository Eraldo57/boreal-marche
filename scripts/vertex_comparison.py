"""Compare measured local and Vertex tuning; keep estimates separate from billed cost."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.gcp import EVIDENCE, ROOT, save


def elapsed(start: str, end: str) -> float:
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()


def measure_local(n_trials: int = 6) -> None:
    if not 1 <= n_trials <= 12:
        raise ValueError("Comparison permits 1–12 trials")
    import optuna

    from src.train import hpo_trial

    trials = []
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    with TemporaryDirectory(prefix="boreal-local-hpo-") as temporary:
        output = Path(temporary)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 250),
                "max_depth": trial.suggest_int("max_depth", 4, 12),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 25),
            }
            score = hpo_trial(ROOT / "data/processed", output, params)
            trials.append(json.loads((output / "hpo_trial.json").read_text()))
            return score

        start = time.perf_counter()
        study.optimize(objective, n_trials=n_trials)
        duration = time.perf_counter() - start
    save(
        "local_optuna_comparison",
        {
            "purpose": "Fresh local RF comparison; same objective, ranges and split as Vertex",
            "trials": trials,
            "seed": 42,
            "wall_seconds": duration,
            "best_validation_pr_auc": study.best_value,
            "best_params": study.best_params,
            "additional_compute_spend": "Existing laptop; electricity not measured",
        },
    )


def compare() -> None:
    local = json.loads((EVIDENCE / "local_optuna_comparison.json").read_text())
    cloud = json.loads((EVIDENCE / "vertex_hpo.json").read_text())
    catalog = json.loads((EVIDENCE / "pricing_catalog.json").read_text())
    if cloud["state"] != "JOB_STATE_SUCCEEDED":
        raise RuntimeError("Only a completed tuning job can be compared")
    completed = [t for t in cloud["trials"] if t["state"] == "SUCCEEDED"]
    # Vertex Trial.State uses SUCCEEDED; do not count failed or infeasible trials.
    if not completed:
        raise RuntimeError("No successful trials with final measurements")
    best = max(
        completed,
        key=lambda t: t["finalMeasurement"]["metrics"][0]["value"],
    )
    prices = {}
    for sku in catalog["skus"]:
        desc = sku["description"]
        if "Training/Pipelines" not in desc or "N1" not in desc:
            continue
        price = sku["pricingInfo"][-1]["pricingExpression"]["tieredRates"][0]["unitPrice"]
        prices[desc] = int(price.get("units", 0)) + price.get("nanos", 0) / 1e9
    core = sum(v for k, v in prices.items() if "Core" in k)
    ram = sum(v for k, v in prices.items() if "Ram" in k or "RAM" in k)
    if not core or not ram:
        raise RuntimeError("Regional N1 CPU/RAM catalog prices were not captured")
    hourly = 4 * core + 15 * ram  # n1-standard-4: four vCPUs, 15 GiB.
    wall = elapsed(cloud["createTime"], cloud["endTime"])
    trial_seconds = sum(elapsed(t["startTime"], t["endTime"]) for t in cloud["trials"])
    parallel = cloud["parallelTrialCount"]
    local_per_trial = local["wall_seconds"] / len(local["trials"])
    # Empirical equal-duration model: local N*t = managed wall. It includes queue,
    # provisioning, tracking and orchestration, not just estimator fitting.
    crossover = wall / len(completed)
    save(
        "comparison_optuna_vs_vertex",
        {
            "protocol": (
                "Same historical matrices, chronological validation split, RF objective "
                "and integer ranges; different optimisers and hardware"
            ),
            "local": local,
            "vertex": {
                "name": cloud["name"],
                "trials": len(cloud["trials"]),
                "completed_trials": len(completed),
                "best_validation_pr_auc": best["finalMeasurement"]["metrics"][0]["value"],
                "best_parameters": best["parameters"],
                "wall_seconds_including_queue": wall,
                "sum_trial_elapsed_seconds_including_provisioning": trial_seconds,
                "parallel_workers": parallel,
                "catalog_ondemand_worker_usd_per_hour": hourly,
                "conservative_compute_estimate_usd": trial_seconds / 3600 * hourly,
                "estimate_basis": (
                    "Uses full trial elapsed time at on-demand regional list rates; "
                    "Spot discounts, actual VM billable times and credits are not measured. "
                    "Excludes disks, tracking, builds, storage and persistent database."
                ),
                "actual_billed_cost": None,
                "pricing_source": (
                    "Google Cloud Billing Catalog API: services/C7E2-9256-1C43/skus; "
                    "pricing_catalog.json"
                ),
            },
            "crossover": {
                "observed_local_seconds_per_trial": local_per_trial,
                "equal_duration_local_seconds_per_trial": crossover,
                "formula": "observed managed job wall seconds / completed trial count",
                "qualification": (
                    "Illustrative fixed-overhead model for this run; not a forecast for "
                    "larger jobs. Re-measure at realistic scale."
                ),
                "local_faster_in_this_experiment": local["wall_seconds"] < wall,
            },
        },
    )
    print("Comparison saved; actual billed charges remain distinct from the estimate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["local", "compare"])
    parser.add_argument("--trials", type=int, default=6)
    args = parser.parse_args()
    measure_local(args.trials) if args.command == "local" else compare()
