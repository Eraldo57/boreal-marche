"""Generate a local candidate, never auto-promote or overwrite the served model.

Run python -m scripts.retrain --dry-run to see commands without performing work.
Promotion/registry aliases require a new chronological holdout and human review.
"""

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-etl", action="store_true")
    parser.add_argument("--trials", type=int, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    candidate = root / "artifacts/candidates" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    commands = (
        []
        if args.skip_etl
        else [[sys.executable, "-m", "src.ingest"], [sys.executable, "-m", "src.features"]]
    )
    train = [sys.executable, "-m", "src.train", "--output-dir", str(candidate)]
    if args.trials is not None:
        train += ["--trials", str(args.trials)]
    commands.append(train)
    for command in commands:
        print(command, flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=root, check=True)
    print(
        "Candidate only. Reusing the same historical holdout is not independent promotion evidence."
    )


if __name__ == "__main__":
    main()
