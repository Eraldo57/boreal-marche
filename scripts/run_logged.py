"""Run one command and retain its exact argv, exit status, elapsed time and output.

Example: python -m scripts.run_logged ingest python -m src.ingest
"""

import json
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def main():
    stage, *command = sys.argv[1:]
    root = Path(__file__).resolve().parents[1]
    output = root / "reports/current_run/logs"
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    timestamp = datetime.now(UTC).isoformat()
    with (output / f"{stage}.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = process.wait()
    record = {
        "stage": stage,
        "command": shlex.join(command),
        "cwd": str(root),
        "started_at": timestamp,
        "elapsed_seconds": time.perf_counter() - start,
        "exit_code": code,
        "log": f"{stage}.log",
    }
    with (output / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    sys.exit(code)


if __name__ == "__main__":
    main()
