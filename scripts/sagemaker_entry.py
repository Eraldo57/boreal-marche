"""SageMaker container contract for the training and inference image.

SageMaker starts the image (docker/Dockerfile, target "sagemaker") as ``train`` or
``serve``. For ``train`` this script does what SageMaker's script mode does: it reads
hyperparameters.json, exposes the train channel and the model folder as SM_CHANNEL_TRAIN
and SM_MODEL_DIR, and runs src/train.py with the hyperparameters as command-line
arguments. That is how the same training file runs locally and on SageMaker. For
``serve`` it starts api/sagemaker_serve.py on port 8080.

SM_BASE (default /opt/ml) moves the whole layout, which is how the tests run it locally.
"""

import json
import os
import sys
import traceback
from pathlib import Path


def train(base: Path) -> None:
    config = base / "input" / "config" / "hyperparameters.json"
    hyperparameters = json.loads(config.read_text()) if config.exists() else {}
    os.environ.setdefault("SM_CHANNEL_TRAIN", str(base / "input" / "data" / "train"))
    os.environ.setdefault("SM_MODEL_DIR", str(base / "model"))
    # SageMaker hands every value over as a string, sometimes JSON-quoted.
    arguments = [
        item
        for key, value in hyperparameters.items()
        for item in (f"--{key}", str(value).strip('"'))
    ]
    sys.argv = ["src.train", *arguments, "--no-register"]  # the registry lives elsewhere
    try:
        from src import train as training

        training.main()
    except Exception:
        # SageMaker shows this file as the job's failure reason.
        (base / "output").mkdir(parents=True, exist_ok=True)
        (base / "output" / "failure").write_text(traceback.format_exc()[-2000:])
        raise SystemExit(1) from None


def serve() -> None:
    import uvicorn

    uvicorn.run("api.sagemaker_serve:app", host="0.0.0.0", port=8080)


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "train"
    if command == "serve":
        serve()
    else:
        train(Path(os.getenv("SM_BASE", "/opt/ml")))


if __name__ == "__main__":
    main()
