"""Read the repository's .env file into the environment.

Docker Compose reads .env by itself. This makes native runs (uvicorn, the evaluation and
SageMaker scripts) see the same file. A variable already set in the environment wins over the
file, and BOREAL_NO_DOTENV=1 turns the loading off: the tests set it, so a developer's real
key never reaches them.
"""

import os

from dotenv import load_dotenv

from src.contracts import PROJECT_ROOT


def load_env_file() -> bool:
    """Load PROJECT_ROOT/.env without overriding the environment; False if skipped or absent."""
    if os.getenv("BOREAL_NO_DOTENV"):
        return False
    return load_dotenv(PROJECT_ROOT / ".env", override=False)
