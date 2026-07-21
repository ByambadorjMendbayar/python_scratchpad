"""Load environment variables from .env file at repo root."""
import os
from pathlib import Path


def load_env():
    """Find and load .env from this repo's root directory."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        # try one level up (if called from a subdirectory script)
        env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        raise FileNotFoundError("No .env file found in repo root")

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


# Auto-load on import
load_env()
