"""Settings loaded from environment / .env file."""

import os

from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Required environment variable '{key}' is not set. Check your .env file.")
    return value


def _optional(key: str, default: str = "") -> str:
    return os.getenv(key, default)


# Database
DB_HOST: str = _optional("DB_HOST", "localhost")
DB_PORT: int = int(_optional("DB_PORT", "5432"))
DB_NAME: str = _optional("DB_NAME", "motion_predictor")
DB_USER: str = _optional("DB_USER", "admin")
DB_PASSWORD: str = _optional("DB_PASSWORD", "")

# API keys
ALPHAVANTAGE_API_KEY: str = _optional("ALPHAVANTAGE_API_KEY", "")

DB_DSN: str = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
