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


# Finnhub
FINNHUB_API_KEY: str = _optional("FINNHUB_API_KEY")

# HuggingFace
HF_TOKEN: str = _optional("HF_TOKEN")
BATCH_SIZE: int = int(_optional("BATCH_SIZE", "64"))
DEVICE: str = _optional("DEVICE", "cpu")

FINANCE_SUBSETS: list[str] = [
    "fnspid_news",
    "benzinga_6000stocks",
    "bloomberg_reuters",
    "cnbc_headlines",
    "sp500_daily_headlines",
    "finsen_us_2007_2023",
    "yahoo_finance_felixdrinkall",
]

# Database
DB_HOST: str = _optional("DB_HOST", "localhost")
DB_PORT: int = int(_optional("DB_PORT", "5432"))
DB_NAME: str = _optional("DB_NAME", "motion_predictor")
DB_USER: str = _optional("DB_USER", "admin")
DB_PASSWORD: str = _optional("DB_PASSWORD", "")

DB_DSN: str = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
