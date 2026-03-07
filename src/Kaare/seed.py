"""Seed script — populate Kaare with the 10 largest stocks, macro data, and FinBERT sentiment.

Usage::

    # Full seed (stocks + macro + sentiment pipeline)
    uv run python -m Kaare.seed

    # Sentiment pipeline only (after stocks/macro are already seeded)
    uv run python -m Kaare.seed --pipeline-only

    # Quick test with a single small subset
    uv run python -m Kaare.seed --subset sp500_daily_headlines
"""

import argparse
import asyncio
import datetime
import logging

from Kaare.client import KaareClient
from Kaare.db import connection as db_connection, migrations
from Kaare.pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Top 10 stocks by market cap (as of early 2026)
TOP_10_SYMBOLS = [
    "AAPL",   # Apple
    "NVDA",   # NVIDIA
    "MSFT",   # Microsoft
    "AMZN",   # Amazon
    "GOOGL",  # Alphabet
    "META",   # Meta
    "TSLA",   # Tesla
    "BRK-B",  # Berkshire Hathaway
    "AVGO",   # Broadcom
    "TSM",    # TSMC
]

END = datetime.date.today()
START = END - datetime.timedelta(days=365 * 3)


async def main(pipeline_only: bool = False, subset: str | None = None) -> None:
    logger.info("Seeding Kaare from %s to %s", START, END)

    if not pipeline_only:
        async with KaareClient() as client:
            for symbol in TOP_10_SYMBOLS:
                logger.info("Fetching %s ...", symbol)
                try:
                    records = await client.get_stock_ohlcv(symbol, START, END)
                    logger.info("  -> %d rows stored for %s", len(records), symbol)
                except Exception as exc:
                    logger.error("  -> Failed to fetch %s: %s", symbol, exc)

            logger.info("Fetching macro data (gold + Treasury yields) ...")
            try:
                macro = await client.get_macro_data(START, END)
                logger.info("  -> %d macro rows stored", len(macro))
            except Exception as exc:
                logger.error("  -> Failed to fetch macro data: %s", exc)

    # Run the FinBERT pipeline: download HF news → score → aggregate
    logger.info("Running FinBERT sentiment pipeline ...")
    try:
        pool = await db_connection.create_pool()
        await migrations.run_migrations(pool)
        subsets = [subset] if subset else None
        await run_pipeline(pool, subsets=subsets, start=START, end=END)
        await pool.close()
    except Exception as exc:
        logger.error("FinBERT pipeline failed: %s", exc)

    logger.info("Seed complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline-only",
        action="store_true",
        help="Skip stock/macro fetch and only run the FinBERT sentiment pipeline.",
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="Run pipeline with a single HF subset (e.g. sp500_daily_headlines) for testing.",
    )
    args = parser.parse_args()
    asyncio.run(main(pipeline_only=args.pipeline_only, subset=args.subset))
