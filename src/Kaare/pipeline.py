"""FinBERT sentiment pipeline — download, score, aggregate.

Implements the three-step pipeline from the plan:

1. **Download** — stream news from HuggingFace and store in ``raw_news``
2. **Score** — run FinBERT on unscored articles and store in ``article_sentiment``
3. **Aggregate** — compute daily market-wide and per-ticker sentiment tables

Usage::

    async with KaareClient() as client:
        await run_pipeline(client._db)

    # Or run individual steps:
    await download_news(pool, subsets=["sp500_daily_headlines"], start=start, end=end)
    await score_articles(pool)
    await aggregate_daily(pool)
"""

import asyncio
import datetime
import logging

import asyncpg

from Kaare import config
from Kaare.db import repository
from Kaare.providers.hf_provider import HFNewsProvider
from Kaare.sentiment import FinBERTAnalyzer

logger = logging.getLogger(__name__)


async def download_news(
    pool: asyncpg.Pool,
    subsets: list[str],
    start: datetime.date,
    end: datetime.date,
    batch_size: int = 1000,
) -> int:
    """Stream news from HuggingFace and store in the ``raw_news`` table.

    Runs the HuggingFace dataset iterator in a thread executor so the async
    event loop is never blocked. Each yielded batch is inserted immediately.

    Args:
        pool: Active asyncpg pool.
        subsets: Dataset subset names to download.
        start: Inclusive start date filter.
        end: Inclusive end date filter.
        batch_size: Records per insert batch.

    Returns:
        Approximate total records inserted (duplicates are silently skipped).
    """
    provider = HFNewsProvider()
    loop = asyncio.get_running_loop()
    gen = provider.stream_batches(subsets, start, end, batch_size)

    _sentinel = object()

    def _next_batch():
        try:
            return next(gen)
        except StopIteration:
            return _sentinel

    total = 0
    while True:
        batch = await loop.run_in_executor(None, _next_batch)
        if batch is _sentinel:
            break
        await repository.insert_raw_news_batch(pool, batch)
        total += len(batch)
        logger.info("  raw_news: ~%d rows processed so far ...", total)

    logger.info("Download complete. ~%d total rows processed.", total)
    return total


async def score_articles(
    pool: asyncpg.Pool,
    batch_size: int | None = None,
) -> int:
    """Run FinBERT on all unscored articles in ``raw_news``.

    Fetches articles in chunks, runs inference in GPU/CPU batch_size sub-batches,
    and stores results in ``article_sentiment``. Falls back to per-article scoring
    if a batch fails.

    Args:
        pool: Active asyncpg pool.
        batch_size: FinBERT inference batch size. Defaults to ``config.BATCH_SIZE``.

    Returns:
        Total articles scored in this run.
    """
    bs = batch_size or config.BATCH_SIZE
    analyzer = FinBERTAnalyzer()
    total = 0

    while True:
        # Fetch a larger chunk from DB, process in smaller inference batches
        rows = await repository.get_unscored_article_batch(pool, bs * 10)
        if not rows:
            break

        for i in range(0, len(rows), bs):
            chunk = rows[i : i + bs]
            ids = [r[0] for r in chunk]
            texts = [r[1] for r in chunk]

            try:
                scores = await analyzer.score_articles(texts)
            except Exception as exc:
                logger.warning("Batch scoring failed, falling back to per-article: %s", exc)
                scores = []
                for text in texts:
                    try:
                        s = await analyzer.score_articles([text])
                        scores.extend(s)
                    except Exception as inner:
                        logger.warning("Skipping article (scoring error): %s", inner)
                        scores.append(
                            {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "net_score": 0.0}
                        )

            await repository.insert_article_sentiment_batch(pool, list(zip(ids, scores)))
            total += len(chunk)

        logger.info("  Scored %d articles so far ...", total)

    logger.info("Scoring complete. Total scored: %d", total)
    return total


async def aggregate_daily(pool: asyncpg.Pool) -> None:
    """Aggregate article sentiment into daily market and ticker sentiment tables.

    Populates ``daily_market_sentiment`` (including 3d/7d rolling averages)
    and ``daily_ticker_sentiment``.

    Args:
        pool: Active asyncpg pool.
    """
    logger.info("Aggregating daily market-wide sentiment ...")
    n_market = await repository.aggregate_daily_market_sentiment(pool)
    logger.info("  -> %d trading days upserted.", n_market)

    logger.info("Aggregating daily per-ticker sentiment ...")
    n_ticker = await repository.aggregate_daily_ticker_sentiment(pool)
    logger.info("  -> %d ticker-day rows upserted.", n_ticker)

    logger.info("Aggregation complete.")


async def run_pipeline(
    pool: asyncpg.Pool,
    subsets: list[str] | None = None,
    start: datetime.date | None = None,
    end: datetime.date | None = None,
) -> None:
    """Run the full FinBERT pipeline: download → score → aggregate.

    Safe to re-run — all steps use ``ON CONFLICT DO NOTHING/UPDATE`` so
    existing data is never duplicated.

    Args:
        pool: Active asyncpg pool.
        subsets: HuggingFace dataset subsets to use. Defaults to
            ``config.FINANCE_SUBSETS``.
        start: Inclusive start date. Defaults to 3 years ago.
        end: Inclusive end date. Defaults to today.
    """
    today = datetime.date.today()
    start = start or (today - datetime.timedelta(days=365 * 3))
    end = end or today
    subsets = subsets or config.FINANCE_SUBSETS

    logger.info("FinBERT pipeline: %s → %s | subsets: %s", start, end, subsets)

    await download_news(pool, subsets, start, end)
    await score_articles(pool)
    await aggregate_daily(pool)

    logger.info("Pipeline complete.")
