"""HuggingFace financial news dataset provider.

Streams articles from ``Brianferrell787/financial-news-multisource`` and
yields batches of :class:`~Kaare.models.RawNews` objects for insertion into
the ``raw_news`` table.
"""

import datetime
import hashlib
import json
import logging
from collections.abc import Generator

from Kaare import config
from Kaare.models import RawNews

logger = logging.getLogger(__name__)

_DATASET_NAME = "Brianferrell787/financial-news-multisource"


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_extras(row: dict) -> dict:
    try:
        return json.loads(row.get("extra_fields") or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}


def _parse_trading_date(row: dict, extras: dict) -> datetime.date | None:
    trading_dt = extras.get("date_trading")
    if trading_dt:
        try:
            return datetime.date.fromisoformat(str(trading_dt)[:10])
        except ValueError:
            pass
    date_str = row.get("date", "")
    try:
        return datetime.date.fromisoformat(str(date_str)[:10])
    except ValueError:
        return None


def _parse_date_utc(row: dict, fallback: datetime.date) -> datetime.datetime:
    date_str = str(row.get("date", ""))
    try:
        if "T" in date_str:
            return datetime.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return datetime.datetime.combine(fallback, datetime.time.min, tzinfo=datetime.timezone.utc)
    except ValueError:
        return datetime.datetime.combine(fallback, datetime.time.min, tzinfo=datetime.timezone.utc)


def _parse_tickers(extras: dict) -> list[str]:
    tickers = extras.get("stocks", [])
    if isinstance(tickers, str):
        tickers = [tickers]
    return [str(t) for t in tickers if t]


def _parse_source(extras: dict) -> str | None:
    return extras.get("source") or extras.get("publisher") or None


def _parse_subset(extras: dict) -> str:
    return extras.get("dataset", "unknown")


class HFNewsProvider:
    """Streams financial news from HuggingFace for FinBERT ingestion.

    Example::

        provider = HFNewsProvider()
        for batch in provider.stream_batches(["sp500_daily_headlines"], start, end):
            await repository.insert_raw_news_batch(pool, batch)
    """

    def stream_batches(
        self,
        subsets: list[str],
        start: datetime.date,
        end: datetime.date,
        batch_size: int = 1000,
    ) -> Generator[list[RawNews], None, None]:
        """Stream and parse news articles, yielding batches of :class:`~Kaare.models.RawNews`.

        This is a **synchronous** generator — call it from a thread executor.

        Args:
            subsets: Dataset subset names to load (e.g. ``["sp500_daily_headlines"]``).
            start: Inclusive start date filter.
            end: Inclusive end date filter.
            batch_size: Number of records per yielded batch.

        Yields:
            Batches of parsed :class:`~Kaare.models.RawNews` objects.
        """
        from datasets import load_dataset  # lazy import — heavy dependency

        data_files = [f"data/{s}/*.parquet" for s in subsets]
        logger.info("Streaming %d HuggingFace subset(s): %s", len(subsets), subsets)

        ds = load_dataset(
            _DATASET_NAME,
            data_files=data_files,
            split="train",
            streaming=True,
            token=config.HF_TOKEN or None,
        )

        batch: list[RawNews] = []
        skipped = 0

        for row in ds:
            text = row.get("text", "")
            if not text or len(text.strip()) < 20:
                skipped += 1
                continue

            extras = _parse_extras(row)
            trading_date = _parse_trading_date(row, extras)
            if trading_date is None or not (start <= trading_date <= end):
                skipped += 1
                continue

            batch.append(
                RawNews(
                    date_utc=_parse_date_utc(row, trading_date),
                    trading_date=trading_date,
                    text=text,
                    text_hash=_text_hash(text),
                    dataset_subset=_parse_subset(extras),
                    source=_parse_source(extras),
                    tickers=_parse_tickers(extras),
                )
            )

            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

        logger.info("Streaming complete. Skipped %d rows (too short or out of date range).", skipped)
