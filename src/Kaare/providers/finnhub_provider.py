"""Finnhub news provider — fetches company news headlines for FinBERT scoring."""

import asyncio
import datetime
import hashlib
import logging
from functools import partial

import finnhub

from Kaare import config
from Kaare.models import RawNews

logger = logging.getLogger(__name__)


class FinnhubProvider:
    """Fetches company news headlines from the Finnhub API.

    All blocking SDK calls are offloaded to a thread-pool executor so the
    async event loop is never blocked.
    """

    def __init__(self) -> None:
        if not config.FINNHUB_API_KEY:
            raise RuntimeError("FINNHUB_API_KEY is not set. Check your .env file.")
        self._client = finnhub.Client(api_key=config.FINNHUB_API_KEY)

    def _fetch_company_news_sync(
        self,
        symbol: str,
        start: datetime.date,
        end: datetime.date,
    ) -> list[dict]:
        return self._client.company_news(
            symbol,
            _from=start.isoformat(),
            to=end.isoformat(),
        )

    async def get_news_by_date(
        self,
        symbols: list[str],
        start: datetime.date,
        end: datetime.date,
    ) -> dict[datetime.date, list[str]]:
        """Fetch news for *symbols* and group headline+summary texts by date.

        For each symbol, one Finnhub API call is made covering the full
        [start, end] range. Articles are then bucketed by their publish date.

        Args:
            symbols: Ticker symbols to fetch news for.
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            Mapping of date → list of article texts (``"headline. summary"``).
        """
        loop = asyncio.get_running_loop()
        by_date: dict[datetime.date, list[str]] = {}

        for symbol in symbols:
            try:
                articles = await loop.run_in_executor(
                    None,
                    partial(self._fetch_company_news_sync, symbol, start, end),
                )
            except Exception as exc:
                logger.warning("Finnhub news failed for %s: %s", symbol, exc)
                continue

            for article in articles:
                ts = article.get("datetime", 0)
                date = datetime.date.fromtimestamp(ts)
                if not (start <= date <= end):
                    continue
                headline = article.get("headline", "")
                summary = article.get("summary", "")
                text = f"{headline}. {summary}".strip(". ")
                if text:
                    by_date.setdefault(date, []).append(text)

        return by_date

    def _fetch_raw_news_sync(
        self,
        symbol: str,
        start: datetime.date,
        end: datetime.date,
    ) -> list[RawNews]:
        articles = self._fetch_company_news_sync(symbol, start, end)
        records: list[RawNews] = []
        for article in articles:
            ts = article.get("datetime", 0)
            date_utc = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
            trading_date = date_utc.date()
            if not (start <= trading_date <= end):
                continue
            headline = article.get("headline", "")
            summary = article.get("summary", "")
            text = f"{headline}. {summary}".strip(". ")
            if not text:
                continue
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            records.append(
                RawNews(
                    date_utc=date_utc,
                    trading_date=trading_date,
                    text=text,
                    text_hash=text_hash,
                    dataset_subset="finnhub",
                    source=article.get("source"),
                    tickers=[symbol],
                )
            )
        return records

    async def fetch_raw_news(
        self,
        symbol: str,
        start: datetime.date,
        end: datetime.date,
    ) -> list[RawNews]:
        """Fetch news for *symbol* and return fully-populated :class:`~Kaare.models.RawNews` objects.

        Args:
            symbol: Ticker symbol.
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            List of :class:`~Kaare.models.RawNews` ready for DB insertion.
        """
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                None, partial(self._fetch_raw_news_sync, symbol, start, end)
            )
        except Exception as exc:
            logger.warning("Finnhub fetch_raw_news failed for %s: %s", symbol, exc)
            return []
