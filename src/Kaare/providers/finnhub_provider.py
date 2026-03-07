"""Finnhub news provider — fetches company news headlines for FinBERT scoring."""

import asyncio
import datetime
import logging
from functools import partial

import finnhub

from Kaare import config

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
