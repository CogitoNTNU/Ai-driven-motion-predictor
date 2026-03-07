"""KaareClient — the public façade for the Kaare data layer."""

import datetime
import logging
from typing import Self

import asyncpg

from Kaare.db import connection as db_connection
from Kaare.db import migrations, repository
from Kaare.models import MacroData, NewsSentimentResult, StockOHLCV
from Kaare.providers.finnhub_provider import FinnhubProvider
from Kaare.providers.yfinance_provider import YFinanceProvider
from Kaare.sentiment import FinBERTAnalyzer

logger = logging.getLogger(__name__)


def _business_dates(start: datetime.date, end: datetime.date) -> set[datetime.date]:
    """Return the set of weekday dates in [start, end]."""
    dates: set[datetime.date] = set()
    current = start
    while current <= end:
        if current.weekday() < 5:  # Mon–Fri
            dates.add(current)
        current += datetime.timedelta(days=1)
    return dates


class KaareClient:
    """Cache-first async data client for stock and macro-economic data.

    Usage::

        async with KaareClient() as client:
            stocks = await client.get_stock_ohlcv("AAPL", start, end)
            macro  = await client.get_macro_data(start, end)

    On every request the database is queried first. Missing dates are fetched
    from yfinance, then stored in the database before returning.
    """

    def __init__(self) -> None:
        self._yf = YFinanceProvider()
        self._finnhub = FinnhubProvider()
        self._analyzer = FinBERTAnalyzer()
        self._pool: asyncpg.Pool | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create the connection pool and run schema migrations."""
        self._pool = await db_connection.create_pool()
        await migrations.run_migrations(self._pool)

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()

    async def __aenter__(self) -> Self:
        await self.initialize()
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _db(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("KaareClient is not initialized. Call initialize() or use as async context manager.")
        return self._pool

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_stock_ohlcv(
        self,
        symbol: str,
        start: datetime.date,
        end: datetime.date,
    ) -> list[StockOHLCV]:
        """Return OHLCV data for *symbol* in [start, end], fetching from upstream if needed.

        Args:
            symbol: Ticker symbol (e.g. ``"AAPL"``).
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            List of :class:`~Kaare.models.StockOHLCV` ordered by date ascending.
        """
        cached = await repository.get_stock_ohlcv(self._db, symbol, start, end)
        cached_dates = {r.date for r in cached}

        expected = _business_dates(start, end)
        missing = expected - cached_dates

        if missing:
            miss_start = min(missing)
            miss_end = max(missing)
            fetched: list[StockOHLCV] = []
            try:
                fetched = await self._yf.get_stock_ohlcv(symbol, miss_start, miss_end)
            except Exception as exc:
                logger.error("yfinance failed for %s: %s", symbol, exc)

            if fetched:
                await repository.insert_stock_ohlcv(self._db, fetched)

        return await repository.get_stock_ohlcv(self._db, symbol, start, end)

    async def get_macro_data(
        self,
        start: datetime.date,
        end: datetime.date,
        symbols: list[str] | None = None,
    ) -> list[MacroData]:
        """Return macro data (gold, Treasury 10Y, sentiment) in [start, end].

        Gold and Treasury yields are fetched from yfinance for missing dates.
        Sentiment is sourced from the ``daily_market_sentiment`` table, which is
        populated by running the FinBERT pipeline (see :mod:`Kaare.pipeline`).

        Args:
            start: Inclusive start date.
            end: Inclusive end date.
            symbols: Unused — kept for API compatibility. Sentiment now comes
                from the pre-computed ``daily_market_sentiment`` table.

        Returns:
            List of :class:`~Kaare.models.MacroData` ordered by date ascending.
        """
        cached = await repository.get_macro_data(self._db, start, end)
        cached_dates = {r.date for r in cached}

        expected = _business_dates(start, end)
        missing = expected - cached_dates

        if missing:
            miss_start = min(missing)
            miss_end = max(missing)

            gold: dict[datetime.date, float] = {}
            try:
                gold = await self._yf.get_gold_prices(miss_start, miss_end)
            except Exception as exc:
                logger.error("yfinance gold failed: %s", exc)

            yields: dict[datetime.date, float] = {}
            try:
                yields = await self._yf.get_treasury_yields(miss_start, miss_end)
            except Exception as exc:
                logger.error("yfinance Treasury failed: %s", exc)

            vix: dict[datetime.date, float] = {}
            try:
                vix = await self._yf.get_vix(miss_start, miss_end)
            except Exception as exc:
                logger.error("yfinance VIX failed: %s", exc)

            all_dates = set(gold) | set(yields) | set(vix) | missing
            new_records = [
                MacroData(
                    date=d,
                    gold_price=gold.get(d),
                    treasury_yield_10y=yields.get(d),
                    vix=vix.get(d),
                )
                for d in sorted(all_dates)
                if start <= d <= end
            ]
            if new_records:
                await repository.upsert_macro_data(self._db, new_records)

        # Pull pre-computed sentiment from daily_market_sentiment into macro_data
        await repository.sync_sentiment_to_macro(self._db, start, end)

        return await repository.get_macro_data(self._db, start, end)

    async def get_stock_news_sentiment(
        self,
        symbol: str,
        start: datetime.date | None = None,
        end: datetime.date | None = None,
    ) -> NewsSentimentResult:
        """Fetch recent Finnhub news for *symbol*, score with FinBERT, and persist.

        Articles are saved to ``raw_news``, per-article scores to
        ``article_sentiment``, and ``daily_ticker_sentiment`` is re-aggregated.
        Re-running for the same ticker/date range is safe — conflicts are ignored.

        Args:
            symbol: Stock ticker (e.g. ``"AAPL"``).
            start: Inclusive start date. Defaults to 7 days before *end*.
            end: Inclusive end date. Defaults to today.

        Returns:
            A :class:`~Kaare.models.NewsSentimentResult` with the average
            sentiment score in ``[-1.0, 1.0]`` and a per-date breakdown.
        """
        today = datetime.date.today()
        end = end or today
        start = start or (end - datetime.timedelta(days=7))

        articles = await self._finnhub.fetch_raw_news(symbol, start, end)

        if not articles:
            logger.warning("No news found for %s between %s and %s.", symbol, start, end)
            return NewsSentimentResult(
                symbol=symbol, start=start, end=end, article_count=0, avg_score=0.0
            )

        # Score every article individually so we can store per-article results
        texts = [a.text for a in articles]
        article_scores = await self._analyzer.score_articles(texts)

        # Persist: raw_news → article_sentiment → re-aggregate ticker sentiment
        ids = await repository.insert_raw_news_returning_ids(self._db, articles)
        await repository.insert_article_sentiment_batch(self._db, list(zip(ids, article_scores)))
        await repository.aggregate_daily_ticker_sentiment(self._db)

        # Build per-date averages for the return value
        by_date: dict[datetime.date, list[float]] = {}
        for article, score in zip(articles, article_scores):
            by_date.setdefault(article.trading_date, []).append(score["net_score"])

        daily_scores = {d: sum(v) / len(v) for d, v in by_date.items()}
        avg_score = sum(s["net_score"] for s in article_scores) / len(article_scores)

        return NewsSentimentResult(
            symbol=symbol,
            start=start,
            end=end,
            article_count=len(articles),
            avg_score=avg_score,
            daily_scores=daily_scores,
        )
