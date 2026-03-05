"""KaareClient — the public façade for the Kaare data layer."""

import datetime
import logging
from typing import Self

import asyncpg

from Kaare.db import connection as db_connection
from Kaare.db import migrations, repository
from Kaare.models import MacroData, StockOHLCV
from Kaare.providers.alphavantage import AlphaVantageProvider
from Kaare.providers.yfinance_provider import YFinanceProvider

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
    from AlphaVantage (primary) with yfinance as a fallback, then stored in
    the database before returning.

    Args:
        alphavantage_api_key: Override the API key from the environment.
    """

    def __init__(self, alphavantage_api_key: str | None = None) -> None:
        self._av = AlphaVantageProvider(api_key=alphavantage_api_key)
        self._yf = YFinanceProvider()
        self._pool: asyncpg.Pool | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create the connection pool and run schema migrations."""
        self._pool = await db_connection.create_pool()
        await migrations.run_migrations(self._pool)

    async def close(self) -> None:
        """Close the connection pool and HTTP clients."""
        if self._pool:
            await self._pool.close()
        await self._av.close()

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
        # 1. Query cache
        cached = await repository.get_stock_ohlcv(self._db, symbol, start, end)
        cached_dates = {r.date for r in cached}

        # 2. Determine missing business dates
        expected = _business_dates(start, end)
        missing = expected - cached_dates

        if missing:
            miss_start = min(missing)
            miss_end = max(missing)
            fetched: list[StockOHLCV] = []

            # 3. Fetch missing data — AlphaVantage first, yfinance fallback
            try:
                fetched = await self._av.get_stock_ohlcv(symbol, miss_start, miss_end)
                if not fetched:
                    raise ValueError("AlphaVantage returned no data")
            except Exception as exc:
                logger.warning("AlphaVantage failed for %s: %s — falling back to yfinance", symbol, exc)
                try:
                    fetched = await self._yf.get_stock_ohlcv(symbol, miss_start, miss_end)
                except Exception as yf_exc:
                    logger.error("yfinance also failed for %s: %s", symbol, yf_exc)

            # 4. Persist new rows
            if fetched:
                await repository.insert_stock_ohlcv(self._db, fetched)

        # 5. Re-query and return
        return await repository.get_stock_ohlcv(self._db, symbol, start, end)

    async def get_macro_data(
        self,
        start: datetime.date,
        end: datetime.date,
    ) -> list[MacroData]:
        """Return macro data (gold, Treasury 10Y) in [start, end], fetching from upstream if needed.

        Args:
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            List of :class:`~Kaare.models.MacroData` ordered by date ascending.
        """
        # 1. Query cache
        cached = await repository.get_macro_data(self._db, start, end)
        cached_dates = {r.date for r in cached}

        # 2. Determine missing business dates
        expected = _business_dates(start, end)
        missing = expected - cached_dates

        if missing:
            miss_start = min(missing)
            miss_end = max(missing)

            # 3a. Fetch gold prices
            gold: dict[datetime.date, float] = {}
            try:
                gold = await self._av.get_gold_prices(miss_start, miss_end)
                if not gold:
                    raise ValueError("AlphaVantage returned no gold data")
            except Exception as exc:
                logger.warning("AlphaVantage gold failed: %s — falling back to yfinance", exc)
                try:
                    gold = await self._yf.get_gold_prices(miss_start, miss_end)
                except Exception as yf_exc:
                    logger.error("yfinance gold also failed: %s", yf_exc)

            # 3b. Fetch Treasury yields
            yields: dict[datetime.date, float] = {}
            try:
                yields = await self._av.get_treasury_yields(miss_start, miss_end)
                if not yields:
                    raise ValueError("AlphaVantage returned no Treasury data")
            except Exception as exc:
                logger.warning("AlphaVantage Treasury failed: %s — falling back to yfinance", exc)
                try:
                    yields = await self._yf.get_treasury_yields(miss_start, miss_end)
                except Exception as yf_exc:
                    logger.error("yfinance Treasury also failed: %s", yf_exc)

            # 4. Merge into MacroData records and upsert
            all_dates = set(gold) | set(yields) | missing
            new_records = [
                MacroData(
                    date=d,
                    gold_price=gold.get(d),
                    treasury_yield_10y=yields.get(d),
                )
                for d in sorted(all_dates)
                if start <= d <= end
            ]
            if new_records:
                await repository.upsert_macro_data(self._db, new_records)

        # 5. Re-query and return
        return await repository.get_macro_data(self._db, start, end)
