"""yfinance-based fallback data provider (runs in executor to avoid blocking)."""

import asyncio
import datetime
import logging
from functools import partial

import pandas as pd
import yfinance as yf

from Kaare.models import StockOHLCV
from Kaare.providers.base import DataProvider

logger = logging.getLogger(__name__)


def _download_sync(
    ticker: str,
    start: datetime.date,
    end: datetime.date,
) -> pd.DataFrame:
    """Synchronous yfinance download (called in executor)."""
    # yfinance end date is exclusive, so add one day
    end_exclusive = end + datetime.timedelta(days=1)
    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=end_exclusive.isoformat(),
        auto_adjust=True,
        progress=False,
    )
    return df


class YFinanceProvider(DataProvider):
    """Fetches data from Yahoo Finance via the yfinance library.

    This provider is used as a fallback when AlphaVantage is unavailable.
    All blocking I/O is offloaded to a thread pool executor.
    """

    async def _download(
        self,
        ticker: str,
        start: datetime.date,
        end: datetime.date,
    ) -> pd.DataFrame:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(_download_sync, ticker, start, end),
        )

    async def get_stock_ohlcv(
        self,
        symbol: str,
        start: datetime.date,
        end: datetime.date,
    ) -> list[StockOHLCV]:
        """Fetch OHLCV data from Yahoo Finance.

        Args:
            symbol: Ticker symbol.
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            List of :class:`~Kaare.models.StockOHLCV` records.
        """
        df = await self._download(symbol, start, end)
        if df.empty:
            return []

        # Flatten MultiIndex columns if present (yfinance ≥0.2 returns them)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        records: list[StockOHLCV] = []
        for idx, row in df.iterrows():
            date = idx.date() if hasattr(idx, "date") else idx
            records.append(
                StockOHLCV(
                    symbol=symbol,
                    date=date,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=int(row["Volume"]),
                )
            )
        return records

    async def get_gold_prices(
        self,
        start: datetime.date,
        end: datetime.date,
    ) -> dict[datetime.date, float]:
        """Fetch daily spot gold prices via yfinance (ticker ``GC=F``).

        Args:
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            Mapping of date → gold price (USD).
        """
        df = await self._download("GC=F", start, end)
        if df.empty:
            return {}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        result: dict[datetime.date, float] = {}
        for idx, row in df.iterrows():
            date = idx.date() if hasattr(idx, "date") else idx
            try:
                result[date] = float(row["Close"])
            except (KeyError, ValueError):
                pass
        return result

    async def get_treasury_yields(
        self,
        start: datetime.date,
        end: datetime.date,
    ) -> dict[datetime.date, float]:
        """Fetch daily 10-year Treasury yields via yfinance (ticker ``^TNX``).

        Args:
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            Mapping of date → yield (%).
        """
        df = await self._download("^TNX", start, end)
        if df.empty:
            return {}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        result: dict[datetime.date, float] = {}
        for idx, row in df.iterrows():
            date = idx.date() if hasattr(idx, "date") else idx
            try:
                result[date] = float(row["Close"])
            except (KeyError, ValueError):
                pass
        return result

    async def get_vix(
        self,
        start: datetime.date,
        end: datetime.date,
    ) -> dict[datetime.date, float]:
        """Fetch daily VIX closing values via yfinance (ticker ``^VIX``).

        Args:
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            Mapping of date → VIX closing value.
        """
        df = await self._download("^VIX", start, end)
        if df.empty:
            return {}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        result: dict[datetime.date, float] = {}
        for idx, row in df.iterrows():
            date = idx.date() if hasattr(idx, "date") else idx
            try:
                result[date] = float(row["Close"])
            except (KeyError, ValueError):
                pass
        return result
