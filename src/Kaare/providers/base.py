"""Abstract base class for data providers."""

import datetime
from abc import ABC, abstractmethod

from Kaare.models import MacroData, StockOHLCV


class DataProvider(ABC):
    """Interface that every concrete data provider must implement."""

    @abstractmethod
    async def get_stock_ohlcv(
        self,
        symbol: str,
        start: datetime.date,
        end: datetime.date,
    ) -> list[StockOHLCV]:
        """Fetch OHLCV data for *symbol* within [start, end].

        Args:
            symbol: Ticker symbol.
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            List of :class:`~Kaare.models.StockOHLCV` records.
        """

    @abstractmethod
    async def get_gold_prices(
        self,
        start: datetime.date,
        end: datetime.date,
    ) -> dict[datetime.date, float]:
        """Fetch daily spot gold prices within [start, end].

        Args:
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            Mapping of date → gold price (USD).
        """

    @abstractmethod
    async def get_treasury_yields(
        self,
        start: datetime.date,
        end: datetime.date,
    ) -> dict[datetime.date, float]:
        """Fetch daily 10-year Treasury yields within [start, end].

        Args:
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            Mapping of date → yield (%).
        """
