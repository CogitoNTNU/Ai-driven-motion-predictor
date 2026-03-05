"""Domain models for the Kaare data layer."""

import datetime
from dataclasses import dataclass


@dataclass
class StockOHLCV:
    """OHLCV record for a single stock on a single date.

    Attributes:
        symbol: Ticker symbol (e.g. ``"AAPL"``).
        date: Trading date.
        open: Opening price.
        high: Daily high price.
        low: Daily low price.
        close: Adjusted closing price.
        volume: Number of shares traded.
    """

    symbol: str
    date: datetime.date
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class MacroData:
    """Macro-economic data for a single date.

    Attributes:
        date: Date of the record.
        gold_price: Spot gold price in USD, or ``None`` if unavailable.
        treasury_yield_10y: 10-year US Treasury yield (%), or ``None`` if unavailable.
    """

    date: datetime.date
    gold_price: float | None
    treasury_yield_10y: float | None
