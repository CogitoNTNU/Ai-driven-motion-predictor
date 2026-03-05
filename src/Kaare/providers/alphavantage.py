"""AlphaVantage async data provider."""

import datetime
import logging

import httpx

from Kaare import config
from Kaare.models import StockOHLCV
from Kaare.providers.base import DataProvider

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.alphavantage.co/query"

# AlphaVantage uses "compact" (100 days) or "full" (20 years)
_COMPACT_THRESHOLD_DAYS = 90


class AlphaVantageProvider(DataProvider):
    """Fetches data from the AlphaVantage REST API using async HTTP.

    Args:
        api_key: AlphaVantage API key. Defaults to ``ALPHAVANTAGE_API_KEY`` from config.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key or config.ALPHAVANTAGE_API_KEY
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _get(self, params: dict) -> dict:
        client = await self._get_client()
        params["apikey"] = self._api_key
        response = await client.get(_BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        if "Error Message" in data:
            raise ValueError(f"AlphaVantage error: {data['Error Message']}")
        if "Note" in data:
            raise RuntimeError(f"AlphaVantage rate-limit: {data['Note']}")
        return data

    def _output_size(self, start: datetime.date, end: datetime.date) -> str:
        delta = (end - start).days
        return "compact" if delta <= _COMPACT_THRESHOLD_DAYS else "full"

    async def get_stock_ohlcv(
        self,
        symbol: str,
        start: datetime.date,
        end: datetime.date,
    ) -> list[StockOHLCV]:
        """Fetch daily adjusted OHLCV data from AlphaVantage.

        Args:
            symbol: Ticker symbol.
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            List of :class:`~Kaare.models.StockOHLCV` records within [start, end].

        Raises:
            httpx.HTTPError: On network or HTTP errors.
            ValueError: On AlphaVantage API errors.
        """
        data = await self._get(
            {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": symbol,
                "outputsize": self._output_size(start, end),
            }
        )
        time_series = data.get("Time Series (Daily)", {})
        records: list[StockOHLCV] = []
        for date_str, values in time_series.items():
            date = datetime.date.fromisoformat(date_str)
            if not (start <= date <= end):
                continue
            records.append(
                StockOHLCV(
                    symbol=symbol,
                    date=date,
                    open=float(values["1. open"]),
                    high=float(values["2. high"]),
                    low=float(values["3. low"]),
                    close=float(values["5. adjusted close"]),
                    volume=int(values["6. volume"]),
                )
            )
        records.sort(key=lambda r: r.date)
        return records

    async def get_gold_prices(
        self,
        start: datetime.date,
        end: datetime.date,
    ) -> dict[datetime.date, float]:
        """Fetch daily spot gold prices from AlphaVantage.

        Args:
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            Mapping of date → gold price (USD).
        """
        data = await self._get({"function": "GOLD", "interval": "daily"})
        result: dict[datetime.date, float] = {}
        for entry in data.get("data", []):
            date = datetime.date.fromisoformat(entry["date"])
            if not (start <= date <= end):
                continue
            try:
                result[date] = float(entry["value"])
            except (ValueError, KeyError):
                pass
        return result

    async def get_treasury_yields(
        self,
        start: datetime.date,
        end: datetime.date,
    ) -> dict[datetime.date, float]:
        """Fetch daily 10-year US Treasury yields from AlphaVantage.

        Args:
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            Mapping of date → yield (%).
        """
        data = await self._get(
            {
                "function": "TREASURY_YIELD",
                "interval": "daily",
                "maturity": "10year",
            }
        )
        result: dict[datetime.date, float] = {}
        for entry in data.get("data", []):
            date = datetime.date.fromisoformat(entry["date"])
            if not (start <= date <= end):
                continue
            try:
                result[date] = float(entry["value"])
            except (ValueError, KeyError):
                pass
        return result
