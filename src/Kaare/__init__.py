"""Kaare — async cache-first data layer for stock and macro data.

Example::

    import asyncio
    import datetime
    from Kaare import KaareClient

    async def main():
        async with KaareClient() as client:
            stocks = await client.get_stock_ohlcv(
                "AAPL",
                datetime.date(2024, 1, 1),
                datetime.date(2024, 3, 1),
            )
            macro = await client.get_macro_data(
                datetime.date(2024, 1, 1),
                datetime.date(2024, 3, 1),
            )

    asyncio.run(main())
"""

from Kaare.client import KaareClient
from Kaare.models import MacroData, StockOHLCV

__all__ = ["KaareClient", "MacroData", "StockOHLCV"]
