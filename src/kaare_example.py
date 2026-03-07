"""Example usage of the KaareClient."""

import asyncio
import datetime

from Kaare import KaareClient


async def main() -> None:
    start = datetime.date(2025, 1, 6)
    end = datetime.date(2025, 1, 31)

    async with KaareClient() as client:
        # Fetch OHLCV data for one or more tickers
        aapl = await client.get_stock_ohlcv("AAPL", start, end)
        print(f"AAPL — {len(aapl)} trading days")
        for row in aapl[:3]:
            print(f"  {row.date}  close={row.close:.2f}  volume={row.volume:,}")
        print("  ...")

        nvda = await client.get_stock_ohlcv("NVDA", start, end)
        print(f"\nNVDA — {len(nvda)} trading days")
        for row in nvda[:3]:
            print(f"  {row.date}  close={row.close:.2f}  volume={row.volume:,}")
        print("  ...")

        # Fetch macro data (gold price + 10-year Treasury yield)
        macro = await client.get_macro_data(start, end)
        print(f"\nMacro — {len(macro)} days")
        for row in macro[:3]:
            gold = f"{row.gold_price:.2f}" if row.gold_price else "N/A"
            yld = f"{row.treasury_yield_10y:.3f}%" if row.treasury_yield_10y else "N/A"
            print(f"  {row.date}  gold={gold}  10Y={yld}")
        print("  ...")

        sentiment = await client.get


asyncio.run(main())
