"""Quick smoke test for KaareClient against the remote DB."""

import asyncio
import datetime
import logging

logging.basicConfig(level=logging.INFO)

import sys
sys.path.insert(0, ".")

from Kaare.client import KaareClient


async def main() -> None:
    end = datetime.date(2025, 1, 10)
    start = datetime.date(2025, 1, 6)

    print(f"Connecting to DB and fetching AAPL {start} → {end} ...")
    async with KaareClient() as client:
        stocks = await client.get_stock_ohlcv("AAPL", start, end)
        print(f"Got {len(stocks)} stock rows:")
        for row in stocks:
            print(f"  {row}")

        print("\nFetching macro data ...")
        macro = await client.get_macro_data(start, end)
        print(f"Got {len(macro)} macro rows:")
        for row in macro:
            print(f"  {row}")

        print("\nFetching news sentiment for AAPL ...")
        sentiment = await client.get_stock_news_sentiment("AAPL", start, end)
        print(f"  symbol:        {sentiment.symbol}")
        print(f"  article_count: {sentiment.article_count}")
        print(f"  avg_score:     {sentiment.avg_score:.4f}")
        print(f"  daily_scores:")
        for date, score in sorted(sentiment.daily_scores.items()):
            print(f"    {date}: {score:.4f}")


asyncio.run(main())
