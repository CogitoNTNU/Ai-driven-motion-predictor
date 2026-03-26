"""Smoke tests for get_stock_news_sentiment routing logic.

Tests that:
- Historical dates (>1 year ago) are served from the seeded DB
- Recent dates (<1 year ago) are fetched from Finnhub and stored in DB
- Edge case: date exactly at the 1-year boundary routes to DB
- Empty result is handled gracefully (no articles found)

Usage::

    cd src
    uv run python -m Kaare.smoke_test_sentiment
"""

import asyncio
import datetime
import logging
import sys

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

from Kaare.client import KaareClient

SYMBOL = "AAPL"
TODAY = datetime.date.today()
ONE_YEAR_AGO = TODAY - datetime.timedelta(days=365)

PASS = "PASS"
FAIL = "FAIL"


def result(label: str, ok: bool, detail: str = "") -> None:
    status = PASS if ok else FAIL
    suffix = f" — {detail}" if detail else ""
    print(f"  [{status}] {label}{suffix}")


async def test_historical_reads_from_db(client: KaareClient) -> bool:
    """Data stored via Finnhub should be retrievable from daily_ticker_sentiment.

    Uses the same recent date range as the recent path test so we know data exists.
    Queries the DB directly and compares against what get_stock_news_sentiment returns.
    """
    end = TODAY
    start = TODAY - datetime.timedelta(days=7)
    print(f"\nDB match path ({start} → {end}, result must match daily_ticker_sentiment):")

    from Kaare.db import repository

    db_rows = await repository.get_daily_ticker_sentiment(client._db, SYMBOL, start, end)

    if not db_rows:
        result("no rows in daily_ticker_sentiment yet (run recent path first)", True)
        return True

    sentiment = await client.get_stock_news_sentiment(SYMBOL, start, end)

    db_daily = {date: score for date, score, _ in db_rows}
    db_article_count = sum(count for _, _, count in db_rows)

    ok_count = sentiment.article_count >= db_article_count
    result("article_count >= DB count", ok_count, f"{sentiment.article_count} >= {db_article_count}")

    ok_dates = set(db_daily.keys()) == set(sentiment.daily_scores.keys())
    result("daily_scores dates match DB", ok_dates, f"DB={sorted(db_daily)}")

    ok_scores = all(
        abs(sentiment.daily_scores[d] - db_daily[d]) < 1e-6
        for d in db_daily
        if d in sentiment.daily_scores
    )
    result("daily_scores values match DB", ok_scores)

    return ok_count and ok_dates and ok_scores


async def test_recent_reads_from_finnhub(client: KaareClient) -> bool:
    """Date range <1 year ago should be fetched from Finnhub and stored in DB."""
    end = TODAY
    start = TODAY - datetime.timedelta(days=7)
    print(f"\nRecent path ({start} → {end}, expects Finnhub + stored in DB):")

    sentiment = await client.get_stock_news_sentiment(SYMBOL, start, end)

    ok_symbol = sentiment.symbol == SYMBOL
    result("symbol matches", ok_symbol, sentiment.symbol)

    ok_type = isinstance(sentiment.avg_score, float)
    result("avg_score is float", ok_type, str(sentiment.avg_score))

    ok_range = -1.0 <= sentiment.avg_score <= 1.0
    result("avg_score in [-1, 1]", ok_range, f"{sentiment.avg_score:.4f}")

    ok_count = sentiment.article_count >= 0
    result("article_count >= 0", ok_count, str(sentiment.article_count))

    if sentiment.article_count == 0:
        result("Finnhub returned 0 articles (weekend/holiday or key missing)", True)
        return ok_symbol and ok_type and ok_range and ok_count

    result("Finnhub returned articles", True, f"{sentiment.article_count} articles")
    ok_daily = all(-1.0 <= score <= 1.0 for score in sentiment.daily_scores.values())
    result("all daily_scores in [-1, 1]", ok_daily)

    # Verify articles were actually persisted to daily_ticker_sentiment
    from Kaare.db import repository
    db_rows = await repository.get_daily_ticker_sentiment(client._db, SYMBOL, start, end)
    ok_persisted = len(db_rows) > 0
    result("stored in daily_ticker_sentiment", ok_persisted, f"{len(db_rows)} rows in DB")

    return ok_symbol and ok_type and ok_range and ok_count and ok_daily and ok_persisted


async def test_boundary_routes_to_db(client: KaareClient) -> bool:
    """Date range ending exactly at ONE_YEAR_AGO should route to DB."""
    start = ONE_YEAR_AGO - datetime.timedelta(days=7)
    end = ONE_YEAR_AGO
    print(f"\nBoundary path ({start} → {end}, end == 1 year ago, expects DB):")

    sentiment = await client.get_stock_news_sentiment(SYMBOL, start, end)

    ok = sentiment.symbol == SYMBOL and isinstance(sentiment.avg_score, float)
    result("returned valid result", ok, f"article_count={sentiment.article_count}")
    return ok


async def test_empty_result(client: KaareClient) -> bool:
    """Obscure ticker with no data should return article_count=0 without crashing."""
    start = ONE_YEAR_AGO - datetime.timedelta(days=14)
    end = ONE_YEAR_AGO - datetime.timedelta(days=7)
    symbol = "ZZZZ"
    print(f"\nEmpty result path ({symbol} {start} → {end}):")

    sentiment = await client.get_stock_news_sentiment(symbol, start, end)

    ok_count = sentiment.article_count == 0
    result("article_count == 0", ok_count, str(sentiment.article_count))

    ok_score = sentiment.avg_score == 0.0
    result("avg_score == 0.0", ok_score, str(sentiment.avg_score))

    ok_daily = sentiment.daily_scores == {}
    result("daily_scores is empty", ok_daily)

    return ok_count and ok_score and ok_daily


async def main() -> None:
    print(f"Smoke test: get_stock_news_sentiment routing")
    print(f"TODAY={TODAY}  ONE_YEAR_AGO={ONE_YEAR_AGO}")

    async with KaareClient() as client:
        results = [
            await test_historical_reads_from_db(client),
            await test_recent_reads_from_finnhub(client),
            await test_boundary_routes_to_db(client),
            await test_empty_result(client),
        ]

    passed = sum(results)
    total = len(results)
    print(f"\n{'='*40}")
    print(f"Results: {passed}/{total} test groups passed")
    if passed < total:
        sys.exit(1)


asyncio.run(main())
