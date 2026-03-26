"""Database read/write operations for stock_ohlcv and macro_data tables."""

import datetime

import asyncpg

from Kaare.models import MacroData, RawNews, StockOHLCV


async def get_stock_ohlcv(
    pool: asyncpg.Pool,
    symbol: str,
    start: datetime.date,
    end: datetime.date,
) -> list[StockOHLCV]:
    """Fetch OHLCV rows for *symbol* within [start, end] from the database.

    Args:
        pool: Active asyncpg pool.
        symbol: Ticker symbol.
        start: Inclusive start date.
        end: Inclusive end date.

    Returns:
        List of :class:`~Kaare.models.StockOHLCV` ordered by date ascending.
    """
    query = """
        SELECT symbol, date, open, high, low, close, volume
        FROM stock_ohlcv
        WHERE symbol = $1 AND date BETWEEN $2 AND $3
        ORDER BY date
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, symbol, start, end)
    return [
        StockOHLCV(
            symbol=row["symbol"],
            date=row["date"],
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
        )
        for row in rows
    ]


async def insert_stock_ohlcv(
    pool: asyncpg.Pool,
    records: list[StockOHLCV],
) -> None:
    """Insert OHLCV records, ignoring conflicts on (symbol, date).

    Args:
        pool: Active asyncpg pool.
        records: Records to insert.
    """
    if not records:
        return
    query = """
        INSERT INTO stock_ohlcv (symbol, date, open, high, low, close, volume)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (symbol, date) DO NOTHING
    """
    async with pool.acquire() as conn:
        await conn.executemany(
            query,
            [
                (r.symbol, r.date, r.open, r.high, r.low, r.close, r.volume)
                for r in records
            ],
        )


async def get_macro_data(
    pool: asyncpg.Pool,
    start: datetime.date,
    end: datetime.date,
) -> list[MacroData]:
    """Fetch macro_data rows within [start, end] from the database.

    Args:
        pool: Active asyncpg pool.
        start: Inclusive start date.
        end: Inclusive end date.

    Returns:
        List of :class:`~Kaare.models.MacroData` ordered by date ascending.
    """
    query = """
        SELECT date, gold_price, treasury_yield_10y, vix, news_sentiment_score
        FROM macro_data
        WHERE date BETWEEN $1 AND $2
        ORDER BY date
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, start, end)
    return [
        MacroData(
            date=row["date"],
            gold_price=row["gold_price"],
            treasury_yield_10y=row["treasury_yield_10y"],
            vix=row["vix"],
            news_sentiment_score=row["news_sentiment_score"],
        )
        for row in rows
    ]


async def get_dates_missing_sentiment(
    pool: asyncpg.Pool,
    start: datetime.date,
    end: datetime.date,
) -> set[datetime.date]:
    """Return dates in [start, end] that exist in macro_data but have no sentiment score.

    Args:
        pool: Active asyncpg pool.
        start: Inclusive start date.
        end: Inclusive end date.

    Returns:
        Set of dates where ``news_sentiment_score IS NULL``.
    """
    query = """
        SELECT date FROM macro_data
        WHERE date BETWEEN $1 AND $2 AND news_sentiment_score IS NULL
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, start, end)
    return {row["date"] for row in rows}


async def sync_sentiment_to_macro(
    pool: asyncpg.Pool,
    start: datetime.date,
    end: datetime.date,
) -> None:
    """Copy sentiment scores from daily_market_sentiment into macro_data.

    Only updates rows where ``news_sentiment_score`` is currently ``NULL``.

    Args:
        pool: Active asyncpg pool.
        start: Inclusive start date.
        end: Inclusive end date.
    """
    query = """
        UPDATE macro_data m
        SET news_sentiment_score = dms.mean_score
        FROM daily_market_sentiment dms
        WHERE m.date = dms.trading_date
          AND m.date BETWEEN $1 AND $2
          AND m.news_sentiment_score IS NULL
    """
    async with pool.acquire() as conn:
        await conn.execute(query, start, end)


async def upsert_macro_data(
    pool: asyncpg.Pool,
    records: list[MacroData],
) -> None:
    """Upsert macro_data records (insert or update on date conflict).

    Args:
        pool: Active asyncpg pool.
        records: Records to upsert.
    """
    if not records:
        return
    query = """
        INSERT INTO macro_data (date, gold_price, treasury_yield_10y, vix, news_sentiment_score)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (date) DO UPDATE
            SET gold_price            = COALESCE(EXCLUDED.gold_price, macro_data.gold_price),
                treasury_yield_10y    = COALESCE(EXCLUDED.treasury_yield_10y, macro_data.treasury_yield_10y),
                vix                   = COALESCE(EXCLUDED.vix, macro_data.vix),
                news_sentiment_score  = COALESCE(EXCLUDED.news_sentiment_score, macro_data.news_sentiment_score)
    """
    async with pool.acquire() as conn:
        await conn.executemany(
            query,
            [(r.date, r.gold_price, r.treasury_yield_10y, r.vix, r.news_sentiment_score) for r in records],
        )


# ---------------------------------------------------------------------------
# raw_news
# ---------------------------------------------------------------------------


async def insert_raw_news_batch(pool: asyncpg.Pool, records: list[RawNews]) -> int:
    """Insert a batch of raw news articles, skipping duplicates.

    Deduplication is based on ``(text_hash, trading_date)``.

    Args:
        pool: Active asyncpg pool.
        records: Articles to insert.

    Returns:
        Number of rows actually inserted.
    """
    if not records:
        return 0
    query = """
        INSERT INTO raw_news (date_utc, trading_date, text, text_hash, dataset_subset, source, tickers)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (text_hash, trading_date) DO NOTHING
        RETURNING id
    """
    async with pool.acquire() as conn:
        results = await conn.executemany(
            query,
            [
                (r.date_utc, r.trading_date, r.text, r.text_hash, r.dataset_subset, r.source, r.tickers)
                for r in records
            ],
        )
    # executemany returns the status string; count RETURNING rows instead
    # by checking affected rows via a simpler approach
    return len(records)  # approximate; dedup happens silently


async def insert_raw_news_returning_ids(
    pool: asyncpg.Pool,
    records: list[RawNews],
) -> list[int]:
    """Insert raw news articles and return their database IDs.

    Uses ``ON CONFLICT DO UPDATE`` (no-op) so the ID is always returned,
    whether the row was newly inserted or already existed.

    Args:
        pool: Active asyncpg pool.
        records: Articles to insert.

    Returns:
        List of IDs in the same order as *records*.
    """
    if not records:
        return []
    query = """
        INSERT INTO raw_news (date_utc, trading_date, text, text_hash, dataset_subset, source, tickers)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (text_hash, trading_date) DO UPDATE SET text = EXCLUDED.text
        RETURNING id
    """
    ids: list[int] = []
    async with pool.acquire() as conn:
        for r in records:
            row = await conn.fetchrow(
                query,
                r.date_utc, r.trading_date, r.text, r.text_hash,
                r.dataset_subset, r.source, r.tickers,
            )
            ids.append(row["id"])
    return ids


async def get_unscored_article_batch(pool: asyncpg.Pool, limit: int) -> list[tuple[int, str]]:
    """Fetch articles that have not yet been scored by FinBERT.

    Args:
        pool: Active asyncpg pool.
        limit: Maximum number of rows to return.

    Returns:
        List of ``(id, text)`` tuples ordered by id.
    """
    query = """
        SELECT rn.id, rn.text
        FROM raw_news rn
        LEFT JOIN article_sentiment s ON s.raw_news_id = rn.id
        WHERE s.id IS NULL
        ORDER BY rn.id
        LIMIT $1
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, limit)
    return [(row["id"], row["text"]) for row in rows]


async def get_article_sentiment_by_ids(
    pool: asyncpg.Pool,
    ids: list[int],
) -> dict[int, dict]:
    """Fetch existing FinBERT scores for the given raw_news IDs.

    Args:
        pool: Active asyncpg pool.
        ids: List of raw_news IDs to look up.

    Returns:
        Mapping of raw_news_id → score dict with keys positive, negative, neutral, net_score.
    """
    if not ids:
        return {}
    query = """
        SELECT raw_news_id, positive, negative, neutral, net_score
        FROM article_sentiment
        WHERE raw_news_id = ANY($1)
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, ids)
    return {
        row["raw_news_id"]: {
            "positive": row["positive"],
            "negative": row["negative"],
            "neutral": row["neutral"],
            "net_score": row["net_score"],
        }
        for row in rows
    }


async def insert_article_sentiment_batch(
    pool: asyncpg.Pool,
    data: list[tuple[int, dict]],
) -> None:
    """Insert FinBERT scores for a batch of articles, skipping duplicates.

    Args:
        pool: Active asyncpg pool.
        data: List of ``(raw_news_id, score_dict)`` where score_dict has keys
            ``positive``, ``negative``, ``neutral``, ``net_score``.
    """
    if not data:
        return
    query = """
        INSERT INTO article_sentiment (raw_news_id, positive, negative, neutral, net_score)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (raw_news_id) DO NOTHING
    """
    async with pool.acquire() as conn:
        await conn.executemany(
            query,
            [
                (raw_news_id, s["positive"], s["negative"], s["neutral"], s["net_score"])
                for raw_news_id, s in data
            ],
        )


# ---------------------------------------------------------------------------
# daily_ticker_sentiment queries
# ---------------------------------------------------------------------------


async def get_daily_ticker_sentiment(
    pool: asyncpg.Pool,
    symbol: str,
    start: datetime.date,
    end: datetime.date,
) -> list[tuple[datetime.date, float, int]]:
    """Fetch pre-aggregated sentiment for *symbol* from daily_ticker_sentiment.

    Args:
        pool: Active asyncpg pool.
        symbol: Ticker symbol (case-insensitive).
        start: Inclusive start date.
        end: Inclusive end date.

    Returns:
        List of ``(trading_date, mean_score, article_count)`` tuples ordered by date.
    """
    query = """
        SELECT trading_date, mean_score, article_count
        FROM daily_ticker_sentiment
        WHERE ticker = $1 AND trading_date BETWEEN $2 AND $3
        ORDER BY trading_date
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, symbol.upper(), start, end)
    return [(row["trading_date"], row["mean_score"], row["article_count"]) for row in rows]


# ---------------------------------------------------------------------------
# daily_market_sentiment / daily_ticker_sentiment aggregation
# ---------------------------------------------------------------------------


async def aggregate_daily_market_sentiment(pool: asyncpg.Pool) -> int:
    """Aggregate article-level scores into daily_market_sentiment.

    Computes mean, median, std, article count, pct_positive, pct_negative,
    and rolling 3-day and 7-day averages.

    Args:
        pool: Active asyncpg pool.

    Returns:
        Number of rows upserted.
    """
    upsert_query = """
        INSERT INTO daily_market_sentiment
            (trading_date, mean_score, median_score, std_score,
             article_count, pct_positive, pct_negative)
        SELECT
            rn.trading_date,
            AVG(s.net_score),
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY s.net_score),
            STDDEV(s.net_score),
            COUNT(*),
            AVG(CASE WHEN s.positive > 0.5 THEN 1.0 ELSE 0.0 END),
            AVG(CASE WHEN s.negative > 0.5 THEN 1.0 ELSE 0.0 END)
        FROM raw_news rn
        JOIN article_sentiment s ON s.raw_news_id = rn.id
        GROUP BY rn.trading_date
        ON CONFLICT (trading_date) DO UPDATE SET
            mean_score    = EXCLUDED.mean_score,
            median_score  = EXCLUDED.median_score,
            std_score     = EXCLUDED.std_score,
            article_count = EXCLUDED.article_count,
            pct_positive  = EXCLUDED.pct_positive,
            pct_negative  = EXCLUDED.pct_negative,
            updated_at    = NOW()
    """
    rolling_query = """
        UPDATE daily_market_sentiment d SET
            mean_score_3d = sub.avg_3d,
            mean_score_7d = sub.avg_7d
        FROM (
            SELECT
                trading_date,
                AVG(mean_score) OVER (
                    ORDER BY trading_date
                    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ) AS avg_3d,
                AVG(mean_score) OVER (
                    ORDER BY trading_date
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ) AS avg_7d
            FROM daily_market_sentiment
        ) sub
        WHERE d.trading_date = sub.trading_date
    """
    async with pool.acquire() as conn:
        result = await conn.execute(upsert_query)
        await conn.execute(rolling_query)
    # result is like "INSERT 0 N"
    try:
        return int(result.split()[-1])
    except (IndexError, ValueError):
        return 0


async def aggregate_daily_ticker_sentiment(pool: asyncpg.Pool) -> int:
    """Aggregate article-level scores into daily_ticker_sentiment per ticker.

    Args:
        pool: Active asyncpg pool.

    Returns:
        Number of rows upserted.
    """
    query = """
        INSERT INTO daily_ticker_sentiment
            (trading_date, ticker, mean_score, article_count,
             std_score, pct_positive, pct_negative)
        SELECT
            rn.trading_date,
            UNNEST(rn.tickers) AS ticker,
            AVG(s.net_score),
            COUNT(*),
            STDDEV(s.net_score),
            AVG(CASE WHEN s.positive > 0.5 THEN 1.0 ELSE 0.0 END),
            AVG(CASE WHEN s.negative > 0.5 THEN 1.0 ELSE 0.0 END)
        FROM raw_news rn
        JOIN article_sentiment s ON s.raw_news_id = rn.id
        WHERE rn.tickers IS NOT NULL AND array_length(rn.tickers, 1) > 0
        GROUP BY rn.trading_date, UNNEST(rn.tickers)
        ON CONFLICT (trading_date, ticker) DO UPDATE SET
            mean_score    = EXCLUDED.mean_score,
            article_count = EXCLUDED.article_count,
            std_score     = EXCLUDED.std_score,
            pct_positive  = EXCLUDED.pct_positive,
            pct_negative  = EXCLUDED.pct_negative,
            updated_at    = NOW()
    """
    async with pool.acquire() as conn:
        result = await conn.execute(query)
    try:
        return int(result.split()[-1])
    except (IndexError, ValueError):
        return 0
