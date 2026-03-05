"""Database read/write operations for stock_ohlcv and macro_data tables."""

import datetime

import asyncpg

from Kaare.models import MacroData, StockOHLCV


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
        SELECT date, gold_price, treasury_yield_10y
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
        )
        for row in rows
    ]


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
        INSERT INTO macro_data (date, gold_price, treasury_yield_10y)
        VALUES ($1, $2, $3)
        ON CONFLICT (date) DO UPDATE
            SET gold_price         = COALESCE(EXCLUDED.gold_price, macro_data.gold_price),
                treasury_yield_10y = COALESCE(EXCLUDED.treasury_yield_10y, macro_data.treasury_yield_10y)
    """
    async with pool.acquire() as conn:
        await conn.executemany(
            query,
            [(r.date, r.gold_price, r.treasury_yield_10y) for r in records],
        )
