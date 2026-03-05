"""Database schema migrations — run once at startup."""

import asyncpg

_CREATE_STOCK_OHLCV = """
CREATE TABLE IF NOT EXISTS stock_ohlcv (
    symbol TEXT            NOT NULL,
    date   DATE            NOT NULL,
    open   DOUBLE PRECISION,
    high   DOUBLE PRECISION,
    low    DOUBLE PRECISION,
    close  DOUBLE PRECISION,
    volume BIGINT,
    PRIMARY KEY (symbol, date)
);
"""

_CREATE_MACRO_DATA = """
CREATE TABLE IF NOT EXISTS macro_data (
    date                DATE PRIMARY KEY,
    gold_price          DOUBLE PRECISION,
    treasury_yield_10y  DOUBLE PRECISION
);
"""


async def run_migrations(pool: asyncpg.Pool) -> None:
    """Create all required tables if they do not already exist.

    Args:
        pool: An active asyncpg connection pool.
    """
    async with pool.acquire() as conn:
        await conn.execute(_CREATE_STOCK_OHLCV)
        await conn.execute(_CREATE_MACRO_DATA)
