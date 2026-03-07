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

_ADD_SENTIMENT_COLUMN = """
ALTER TABLE macro_data
    ADD COLUMN IF NOT EXISTS news_sentiment_score DOUBLE PRECISION;
"""

_CREATE_RAW_NEWS = """
CREATE TABLE IF NOT EXISTS raw_news (
    id              BIGSERIAL PRIMARY KEY,
    date_utc        TIMESTAMPTZ NOT NULL,
    trading_date    DATE NOT NULL,
    text            TEXT NOT NULL,
    text_hash       CHAR(64) NOT NULL,
    dataset_subset  TEXT NOT NULL,
    source          TEXT,
    tickers         TEXT[],
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
"""

_CREATE_RAW_NEWS_INDEXES = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_news_dedup
    ON raw_news (text_hash, trading_date);
CREATE INDEX IF NOT EXISTS idx_raw_news_trading_date
    ON raw_news (trading_date);
CREATE INDEX IF NOT EXISTS idx_raw_news_tickers
    ON raw_news USING GIN (tickers);
"""

_CREATE_ARTICLE_SENTIMENT = """
CREATE TABLE IF NOT EXISTS article_sentiment (
    id              BIGSERIAL PRIMARY KEY,
    raw_news_id     BIGINT NOT NULL REFERENCES raw_news(id),
    positive        REAL NOT NULL,
    negative        REAL NOT NULL,
    neutral         REAL NOT NULL,
    net_score       REAL NOT NULL,
    model_version   TEXT DEFAULT 'ProsusAI/finbert',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_article_sentiment_news_id
    ON article_sentiment (raw_news_id);
"""

_CREATE_DAILY_MARKET_SENTIMENT = """
CREATE TABLE IF NOT EXISTS daily_market_sentiment (
    trading_date    DATE PRIMARY KEY,
    mean_score      REAL NOT NULL,
    median_score    REAL NOT NULL,
    std_score       REAL,
    article_count   INT NOT NULL,
    pct_positive    REAL,
    pct_negative    REAL,
    mean_score_3d   REAL,
    mean_score_7d   REAL,
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
"""

_CREATE_DAILY_TICKER_SENTIMENT = """
CREATE TABLE IF NOT EXISTS daily_ticker_sentiment (
    trading_date    DATE NOT NULL,
    ticker          TEXT NOT NULL,
    mean_score      REAL NOT NULL,
    article_count   INT NOT NULL,
    std_score       REAL,
    pct_positive    REAL,
    pct_negative    REAL,
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (trading_date, ticker)
);
CREATE INDEX IF NOT EXISTS idx_daily_ticker_sentiment_ticker
    ON daily_ticker_sentiment (ticker, trading_date);
"""


async def run_migrations(pool: asyncpg.Pool) -> None:
    """Create all required tables if they do not already exist.

    Args:
        pool: An active asyncpg connection pool.
    """
    async with pool.acquire() as conn:
        await conn.execute(_CREATE_STOCK_OHLCV)
        await conn.execute(_CREATE_MACRO_DATA)
        await conn.execute(_ADD_SENTIMENT_COLUMN)
        await conn.execute(_CREATE_RAW_NEWS)
        await conn.execute(_CREATE_RAW_NEWS_INDEXES)
        await conn.execute(_CREATE_ARTICLE_SENTIMENT)
        await conn.execute(_CREATE_DAILY_MARKET_SENTIMENT)
        await conn.execute(_CREATE_DAILY_TICKER_SENTIMENT)
