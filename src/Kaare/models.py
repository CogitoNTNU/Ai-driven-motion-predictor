"""Domain models for the Kaare data layer."""

import datetime
from dataclasses import dataclass, field


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
        news_sentiment_score: Daily market news sentiment in [-1.0, 1.0] computed by
            FinBERT, or ``None`` if not yet computed.
    """

    date: datetime.date
    gold_price: float | None
    treasury_yield_10y: float | None
    vix: float | None = None
    news_sentiment_score: float | None = None


@dataclass
class RawNews:
    """A single raw news article from the HuggingFace dataset.

    Attributes:
        date_utc: Publication timestamp in UTC.
        trading_date: NYSE-aligned trading date.
        text: Article text (headline + body).
        text_hash: SHA-256 of text for deduplication.
        dataset_subset: Source subset name (e.g. ``"fnspid_news"``).
        source: Publisher name, or ``None`` if unknown.
        tickers: List of associated ticker symbols.
        id: Database-assigned ID (``None`` before insertion).
    """

    date_utc: datetime.datetime
    trading_date: datetime.date
    text: str
    text_hash: str
    dataset_subset: str
    source: str | None
    tickers: list[str] = field(default_factory=list)
    id: int | None = None


@dataclass
class ArticleSentiment:
    """FinBERT sentiment scores for a single article.

    Attributes:
        raw_news_id: FK to ``raw_news.id``.
        positive: Probability of positive sentiment.
        negative: Probability of negative sentiment.
        neutral: Probability of neutral sentiment.
        net_score: ``positive - negative``, range ``[-1, 1]``.
    """

    raw_news_id: int
    positive: float
    negative: float
    neutral: float
    net_score: float


@dataclass
class DailyMarketSentiment:
    """Aggregated market-wide sentiment for a single trading day.

    Attributes:
        trading_date: The trading day.
        mean_score: Average net sentiment across all articles.
        median_score: Median net sentiment.
        article_count: Number of articles scored.
        std_score: Standard deviation of net scores.
        pct_positive: Fraction of articles with positive probability > 0.5.
        pct_negative: Fraction of articles with negative probability > 0.5.
        mean_score_3d: 3-day rolling average of mean_score.
        mean_score_7d: 7-day rolling average of mean_score.
    """

    trading_date: datetime.date
    mean_score: float
    median_score: float
    article_count: int
    std_score: float | None = None
    pct_positive: float | None = None
    pct_negative: float | None = None
    mean_score_3d: float | None = None
    mean_score_7d: float | None = None


@dataclass
class DailyTickerSentiment:
    """Aggregated per-ticker sentiment for a single trading day.

    Attributes:
        trading_date: The trading day.
        ticker: Ticker symbol.
        mean_score: Average net sentiment for this ticker.
        article_count: Number of articles tagged with this ticker.
        std_score: Standard deviation of net scores.
        pct_positive: Fraction of articles with positive probability > 0.5.
        pct_negative: Fraction of articles with negative probability > 0.5.
    """

    trading_date: datetime.date
    ticker: str
    mean_score: float
    article_count: int
    std_score: float | None = None
    pct_positive: float | None = None
    pct_negative: float | None = None
