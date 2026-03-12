"""Tests for KaareClient.get_stock_news_sentiment."""

import asyncio
import datetime
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from Kaare.client import KaareClient
from Kaare.models import NewsSentimentResult, RawNews


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


TODAY = datetime.date(2025, 1, 10)
START = TODAY - datetime.timedelta(days=7)


def _make_article(symbol: str, trading_date: datetime.date, headline: str) -> RawNews:
    text = headline
    return RawNews(
        date_utc=datetime.datetime(trading_date.year, trading_date.month, trading_date.day, tzinfo=datetime.timezone.utc),
        trading_date=trading_date,
        text=text,
        text_hash=hashlib.sha256(text.encode()).hexdigest(),
        dataset_subset="finnhub",
        source="TestSource",
        tickers=[symbol],
    )


_SAMPLE_ARTICLES = [
    _make_article("AAPL", datetime.date(2025, 1, 8), "Apple reports record revenue."),
    _make_article("AAPL", datetime.date(2025, 1, 9), "Apple faces antitrust scrutiny."),
]

_SAMPLE_SCORES = [
    {"positive": 0.8, "negative": 0.1, "neutral": 0.1, "net_score": 0.7},
    {"positive": 0.1, "negative": 0.7, "neutral": 0.2, "net_score": -0.6},
]


def _make_client(articles: list, scores: list) -> KaareClient:
    """Return a KaareClient with all external dependencies mocked."""
    with patch("Kaare.client.FinnhubProvider"), patch("Kaare.client.FinBERTAnalyzer"):
        client = KaareClient()

    client._finnhub = MagicMock()
    client._finnhub.fetch_raw_news = AsyncMock(return_value=articles)

    client._analyzer = MagicMock()
    client._analyzer.score_articles = AsyncMock(return_value=scores)

    # Mock pool / DB so repository calls don't hit a real database
    client._pool = MagicMock()
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch("Kaare.client.repository.insert_raw_news_returning_ids", new_callable=AsyncMock, return_value=[1, 2])
@patch("Kaare.client.repository.insert_article_sentiment_batch", new_callable=AsyncMock)
@patch("Kaare.client.repository.aggregate_daily_ticker_sentiment", new_callable=AsyncMock)
def test_returns_sentiment_result(mock_agg, mock_sent, mock_ids):
    client = _make_client(_SAMPLE_ARTICLES, _SAMPLE_SCORES)
    result = _run(client.get_stock_news_sentiment("AAPL", start=START, end=TODAY))

    assert isinstance(result, NewsSentimentResult)
    assert result.symbol == "AAPL"
    assert result.start == START
    assert result.end == TODAY


@patch("Kaare.client.repository.insert_raw_news_returning_ids", new_callable=AsyncMock, return_value=[1, 2])
@patch("Kaare.client.repository.insert_article_sentiment_batch", new_callable=AsyncMock)
@patch("Kaare.client.repository.aggregate_daily_ticker_sentiment", new_callable=AsyncMock)
def test_article_count(mock_agg, mock_sent, mock_ids):
    client = _make_client(_SAMPLE_ARTICLES, _SAMPLE_SCORES)
    result = _run(client.get_stock_news_sentiment("AAPL", start=START, end=TODAY))

    assert result.article_count == 2


@patch("Kaare.client.repository.insert_raw_news_returning_ids", new_callable=AsyncMock, return_value=[1, 2])
@patch("Kaare.client.repository.insert_article_sentiment_batch", new_callable=AsyncMock)
@patch("Kaare.client.repository.aggregate_daily_ticker_sentiment", new_callable=AsyncMock)
def test_daily_scores_keys(mock_agg, mock_sent, mock_ids):
    client = _make_client(_SAMPLE_ARTICLES, _SAMPLE_SCORES)
    result = _run(client.get_stock_news_sentiment("AAPL", start=START, end=TODAY))

    expected_dates = {a.trading_date for a in _SAMPLE_ARTICLES}
    assert set(result.daily_scores.keys()) == expected_dates


@patch("Kaare.client.repository.insert_raw_news_returning_ids", new_callable=AsyncMock, return_value=[1, 2])
@patch("Kaare.client.repository.insert_article_sentiment_batch", new_callable=AsyncMock)
@patch("Kaare.client.repository.aggregate_daily_ticker_sentiment", new_callable=AsyncMock)
def test_avg_score_is_mean_of_net_scores(mock_agg, mock_sent, mock_ids):
    client = _make_client(_SAMPLE_ARTICLES, _SAMPLE_SCORES)
    result = _run(client.get_stock_news_sentiment("AAPL", start=START, end=TODAY))

    expected = (0.7 + -0.6) / 2
    assert result.avg_score == pytest.approx(expected)


@patch("Kaare.client.repository.insert_raw_news_returning_ids", new_callable=AsyncMock, return_value=[1, 2])
@patch("Kaare.client.repository.insert_article_sentiment_batch", new_callable=AsyncMock)
@patch("Kaare.client.repository.aggregate_daily_ticker_sentiment", new_callable=AsyncMock)
def test_db_functions_called(mock_agg, mock_sent, mock_ids):
    client = _make_client(_SAMPLE_ARTICLES, _SAMPLE_SCORES)
    _run(client.get_stock_news_sentiment("AAPL", start=START, end=TODAY))

    mock_ids.assert_called_once()
    mock_sent.assert_called_once()
    mock_agg.assert_called_once()


def test_no_news_returns_zero_score_and_skips_db():
    client = _make_client([], [])
    with patch("Kaare.client.repository.insert_raw_news_returning_ids") as mock_ids:
        result = _run(client.get_stock_news_sentiment("AAPL", start=START, end=TODAY))
        mock_ids.assert_not_called()

    assert result.article_count == 0
    assert result.avg_score == 0.0
    assert result.daily_scores == {}


def test_default_date_range_uses_last_7_days():
    client = _make_client([], [])
    result = _run(client.get_stock_news_sentiment("TSLA"))

    today = datetime.date.today()
    assert result.end == today
    assert result.start == today - datetime.timedelta(days=7)


@patch("Kaare.client.repository.insert_raw_news_returning_ids", new_callable=AsyncMock, return_value=[1, 2])
@patch("Kaare.client.repository.insert_article_sentiment_batch", new_callable=AsyncMock)
@patch("Kaare.client.repository.aggregate_daily_ticker_sentiment", new_callable=AsyncMock)
def test_provider_called_with_correct_symbol(mock_agg, mock_sent, mock_ids):
    client = _make_client(_SAMPLE_ARTICLES, _SAMPLE_SCORES)
    _run(client.get_stock_news_sentiment("MSFT", start=START, end=TODAY))

    client._finnhub.fetch_raw_news.assert_called_once_with("MSFT", START, TODAY)
