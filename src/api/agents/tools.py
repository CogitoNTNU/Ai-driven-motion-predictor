"""Tools for the stock analyst sub-agent."""

import asyncio
from datetime import datetime, timedelta
from typing import Optional
import yfinance as yf
from langchain.tools import tool

from Kaare import KaareClient


@tool(response_format="content_and_artifact")
def get_stock_growth(
    symbol: str,
    start_date: str,
    end_date: Optional[str] = None,
    chart_type: Optional[str] = "line",
):
    """Get stock price growth for a given symbol and date range.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'MSFT')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (defaults to today if not provided)
        chart_type: Type of chart to generate (line, bar, area, pie)

    Returns:
        A tuple of text_summary, chart_artifact where text_summary is a brief
        summary for the LLM context and chart_artifact contains chart data.
    """
    try:
        # Set default end_date to today if not provided
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Validate dates
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        if end < start:
            return (
                f"Error: End date ({end_date}) must be after start date ({start_date})",
                None,
            )

        # Download stock data
        ticker = yf.Ticker(symbol)

        # Get historical data for the date range
        hist = ticker.history(start=start_date, end=end_date)

        if hist.empty:
            return (
                f"Error: No data found for {symbol} between {start_date} and {end_date}",
                None,
            )

        # Get first and last available trading days
        first_day = hist.index[0]
        last_day = hist.index[-1]

        start_price = hist["Close"].iloc[0]
        end_price = hist["Close"].iloc[-1]

        # Calculate growth metrics
        absolute_growth = end_price - start_price
        percentage_growth = (absolute_growth / start_price) * 100

        # Count trading days
        trading_days = len(hist)

        # Prepare chart data - sample data points to avoid too many points
        chart_data = []
        if len(hist) <= 30:
            # Use all data points if 30 or fewer
            for idx, row in hist.iterrows():
                chart_data.append(
                    {
                        "date": idx.strftime("%Y-%m-%d"),
                        "price": round(float(row["Close"]), 2),
                    }
                )
        else:
            # Sample approximately 30 points evenly distributed
            step = max(1, len(hist) // 30)
            for i in range(0, len(hist), step):
                idx = hist.index[i]
                row = hist.iloc[i]
                chart_data.append(
                    {
                        "date": idx.strftime("%Y-%m-%d"),
                        "price": round(float(row["Close"]), 2),
                    }
                )

        # Create chart artifact for AI SDK v5 tool invocation
        chart_id = (
            f"{symbol.lower()}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
        )
        chart_artifact = {
            "tool_name": "get_stock_growth",  # For AI SDK tool routing
            "chart_id": chart_id,
            "symbol": symbol.upper(),
            "type": f"{chart_type}_chart",
            # line_chart, bar_chart, area_chart, pie_chart
            "data": chart_data,
            "metadata": {
                "start_date": first_day.strftime("%Y-%m-%d"),
                "end_date": last_day.strftime("%Y-%m-%d"),
                "start_price": round(float(start_price), 2),
                "end_price": round(float(end_price), 2),
                "absolute_growth": round(float(absolute_growth), 2),
                "percentage_growth": round(float(percentage_growth), 2),
                "trading_days": trading_days,
            },
        }

        # Brief text summary for LLM context (keeps context window small)
        text_summary = (
            f"Stock Analysis for {symbol.upper()}\n"
            f"Period: {first_day.strftime('%Y-%m-%d')} to {last_day.strftime('%Y-%m-%d')} "
            f"({trading_days} trading days)\n"
            f"Growth: {percentage_growth:+.2f}% (${float(start_price):.2f} → ${float(end_price):.2f})\n"
            f"[Chart: {symbol.upper()}]"
        )

        return text_summary, chart_artifact

    except ValueError as e:
        return (
            f"Error: Invalid date format. Please use YYYY-MM-DD format. Details: {str(e)}",
            None,
        )
    except Exception as e:
        return f"Error retrieving data for {symbol}: {str(e)}", None


@tool
def get_current_price(symbol: str) -> str:
    """Get the current/latest stock price for a given symbol.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'MSFT')

    Returns:
        Current stock price and basic information.
    """
    try:
        ticker = yf.Ticker(symbol)

        # Get current info
        info = ticker.info
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        previous_close = info.get("previousClose")

        if current_price is None:
            # Try getting from history
            hist = ticker.history(period="1d")
            if not hist.empty:
                current_price = hist["Close"].iloc[-1]

        if current_price is None:
            return f"Error: Could not retrieve current price for {symbol}"

        # Calculate daily change
        if previous_close:
            change = current_price - previous_close
            change_pct = (change / previous_close) * 100
            change_str = f"{change:+.2f} ({change_pct:+.2f}%)"
        else:
            change_str = "N/A"

        company_name = info.get("longName", symbol.upper())

        return f"""{company_name} ({symbol.upper()})
Current Price: ${current_price:.2f}
Daily Change: {change_str}

Market Data:
- 52 Week High: ${info.get("fiftyTwoWeekHigh", "N/A")}
- 52 Week Low: ${info.get("fiftyTwoWeekLow", "N/A")}
- Market Cap: ${f"{info.get('marketCap'):,}" if info.get("marketCap") is not None else "N/A"}"""

    except Exception as e:
        return f"Error retrieving current price for {symbol}: {str(e)}"


@tool(response_format="content_and_artifact")
def get_stock_news_sentiment(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """Get news sentiment analysis for a stock using Kaare's FinBERT pipeline.

    Analyzes recent news articles for the given stock symbol using FinBERT sentiment
    analysis. Returns an overall sentiment score and daily breakdown.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'MSFT')
        start_date: Start date in YYYY-MM-DD format (defaults to 7 days ago if not provided)
        end_date: End date in YYYY-MM-DD format (defaults to today if not provided)

    Returns:
        A tuple of text_summary, chart_artifact where text_summary contains the
        sentiment analysis results and chart_artifact contains sentiment trend data.
    """
    try:
        # Set default dates if not provided
        today = datetime.now().date()
        if end_date is None:
            end_date = today.strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")

        # Parse dates
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()

        if end < start:
            return (
                f"Error: End date ({end_date}) must be after start date ({start_date})",
                None,
            )

        # Run async KaareClient method
        async def _fetch_sentiment():
            async with KaareClient() as client:
                return await client.get_stock_news_sentiment(symbol, start, end)

        result = asyncio.run(_fetch_sentiment())

        if result.article_count == 0:
            return (
                f"No news articles found for {symbol.upper()} between {start_date} and {end_date}.",
                None,
            )

        # Prepare chart data for sentiment over time
        chart_data = []
        if result.daily_scores:
            for date_obj, score in sorted(result.daily_scores.items()):
                chart_data.append(
                    {
                        "date": date_obj.strftime("%Y-%m-%d"),
                        "sentiment": round(float(score), 4),
                    }
                )
        else:
            # Single data point for overall average
            chart_data.append(
                {
                    "date": start_date,
                    "sentiment": round(float(result.avg_score), 4),
                }
            )

        # Create chart artifact
        chart_id = f"sentiment_{symbol.lower()}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
        chart_artifact = {
            "tool_name": "get_stock_news_sentiment",
            "chart_id": chart_id,
            "symbol": symbol.upper(),
            "type": "line_chart",
            "data": chart_data,
            "metadata": {
                "start_date": start_date,
                "end_date": end_date,
                "symbol": symbol.upper(),
                "article_count": result.article_count,
                "avg_sentiment": round(float(result.avg_score), 4),
                "sentiment_label": _get_sentiment_label(result.avg_score),
            },
        }

        # Build text summary
        sentiment_label = _get_sentiment_label(result.avg_score)
        daily_breakdown = ""
        if len(result.daily_scores) > 1:
            daily_breakdown = "\n\nDaily Sentiment Breakdown:\n"
            for date_obj, score in sorted(result.daily_scores.items()):
                daily_label = _get_sentiment_label(score)
                daily_breakdown += (
                    f"- {date_obj.strftime('%Y-%m-%d')}: {score:+.4f} ({daily_label})\n"
                )

        text_summary = (
            f"News Sentiment Analysis for {symbol.upper()}\n"
            f"Period: {start_date} to {end_date}\n"
            f"Articles Analyzed: {result.article_count}\n"
            f"Overall Sentiment: {result.avg_score:+.4f} ({sentiment_label})\n"
            f"\nInterpretation:\n"
            f"- Score range: -1.0 (very negative) to +1.0 (very positive)\n"
            f"- Current score indicates {sentiment_label.lower()} market sentiment"
            f"{daily_breakdown}"
        )

        return text_summary, chart_artifact

    except ValueError as e:
        return (
            f"Error: Invalid date format. Please use YYYY-MM-DD format. Details: {str(e)}",
            None,
        )
    except Exception as e:
        return (
            f"Error retrieving sentiment data for {symbol}: {str(e)}",
            None,
        )


def _get_sentiment_label(score: float) -> str:
    """Convert sentiment score to human-readable label."""
    if score >= 0.5:
        return "Very Positive"
    elif score >= 0.1:
        return "Positive"
    elif score > -0.1:
        return "Neutral"
    elif score > -0.5:
        return "Negative"
    else:
        return "Very Negative"
