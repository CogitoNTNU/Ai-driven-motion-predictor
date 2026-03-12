"""Tools for the stock analyst sub-agent."""

from datetime import datetime
from typing import Optional
import yfinance as yf
from langchain.tools import tool


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
        A tuple of (text_summary, chart_artifact) where text_summary is a brief
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
            "type": f"{chart_type}_chart",  # line_chart, bar_chart, area_chart, pie_chart
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

        return (text_summary, chart_artifact)

    except ValueError as e:
        return (
            f"Error: Invalid date format. Please use YYYY-MM-DD format. Details: {str(e)}",
            None,
        )
    except Exception as e:
        return (f"Error retrieving data for {symbol}: {str(e)}", None)


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
