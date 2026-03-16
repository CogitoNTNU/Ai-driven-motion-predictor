import yfinance as yf
import pandas as pd
import numpy as np


def _normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


def _price_column(data: pd.DataFrame) -> str:
    if "Adj Close" in data.columns:
        return "Adj Close"

    if "Close" in data.columns:
        return "Close"

    raise KeyError("Expected 'Adj Close' or 'Close' column from yfinance data.")


def load_price_data(ticker="", start=""):
    data = yf.download(ticker, start=start)

    if data is None or data.empty:
        raise ValueError(f"No price data returned for ticker '{ticker}'.")

    data = _normalize_columns(data)
    price_col = _price_column(data)

    log_prices = data[price_col].astype("float64").apply(np.log)
    data["return"] = log_prices.diff()

    data = data.dropna()

    return data
