import datetime as dt
from collections.abc import Sequence
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Kaare import KaareClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

async def load_stock_df(
    symbol: str,
    start: dt.date,
    end: dt.date,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Creates dataframe from kaare client

    Args:
        symbol (str): ticker
        start (dt.date): start date
        end (dt.date): end date

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: total df, stock df, macro df, sentiment df
    """
    async with KaareClient() as client:
        rows = await client.get_stock_ohlcv(symbol, start, end)
        rows_sentiment = await client.get_stock_news_sentiment(symbol, start, end)
        rows_macro = await client.get_macro_data(start, end)
    
    stock_df = pd.DataFrame(
        {
            "date": [r.date for r in rows],
            
            "open": [r.open for r in rows],
            "close": [r.close for r in rows],
            "low": [r.low for r in rows],
            "high": [r.high for r in rows],
            "volume": [r.volume for r in rows]
        }
    )    
    stock_df = stock_df.dropna(subset = ["close"]).reset_index(drop = True)
    
    macro_df = pd.DataFrame(
        {
            "date": [r.date for r in rows_macro],
            "gold_price": [r.gold_price for r in rows_macro],
            "treasury_yield_10y": [r.treasury_yield_10y for r in rows_macro],
            "vix": [r.vix for r in rows_macro]
        }
    )
    
    sentiment_df = pd.DataFrame(
        {
            "date": list(rows_sentiment.daily_scores.keys()),
            "sentiment_score": list(rows_sentiment.daily_scores.values())
        }
    )
    
    for d in (stock_df, macro_df, sentiment_df):
        d["date"] = pd.to_datetime(d["date"])
    
    total_df = (
        stock_df
        .merge(macro_df, on = "date", how = "left")
        .merge(sentiment_df, on = "date", how = "left")
        .sort_values("date")
        .reset_index(drop = True)
    )

    return total_df, stock_df, macro_df, sentiment_df

async def build_features(
    base_df: pd.DataFrame,
    horizon: int = 5,
    decision_time: str = "close",
) -> pd.DataFrame:
    """

    Args:
        base_df (pd.DataFrame): base df from kaareclient
        horizon (int, optional): Prediction window. Defaults to 5.
        decision_time (str, optional): column to predict on. Defaults to "close".

    Raises:
        KeyError: No decision_time column

    Returns:
        pd.DataFrame: dataframe with features
    """
    df = base_df.copy()
    
    if decision_time not in df.columns:
        raise KeyError(f"Column '{decision_time}' is missing from input DataFrame.")
    
    df["target"] = df[decision_time].shift(-horizon)/df[decision_time] - 1
    
    #classic feaatures
    df["ret_1d"] = df[decision_time].pct_change(1)
    df["ret_3d"] = df[decision_time].pct_change(3)
    df["ret_5d"] = df[decision_time].pct_change(5)
    df["ret_10d"] = df[decision_time].pct_change(10)
    df["ret_21d"] = df[decision_time].pct_change(21)
    
    df["h1_range"] = (df["high"] - df["low"]) / df[decision_time]
    df["oc_range"] = (df[decision_time] - df["open"]) / df["open"]
    df["gap"] = (df["open"] - df[decision_time].shift(1)) / df[decision_time].shift(1)
    
    df["volat_5d"] = df[decision_time].rolling(5).std()
    df["volat_10d"] = df[decision_time].rolling(10).std()
    df["volat_21d"] = df[decision_time].rolling(21).std()
    df["volat_ratio"] =df["volat_5d"]/df["volat_21d"] 
    
    df["sma_5"] = df[decision_time].rolling(5).mean()
    df["sma_20"] = df[decision_time].rolling(20).mean()
    df["sma_50"] = df[decision_time].rolling(50).mean()
    df["decision_time_vs_sma5"] = df[decision_time] / df["sma_5"]
    df["decision_time_vs_sma20"] = df[decision_time] / df["sma_20"]
    df["decision_time_vs_sma50"] = df[decision_time] / df["sma_50"]
    
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio20"] = df["volume"] / df["volume_sma_20"]
    
    #macro
    df["sentiment_ma3"] = df["sentiment_score"].rolling(3).mean()
    df["sentiment_ma7"] = df["sentiment_score"].rolling(7).mean()
    df["sentiment_ma21"] = df["sentiment_score"].rolling(21).mean()
    
    df["sentiment_chg1"] = df["sentiment_score"].diff(1)
    df["sentiment_cgg3"] = df["sentiment_score"].diff(3)
    df["sentiment_chg7"] = df["sentiment_score"].diff(7)
    
    df["sentiment_ratio_volat3"] = df["sentiment_score"].rolling(3).std()
    df["sentiment_ratio_volat7"] = df["sentiment_score"].rolling(7).std()
    df["sentiment_ratio_volat21"] = df["sentiment_score"].rolling(21).std()
    
    df["sentiment_vs_ma7"] = df["sentiment_score"] - df["sentiment_ma7"]
    
    df["price_sentiment_div"] = df["ret_1d"] - df["sentiment_score"]
    
    #macro features
    df["gold_ret_1d"] = df["gold_price"].pct_change(1)
    df["yield_chg_1d"] = df["treasury_yield_10y"].diff(1)
    df["vix_chg_1d"] = df["vix"].diff(1)
    
    df["vix_ma10"] = df["vix"].rolling(10).mean()
    df["vix_vs_ma"] = df["vix"] / df["vix_ma10"] - 1
    
    df["risk_off"] = df["vix"] * df["treasury_yield_10y"]
    
    #calendaer features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["is_quarter_end"] = df["date"].dt.is_quarter_end.astype(int)

    # Rolling and shift operations create edge NaNs by construction.
    df = df.dropna().reset_index(drop = True)
    
    return df

def eval_model(
    y_pred: Sequence[float] | np.ndarray | pd.Series,
    y_test: Sequence[float] | np.ndarray | pd.Series,
    y_train: Sequence[float] | np.ndarray | pd.Series,
    d_test: Sequence[Any] | pd.Series | None,
    stock_ticker: str,
    plot: bool = False,
) -> dict[str, float]:
    """Evaluate model predictions against true values and simple baseline forecasts.

    Args:
        y_pred: Predicted values for the test window.
        y_test: Ground-truth target values for the test window.
        y_train: Historical training targets used to build baseline predictors.
        d_test: Date/index values aligned with the test window, used when plotting.
        stock_ticker: stock ticker that you predict on
        plot: If ``True``, plot actual vs predicted vs naive baseline across ``d_test``.

    Returns:
        Dictionary containing regression and directional metrics for the model and
        reference baselines (zero, train-mean, and naive last-value baseline).

    Raises:
        ValueError: If inputs are empty or have incompatible lengths.
    """
    y_pred_arr = np.asarray(y_pred, dtype = float).ravel()
    y_true = np.asarray(y_test, dtype = float).ravel()
    y_tr = np.asarray(y_train, dtype = float).ravel()

    if y_true.size == 0:
        raise ValueError("y_test must contain at least one value.")
    if y_tr.size == 0:
        raise ValueError("y_train must contain at least one value.")
    if y_pred_arr.size != y_true.size:
        raise ValueError("y_pred and y_test must have the same length.")
    
    pred_zero = np.zeros_like(y_true)
    pred_mean = np.full_like(y_true, y_tr.mean())
    pred_naive = np.r_[y_tr[-1], y_true[:-1]]
    
    def rmse(a, b):
        return np.sqrt(mean_squared_error(a, b))
    
    def direction_acc(a, b):
        return np.mean(np.sign(a) == np.sign(b))
    
    def corr(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return np.nan
        return np.corrcoef(a, b)[0, 1]
    
    results = {
        "mae": float(mean_absolute_error(y_true, y_pred_arr)),
        "rmse": float(rmse(y_true, y_pred_arr)),
        "bias": float(np.mean(y_pred_arr - y_true)),
        "r2": float(r2_score(y_true, y_pred_arr)),
        "direction_acc": float(direction_acc(y_true, y_pred_arr)),
        "corr": float(corr(y_true, y_pred_arr)),

        "mae_zero": float(mean_absolute_error(y_true, pred_zero)),
        "rmse_zero": float(rmse(y_true, pred_zero)),

        "mae_mean": float(mean_absolute_error(y_true, pred_mean)),
        "rmse_mean": float(rmse(y_true, pred_mean)),

        "mae_naive": float(mean_absolute_error(y_true, pred_naive)),
        "rmse_naive": float(rmse(y_true, pred_naive)),
        "direction_acc_naive": float(direction_acc(y_true, pred_naive)),
    }
    
    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))

        if d_test is None:
            x = np.arange(y_true.size)
            use_dates = False
        else:
            x = np.asarray(d_test)
            if x.size != y_true.size:
                raise ValueError("d_test and y_test must have the same length when plotting.")
            use_dates = True

        ax.plot(x, y_true, label="Actual")
        ax.plot(x, y_pred_arr, label="Predicted")
        ax.plot(x, pred_naive, label="Naive", linestyle="--", alpha=0.8)

        ax.set_xlabel("Date" if use_dates else "Index")
        ax.set_ylabel("Return")
        ax.set_title(f"Stock return prediction for {stock_ticker}")
        ax.legend()
        ax.grid(True)

        if use_dates:
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            fig.autofmt_xdate()

        plt.tight_layout()
        plt.show()

    results["mase"] = float(results["mae"] / results["mae_naive"]) if results["mae_naive"] != 0 else float(np.nan)
    
    return results
