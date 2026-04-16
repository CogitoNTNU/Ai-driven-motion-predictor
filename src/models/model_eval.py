from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

async def eval_model(
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
    """
    y_true = np.asarray(y_test, dtype = float).ravel()
    y_tr = np.asarray(y_train, dtype = float).ravel()
    
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
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "bias": np.mean(y_pred - y_true),
        "r2": r2_score(y_true, y_pred),
        "direction_acc": direction_acc(y_true, y_pred),
        "corr": corr(y_true, y_pred),

        "mae_zero": mean_absolute_error(y_true, pred_zero),
        "rmse_zero": rmse(y_true, pred_zero),

        "mae_mean": mean_absolute_error(y_true, pred_mean),
        "rmse_mean": rmse(y_true, pred_mean),

        "mae_naive": mean_absolute_error(y_true, pred_naive),
        "rmse_naive": rmse(y_true, pred_naive),
        "direction_acc_naive": direction_acc(y_true, pred_naive),
    }
    
    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))

        x = d_test

        ax.plot(x, y_true, label="Actual")
        ax.plot(x, y_pred, label="Predicted")
        ax.plot(x, pred_naive, label="Naive", linestyle="--", alpha=0.8)

        ax.set_xlabel("Date" if d_test is not None else "Index")
        ax.set_ylabel("Return")
        ax.set_title(f"Stock return prediction for {stock_ticker}")
        ax.legend()
        ax.grid(True)

        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()

        plt.tight_layout()
        plt.show()

    results["mase"] = results["mae"] / results["mae_naive"] if results["mae_naive"] != 0 else np.nan
    
    return results