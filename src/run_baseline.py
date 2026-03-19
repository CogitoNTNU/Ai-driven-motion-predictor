from pathlib import Path
import json

import numpy as np
import pandas as pd

from data.yfinanceLoader import load_price_data
from models.baseline_model import PersistenceBaselineModel


def _price_column(df: pd.DataFrame) -> str:
    if "Adj Close" in df.columns:
        return "Adj Close"

    if "Close" in df.columns:
        return "Close"

    raise KeyError("Expected 'Adj Close' or 'Close' column in dataset.")


def _mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    safe_true = y_true.replace(0, np.nan)
    value = np.abs((safe_true - y_pred) / safe_true)
    return float(np.nanmean(value) * 100)


def main():
    ticker = "AAPL"
    start_date = "2020-01-01"
    split_ratio = 0.8

    data = load_price_data(ticker=ticker, start=start_date)
    price_col = _price_column(data)

    prices = data[price_col].astype("float64")
    split_index = int(len(prices) * split_ratio)

    train = prices.iloc[:split_index]
    test = prices.iloc[split_index:]

    baseline = PersistenceBaselineModel()
    baseline.train(train)
    predictions = baseline.walk_forward_predict(train, test)

    mae = _mae(test, predictions)
    rmse = _rmse(test, predictions)
    mape = _mape(test, predictions)

    output_dir = Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    chart_path = output_dir / "baseline_vs_actual.html"
    dates = [idx.strftime("%Y-%m-%d") for idx in test.index]
    actual_values = test.values.tolist()
    predicted_values = predictions.values.tolist()

    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>{ticker} Baseline Forecast</title>
  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
</head>
<body>
  <div id=\"chart\" style=\"width:100%;height:650px;\"></div>
  <script>
    const dates = {json.dumps(dates)};
    const actual = {json.dumps(actual_values)};
    const baseline = {json.dumps(predicted_values)};

    const traces = [
      {{ x: dates, y: actual, mode: 'lines', name: 'Actual', line: {{ width: 2 }} }},
      {{ x: dates, y: baseline, mode: 'lines', name: 'Baseline (t-1)', line: {{ width: 2, dash: 'dash' }} }}
    ];

    const layout = {{
      title: '{ticker} Baseline Forecast (Persistence Model)',
      xaxis: {{ title: 'Date' }},
      yaxis: {{ title: 'Price' }},
      template: 'plotly_white'
    }};

    Plotly.newPlot('chart', traces, layout, {{ responsive: true }});
  </script>
</body>
</html>
"""
    chart_path.write_text(html, encoding="utf-8")

    metrics = pd.DataFrame(
        {
            "model": ["Persistence Baseline (t-1)"],
            "ticker": [ticker],
            "start_date": [start_date],
            "train_rows": [len(train)],
            "test_rows": [len(test)],
            "MAE": [mae],
            "RMSE": [rmse],
            "MAPE_percent": [mape],
        }
    )
    metrics_path = output_dir / "baseline_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    print("Baseline evaluation complete.")
    print(f"Chart saved to: {chart_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
