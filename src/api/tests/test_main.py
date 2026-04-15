import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from main import get_stock_history, get_stock_predictions


class FakeTicker:
    def __init__(self, history_frame: pd.DataFrame, last_price: float = 0.0):
        self._history_frame = history_frame
        self.fast_info = SimpleNamespace(last_price=last_price)

    def history(self, period=None, start=None, end=None):
        return self._history_frame


class StockEndpointTests(unittest.TestCase):
    def test_history_endpoint_transforms_market_data(self):
        history_frame = pd.DataFrame(
            {"Close": [100.0, 110.0]},
            index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
        )

        with patch("main.yf.Ticker", return_value=FakeTicker(history_frame)):
            payload = asyncio.run(get_stock_history("aapl", "1M"))

        self.assertEqual(payload["symbol"], "AAPL")
        self.assertEqual(payload["range"], "1M")
        self.assertEqual(
            payload["data"],
            [
                {"date": "2026-01-02", "price": 100.0},
                {"date": "2026-01-05", "price": 110.0},
            ],
        )
        self.assertEqual(payload["metadata"]["percentage_growth"], 10.0)
        self.assertEqual(payload["metadata"]["trading_days"], 2)

    def test_predictions_endpoint_reports_unavailable_without_models(self):
        history_frame = pd.DataFrame(
            {"Close": [198.0, 201.5]},
            index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
        )

        with patch(
            "main.yf.Ticker", return_value=FakeTicker(history_frame, last_price=201.5)
        ):
            payload = asyncio.run(get_stock_predictions("msft"))

        self.assertEqual(payload["symbol"], "MSFT")
        self.assertEqual(payload["status"], "unavailable")
        self.assertEqual(payload["current_price"], 201.5)
        self.assertEqual(payload["models"], [])
        self.assertIsNone(payload["ensemble"])
        self.assertIn("No deployed prediction models", payload["message"])
