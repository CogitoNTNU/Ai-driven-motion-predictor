import numpy as np
import pandas as pd


class PersistenceBaselineModel:
    def __init__(self):
        self.last_value = None

    def train(self, series: pd.Series):
        if series.empty:
            raise ValueError("Training series is empty.")

        self.last_value = float(series.iloc[-1])

    def predict(self, steps=1):
        if self.last_value is None:
            raise ValueError("Model must be trained before calling predict().")

        if steps < 1:
            raise ValueError("Steps must be >= 1.")

        return np.full(shape=steps, fill_value=self.last_value, dtype="float64")

    @staticmethod
    def walk_forward_predict(
        train_series: pd.Series, test_series: pd.Series
    ) -> pd.Series:
        if train_series.empty or test_series.empty:
            raise ValueError("Train and test series must both be non-empty.")

        previous = float(train_series.iloc[-1])
        predictions = []

        for value in test_series:
            predictions.append(previous)
            previous = float(value)

        return pd.Series(
            predictions, index=test_series.index, name="baseline_prediction"
        )
