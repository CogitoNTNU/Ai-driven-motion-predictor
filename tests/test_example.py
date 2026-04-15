import pandas as pd
import pytest

from src.models.baseline_model import PersistenceBaselineModel


def test_predict_repeats_last_training_value():
    model = PersistenceBaselineModel()
    model.train(pd.Series([101.25, 103.5, 99.75]))

    prediction = model.predict(steps=3)

    assert prediction.tolist() == pytest.approx([99.75, 99.75, 99.75])


def test_walk_forward_prediction_uses_previous_observation():
    train = pd.Series([10.0, 11.0], index=pd.RangeIndex(start=0, stop=2))
    test = pd.Series([12.0, 13.5, 14.0], index=pd.RangeIndex(start=2, stop=5))

    prediction = PersistenceBaselineModel.walk_forward_predict(train, test)

    assert prediction.name == "baseline_prediction"
    assert prediction.tolist() == pytest.approx([11.0, 12.0, 13.5])
