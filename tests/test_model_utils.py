
import numpy as np
from src.model_utils import evaluate_model, persistence_forecast


def test_persistence_forecast():
    current = [20, 21, 22, 23]
    predictions = persistence_forecast(current)

    np.testing.assert_array_equal(
        predictions,
        [20, 21, 22, 23]
    )


def test_evaluate_model():
    actual = [20.5, 21.5, 22.5, 23.5]
    predicted = [20, 21, 22, 23]

    results = evaluate_model(actual, predicted)

    assert np.isclose(results["MAE"], 0.5)
    assert np.isclose(results["RMSE"], 0.5)
    assert np.isclose(results["R2"], 0.8)
