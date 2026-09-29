import numpy as np
import pandas as pd
import pytest

from src.model_utils import (
    evaluate_model,
    persistence_forecast,
    validate_temperature_inputs
)


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


# Example valid sensor observation
def valid_input():
    return pd.DataFrame({
        "Temperature": [22.5],
        "Humidity": [55.0],
        "Carbon dioxide": [600.0],
        "hour": [14.5],
        "day_of_week": [2]
    })


def test_valid_temperature_inputs():
    data = valid_input()

    result = validate_temperature_inputs(data)

    assert len(result) == 1
    assert list(result.columns) == list(data.columns)


def test_missing_input_column():
    data = valid_input()
    data = data.drop(columns=["Humidity"])

    with pytest.raises(ValueError):
        validate_temperature_inputs(data)


def test_missing_input_value():
    data = valid_input()
    data.loc[0, "Temperature"] = np.nan

    with pytest.raises(ValueError):
        validate_temperature_inputs(data)


def test_invalid_humidity():
    data = valid_input()
    data.loc[0, "Humidity"] = 150

    with pytest.raises(ValueError):
        validate_temperature_inputs(data)