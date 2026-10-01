import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_model(actual, predicted):
    """
    Evaluate the accuracy of a regression model.

    Parameters
    ----------
    actual : array-like
        Observed target values, in degrees Celsius.
    predicted : array-like
        Predicted target values, in degrees Celsius.

    Returns
    -------
    dict
        MAE and RMSE in degrees Celsius, and
        dimensionless R2.

    Notes
    -----
    Inputs must contain matching, non-empty arrays
    of finite numerical values.
    """
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted)
    }


def persistence_forecast(current_values):
    """
    Generate persistence baseline predictions.

    Assumes the future temperature equals the
    temperature at the current observation.

    Parameters
    ----------
    current_values : array-like
        Current temperature observations in degrees Celsius.

    Returns
    -------
    numpy.ndarray
        Forecast temperatures in degrees Celsius.
    """
    return np.asarray(current_values)


def validate_temperature_inputs(data):
    """
    Validate sensor inputs for temperature forecasting.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the following features:
        - Temperature: degrees Celsius
        - Humidity: percentage
        - Carbon dioxide: ppm
        - hour: decimal hour, 0 to less than 24
        - day_of_week: integer, 0 to 6

    Returns
    -------
    pandas.DataFrame
        Validated input data in the required feature order.

    Raises
    ------
    ValueError
        If required columns are missing, values are
        non-numeric, values are missing or infinite,
        humidity is outside 0–100%, hour is outside
        0–24, or day_of_week is not an integer
        between 0 and 6.

    Notes
    -----
    This function performs basic input validation.
    It does not guarantee reliable predictions for
    observations outside the training data ranges.
    """
    required = [
        "Temperature",
        "Humidity",
        "Carbon dioxide",
        "hour",
        "day_of_week"
    ]

    missing_columns = [
        col for col in required if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

X = data[required].copy()

try:
    X = X.astype(float)
except (TypeError, ValueError):
    raise ValueError(
        "All input features must contain numeric values."
    )

    values = X.to_numpy()

    if not np.isfinite(values).all():
        raise ValueError(
            "Input contains missing or infinite values."
        )

    if not X["Humidity"].between(0, 100).all():
        raise ValueError(
            "Humidity must be between 0 and 100%."
        )

    if not X["hour"].between(
        0, 24, inclusive="left"
    ).all():
        raise ValueError(
            "Hour must be between 0 and 24."
        )

    if not X["day_of_week"].isin(range(7)).all():
        raise ValueError(
            "Day of week must be an integer from 0 to 6."
        )
actual = np.asarray(actual, dtype=float).ravel()
predicted = np.asarray(predicted, dtype=float).ravel()

if actual.size != predicted.size or actual.size < 2:
    raise ValueError(
        "Inputs must have matching lengths and at least two observations."
    )

if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
    raise ValueError(
        "Inputs must contain finite numerical values."
    )
    return X
