import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_model(actual, predicted):
    """Calculate regression model performance metrics."""

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted)
    }


def persistence_forecast(current_values):
    """Predict that the future value equals the current value."""

    return np.asarray(current_values)


def validate_temperature_inputs(data):
    """
    Validate input data for temperature forecasting.

    Required inputs:
        Temperature: degrees Celsius
        Humidity: percentage
        Carbon dioxide: ppm
        hour: decimal hour, 0 to less than 24
        day_of_week: integer, 0 to 6

    Returns:
        Validated input DataFrame in the correct feature order.

    Raises:
        ValueError for missing columns, non-numeric values,
        missing or infinite values, or invalid input ranges.
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
        values = X.to_numpy(dtype=float)
    except (TypeError, ValueError):
        raise ValueError(
            "All input features must contain numeric values."
        )

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

    return X