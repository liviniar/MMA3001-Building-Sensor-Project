
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(actual, predicted):
    """Calculate regression MAE, RMSE and R2 for temperatures in Celsius."""
    actual = np.asarray(actual, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()

    if actual.size != predicted.size or actual.size < 2:
        raise ValueError("Inputs must have matching lengths and at least two observations.")

    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Inputs must contain finite numerical values.")

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted)
    }


def persistence_forecast(current_values):
    """Return current temperatures as persistence forecasts in Celsius."""
    return np.asarray(current_values)


def validate_temperature_inputs(data):
    """
    Validate sensor data and return numeric features in the required order.

    Parameters
    ----------
    data : pandas.DataFrame
        Required columns:
        Temperature (Celsius), Humidity (%), Carbon dioxide (ppm),
        hour (0 inclusive to 24 exclusive), day_of_week (integer 0-6).

    Returns
    -------
    pandas.DataFrame
        Validated numeric input features in the required order.

    Raises
    ------
    ValueError
        If required columns are missing or values are invalid.
    """
    required = [
        "Temperature",
        "Humidity",
        "Carbon dioxide",
        "hour",
        "day_of_week"
    ]

    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = data[required].copy()

    try:
        X = X.astype(float)
    except (TypeError, ValueError):
        raise ValueError("All input features must contain numeric values.")

    values = X.to_numpy()

    if not np.isfinite(values).all():
        raise ValueError("Input contains missing or infinite values.")

    if not X["Humidity"].between(0, 100).all():
        raise ValueError("Humidity must be between 0 and 100%.")

    if not X["hour"].between(0, 24, inclusive="left").all():
        raise ValueError("Hour must be between 0 and 24.")

    if not X["day_of_week"].isin(range(7)).all():
        raise ValueError("Day of week must be an integer from 0 to 6.")

    return X
