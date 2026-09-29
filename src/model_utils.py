
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
