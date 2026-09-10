from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.forecasting.models import create_model


def run_backtest(features, model_name: str, initial_train_size: int = 120, step_size: int = 7) -> dict:
    predictions = []
    actuals = []
    timestamps = []
    for test_start in range(initial_train_size, len(features), step_size):
        test_end = min(test_start + step_size, len(features))
        model = create_model(model_name)
        model.fit(features.iloc[:test_start], features.iloc[:test_start]["target_return_1d"])
        predictions.extend(model.predict(features.iloc[test_start:test_end]))
        actuals.extend(features.iloc[test_start:test_end]["target_return_1d"])
        timestamps.extend(features.iloc[test_start:test_end]["timestamp"])

    if not predictions:
        raise ValueError("Not enough rows for the requested backtest")
    prediction_array = np.asarray(predictions)
    actual_array = np.asarray(actuals)
    return {
        "model": model_name,
        "observations": len(actual_array),
        "mae": float(mean_absolute_error(actual_array, prediction_array)),
        "rmse": float(np.sqrt(mean_squared_error(actual_array, prediction_array))),
        "directional_accuracy": float(np.mean(np.sign(actual_array) == np.sign(prediction_array))),
        "predictions": [
            {"timestamp": str(timestamp), "actual": float(actual), "predicted": float(predicted)}
            for timestamp, actual, predicted in zip(timestamps, actual_array, prediction_array)
        ],
    }
