from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

FEATURE_COLUMNS = [
    "return_lag_1",
    "return_lag_2",
    "return_lag_7",
    "volatility_7d",
    "volume_change_1d",
]


@dataclass
class ForecastModel:
    name: str
    estimator: object

    def fit(self, features, target) -> "ForecastModel":
        self.estimator.fit(features[FEATURE_COLUMNS], target)
        return self

    def predict(self, features) -> np.ndarray:
        return self.estimator.predict(features[FEATURE_COLUMNS])


def create_model(name: str = "ridge") -> ForecastModel:
    if name == "zero_return":
        return ForecastModel(name, _ZeroReturnEstimator())
    if name == "ridge":
        return ForecastModel(name, Ridge(alpha=1.0))
    if name == "random_forest":
        return ForecastModel(
            name,
            RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=7,
                n_jobs=-1,
            ),
        )
    raise ValueError(f"Unknown model: {name}")


class _ZeroReturnEstimator:
    def fit(self, features, target):
        return self

    def predict(self, features):
        return np.zeros(len(features))
