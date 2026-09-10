from __future__ import annotations

import pandas as pd


def build_features(market_data: pd.DataFrame) -> pd.DataFrame:
    """Build only backward-looking features and the next-day return target."""
    frame = market_data.sort_values("timestamp").copy()
    frame["return_1d"] = frame["close"].pct_change()
    frame["return_lag_1"] = frame["return_1d"].shift(1)
    frame["return_lag_2"] = frame["return_1d"].shift(2)
    frame["return_lag_7"] = frame["return_1d"].shift(7)
    frame["volatility_7d"] = frame["return_1d"].rolling(7).std()
    frame["volume_change_1d"] = frame["volume"].pct_change()
    frame["target_return_1d"] = frame["return_1d"].shift(-1)
    return frame.dropna().reset_index(drop=True)
