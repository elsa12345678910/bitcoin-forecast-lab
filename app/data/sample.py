from __future__ import annotations

import numpy as np
import pandas as pd


def load_sample_data(periods: int = 420, seed: int = 7) -> pd.DataFrame:
    """Return deterministic daily BTC/USD-like OHLCV data for local development."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=periods, freq="D", tz="UTC")
    returns = rng.normal(0.0008, 0.028, periods)
    close = 42_000 * np.exp(np.cumsum(returns))
    volume = rng.lognormal(mean=19.2, sigma=0.35, size=periods)
    return pd.DataFrame({"timestamp": dates, "close": close, "volume": volume})


def validate_market_data(frame: pd.DataFrame) -> None:
    required = {"timestamp", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps are not allowed")
    if (frame["close"] <= 0).any() or (frame["volume"] <= 0).any():
        raise ValueError("Prices and volume must be positive")
