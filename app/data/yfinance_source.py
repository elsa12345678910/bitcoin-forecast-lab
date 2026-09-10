from __future__ import annotations

import pandas as pd
import yfinance as yf

from app.data.sample import validate_market_data


def fetch_bitcoin_data(period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Fetch BTC/USD OHLCV history and return the app's canonical data shape."""
    history = yf.Ticker("BTC-USD").history(period=period, interval=interval, auto_adjust=False)
    if history.empty:
        raise RuntimeError("Yahoo Finance returned no Bitcoin data")

    data = history.reset_index()
    timestamp_column = "Datetime" if "Datetime" in data.columns else "Date"
    data = data.rename(columns={timestamp_column: "timestamp"})
    data = data[["timestamp", "Close", "Volume"]].rename(
        columns={"Close": "close", "Volume": "volume"}
    )
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.dropna(subset=["close", "volume"]).reset_index(drop=True)
    validate_market_data(data)
    return data


if __name__ == "__main__":
    btc_data = fetch_bitcoin_data(period="5y")
    print(btc_data.head().to_string(index=False))
