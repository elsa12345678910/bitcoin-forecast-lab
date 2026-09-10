from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from app.data.yfinance_source import fetch_bitcoin_data

COINBASE_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"


def fetch_coinbase_data(days: int = 1825) -> pd.DataFrame:
    """Fetch Coinbase daily candles in chunks because each response is limited."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    rows = []
    while start < end:
        chunk_end = min(start + timedelta(days=300), end)
        response = requests.get(
            COINBASE_URL,
            params={"granularity": 86400, "start": start.isoformat(), "end": chunk_end.isoformat()},
            timeout=30,
        )
        response.raise_for_status()
        rows.extend(response.json())
        start = chunk_end
    if not rows:
        raise RuntimeError("Coinbase returned no Bitcoin candles")
    data = pd.DataFrame(rows, columns=["timestamp_s", "low", "high", "open", "close", "volume"])
    data["timestamp"] = pd.to_datetime(data["timestamp_s"], unit="s", utc=True).dt.floor("D")
    return data[["timestamp", "close"]].drop_duplicates("timestamp").sort_values("timestamp")


def compare_bitcoin_sources(days: int = 1825) -> tuple[pd.DataFrame, dict]:
    yahoo = fetch_bitcoin_data(period="5y", interval="1d")[["timestamp", "close"]]
    yahoo["date"] = yahoo["timestamp"].dt.floor("D")
    coinbase = fetch_coinbase_data(days=days)
    coinbase["date"] = coinbase["timestamp"].dt.floor("D")
    comparison = yahoo.merge(coinbase, on="date", how="inner", suffixes=("_yahoo", "_coinbase"))
    if comparison.empty:
        raise RuntimeError("The Bitcoin sources have no overlapping dates")
    comparison["price_difference_pct"] = (
        (comparison["close_yahoo"] - comparison["close_coinbase"]).abs()
        / comparison["close_coinbase"]
        * 100
    )
    comparison["yahoo_return"] = comparison["close_yahoo"].pct_change()
    comparison["coinbase_return"] = comparison["close_coinbase"].pct_change()
    returns = comparison[["yahoo_return", "coinbase_return"]].dropna()
    summary = {
        "yahoo_rows": len(yahoo),
        "coinbase_rows": len(coinbase),
        "overlapping_days": len(comparison),
        "start": str(comparison["date"].min().date()),
        "end": str(comparison["date"].max().date()),
        "mean_absolute_price_difference_pct": float(comparison["price_difference_pct"].mean()),
        "max_absolute_price_difference_pct": float(comparison["price_difference_pct"].max()),
        "daily_return_correlation": float(returns["yahoo_return"].corr(returns["coinbase_return"])),
    }
    return comparison, summary
