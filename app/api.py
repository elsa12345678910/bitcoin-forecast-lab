from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.backtesting.walk_forward import run_backtest
from app.data.yfinance_source import fetch_bitcoin_data
from app.features.build import build_features
from app.forecasting.models import create_model

app = FastAPI(title="Bitcoin Forecast Lab", version="0.1.0")


@lru_cache(maxsize=1)
def dataset():
    data = fetch_bitcoin_data(period="5y", interval="1d")
    return build_features(data)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/models")
def models():
    return {"models": ["zero_return", "ridge", "random_forest"]}


@app.get("/api/backtest")
def backtest(model: str = "ridge", initial_train_size: int = 120, step_size: int = 7):
    if initial_train_size < 30 or step_size < 1:
        raise ValueError("initial_train_size must be at least 30 and step_size must be positive")
    return run_backtest(
        dataset(),
        model_name=model,
        initial_train_size=initial_train_size,
        step_size=step_size,
    )


@app.get("/api/forecast")
def forecast(model: str = "ridge"):
    features = dataset()
    estimator = create_model(model).fit(features.iloc[:-1], features.iloc[:-1]["target_return_1d"])
    latest = features.iloc[[-1]]
    return {
        "model": model,
        "timestamp": str(latest.iloc[0]["timestamp"]),
        "forecast_return": float(estimator.predict(latest)[0]),
        "latest_close": float(latest.iloc[0]["close"]),
    }


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).parent.parent / "frontend" / "index.html")
