# Bitcoin Forecast Lab

A modular research tool for daily Bitcoin return forecasts. The first slice keeps data, features, forecasting, backtesting, and presentation separate so each part can evolve independently.

## MVP scope

- Generate deterministic sample BTC/USD daily data.
- Build lagged-return and rolling-volatility features.
- Compare a zero-return baseline with a linear regression model.
- Run chronological walk-forward backtests.
- Serve forecasts and metrics through FastAPI.
- View the results in a small browser dashboard.

Forecasts are estimates for research, not financial advice or guaranteed future prices.

## Run locally

```bash
cd "bitcoin app"
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.api:app --reload
```

Open http://127.0.0.1:8000 in a browser.

The Streamlit version can be launched with:

```bash
streamlit run app.py
```

It opens at http://localhost:8501 and provides the same live five-year BTC/USD data with interactive model controls.

Run tests with:

```bash
pytest
```

Fetch live BTC/USD daily data through Yahoo Finance:

```bash
.venv/bin/python -m app.data.yfinance_source
```

The adapter uses `BTC-USD` and returns the canonical `timestamp`, `close`, and `volume` columns used by the forecasting pipeline.

## Structure

- `app/data`: source adapters and validation
- `app/features`: reproducible feature engineering
- `app/forecasting`: model implementations
- `app/backtesting`: chronological evaluation
- `app/api.py`: API boundary for the UI
- `frontend`: intentionally small static dashboard
- `configs`: experiment configuration
- `docs`: project decisions and architecture
