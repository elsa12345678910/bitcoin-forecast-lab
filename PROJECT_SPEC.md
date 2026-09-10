# Project specification

## First forecasting problem

Forecast daily BTC/USD returns at a one-day horizon, with a seven-day horizon reserved for the next iteration. The initial output is a point estimate plus historical backtest metrics.

## Guardrails

- Use chronological splits only.
- Features must use information available before the forecast timestamp.
- Keep raw data immutable once real source adapters are added.
- Treat forecasts as estimates, not certainty or financial advice.

## Initial success criteria

A user can select the sample dataset, choose a model, run a walk-forward backtest, inspect MAE/RMSE/directional accuracy, and request the next forecast through the API.

## Next decisions

- Select and document a primary historical market-data provider.
- Add raw/processed storage and provenance metadata.
- Add prediction intervals and experiment persistence.
- Replace the static dashboard with the planned React application after the API stabilises.
