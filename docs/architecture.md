# Architecture

```text
Data source -> validation -> features -> model -> walk-forward backtest -> API -> UI
```

The model layer receives a feature table and never fetches data. The backtester owns chronological splitting. The API is the boundary for the future React client. Sample data is deterministic so tests and local demonstrations remain reproducible.
