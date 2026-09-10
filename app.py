from __future__ import annotations

import pandas as pd
import streamlit as st

from app.backtesting.walk_forward import run_backtest
from app.data.compare_sources import compare_bitcoin_sources
from app.data.yfinance_source import fetch_bitcoin_data
from app.features.build import build_features
from app.forecasting.models import create_model

st.set_page_config(page_title="Bitcoin Forecast Lab", page_icon="₿", layout="wide")


@st.cache_data(ttl=900)
def load_features() -> pd.DataFrame:
    """Fetch five years of BTC/USD data and build forecasting features."""
    market_data = fetch_bitcoin_data(period="5y", interval="1d")
    return build_features(market_data)


@st.cache_data(ttl=900)
def run_cached_backtest(model_name: str, initial_train_size: int, step_size: int) -> dict:
    """Run a cached chronological backtest for the selected configuration."""
    return run_backtest(
        load_features(),
        model_name=model_name,
        initial_train_size=initial_train_size,
        step_size=step_size,
    )


st.title("Bitcoin Forecast Lab")
st.caption("Research estimates for BTC-USD spot returns, not financial advice.")

with st.sidebar:
    st.header("Forecast controls")
    model_name = st.selectbox(
        "Model",
        options=["ridge", "random_forest", "zero_return"],
        format_func=lambda value: {
            "ridge": "Ridge regression",
            "random_forest": "Random forest",
            "zero_return": "Zero-return baseline",
        }[value],
    )
    initial_train_size = st.number_input(
        "Initial training days", min_value=30, max_value=1000, value=120, step=10
    )
    step_size = st.number_input("Walk-forward step", min_value=1, max_value=30, value=7)
    run = st.button("Run backtest", type="primary", use_container_width=True)

backtest_config = (model_name, int(initial_train_size), int(step_size))
if (
    "backtest" not in st.session_state
    or st.session_state.get("backtest_config") != backtest_config
    or run
):
    with st.spinner("Fetching BTC/USD data and running the backtest..."):
        try:
            st.session_state.backtest = run_cached_backtest(*backtest_config)
            st.session_state.backtest_config = backtest_config
        except Exception as error:
            st.error(f"Unable to run the backtest: {error}")
            st.stop()

try:
    features = load_features()
    estimator = create_model(model_name).fit(
        features.iloc[:-1], features.iloc[:-1]["target_return_1d"]
    )
    latest = features.iloc[[-1]]
    forecast_return = float(estimator.predict(latest)[0])
except Exception as error:
    st.error(f"Unable to generate the forecast: {error}")
    st.stop()

backtest = st.session_state.backtest
metrics = st.columns(4)
metrics[0].metric("Next daily return", f"{forecast_return:.2%}")
metrics[1].metric("Latest BTC price", f"${latest.iloc[0]['close']:,.0f}")
metrics[2].metric("Backtest MAE", f"{backtest['mae']:.2%}")
metrics[3].metric("Direction accuracy", f"{backtest['directional_accuracy']:.1%}")

st.subheader("Walk-forward predictions")
predictions = pd.DataFrame(backtest["predictions"])
predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])
chart_data = predictions.set_index("timestamp")[["actual", "predicted"]].rename(
    columns={"actual": "Actual return", "predicted": "Predicted return"}
)
st.line_chart(chart_data.tail(100))

left, right = st.columns([1.6, 1])
with left:
    st.subheader("Latest observations")
    recent = predictions.tail(8).iloc[::-1].copy()
    recent["timestamp"] = recent["timestamp"].dt.strftime("%d %b %Y")
    recent["actual"] = recent["actual"].map(lambda value: f"{value:.2%}")
    recent["predicted"] = recent["predicted"].map(lambda value: f"{value:.2%}")
    recent["error"] = (
        predictions["predicted"] - predictions["actual"]
    ).tail(8).iloc[::-1].map(lambda value: f"{value:.2%}").to_numpy()
    st.dataframe(
        recent.rename(
            columns={
                "timestamp": "Date",
                "actual": "Actual return",
                "predicted": "Predicted return",
                "error": "Error",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
with right:
    st.subheader("Run details")
    st.write(f"**Model:** {model_name}")
    st.write(f"**Observations:** {backtest['observations']}")
    st.write(f"**Training window:** {initial_train_size} days")
    st.write(f"**Walk-forward step:** {step_size} days")
    if backtest["directional_accuracy"] >= 0.5:
        st.info("At least half of historical return directions were correct. Compare this with the baseline before drawing conclusions.")
    else:
        st.warning("Directional accuracy was below 50%. This is useful evidence about the limits of this feature set.")

with st.expander("Bitcoin instruments"):
    st.write("This app currently forecasts **BTC-USD spot returns**.")
    st.table(
        pd.DataFrame(
            [
                ["BTC spot", "Direct Bitcoin price and the focus of this model."],
                ["Bitcoin ETFs", "Funds designed to track Bitcoin's price."],
                ["Futures", "Contracts based on an agreed future BTC price."],
                ["Options", "Contracts giving a right to buy or sell at a set price."],
                ["Wrapped BTC", "Tokenised Bitcoin used on another blockchain."],
                ["Mining shares", "Company shares exposed to mining, not Bitcoin itself."],
            ],
            columns=["Instrument", "Description"],
        )
    )

with st.expander("Compare independent data sources"):
    st.write("This compares Yahoo Finance BTC-USD with Coinbase BTC-USD over their shared five-year window. Close agreement does not guarantee either source is correct, but large gaps or missing dates are useful warning signs.")
    if st.button("Run source comparison"):
        with st.spinner("Fetching the comparison dataset..."):
            try:
                comparison, summary = compare_bitcoin_sources()
                source_metrics = st.columns(4)
                source_metrics[0].metric("Overlapping days", f"{summary['overlapping_days']:,}")
                source_metrics[1].metric("Return correlation", f"{summary['daily_return_correlation']:.4f}")
                source_metrics[2].metric("Mean price difference", f"{summary['mean_absolute_price_difference_pct']:.3f}%")
                source_metrics[3].metric("Maximum difference", f"{summary['max_absolute_price_difference_pct']:.2f}%")
                st.caption(f"Shared range: {summary['start']} to {summary['end']} · Yahoo rows: {summary['yahoo_rows']:,} · Coinbase rows: {summary['coinbase_rows']:,}")
                source_chart = comparison.set_index("date")[["close_yahoo", "close_coinbase"]].rename(columns={"close_yahoo": "Yahoo Finance", "close_coinbase": "Coinbase"})
                st.line_chart(source_chart)
                st.dataframe(comparison[["date", "close_yahoo", "close_coinbase", "price_difference_pct"]].tail(10).iloc[::-1], hide_index=True, use_container_width=True)
            except Exception as error:
                st.error(f"Unable to compare sources: {error}")
