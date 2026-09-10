from app.data.sample import load_sample_data, validate_market_data
from app.features.build import build_features
from app.backtesting.walk_forward import run_backtest


def test_sample_data_is_valid():
    data = load_sample_data(periods=30)
    validate_market_data(data)
    assert len(data) == 30


def test_features_are_backward_looking_and_complete():
    features = build_features(load_sample_data(periods=40))
    assert "target_return_1d" in features
    assert not features.isna().any().any()


def test_walk_forward_backtest_is_chronological():
    result = run_backtest(build_features(load_sample_data(periods=180)), "zero_return", 100, 10)
    assert result["observations"] > 0
    assert 0 <= result["directional_accuracy"] <= 1
