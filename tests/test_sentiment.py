import pandas as pd
import numpy as np
import pytest
from lasps.data.processors.market_sentiment import MarketSentimentCalculator


@pytest.fixture
def sample_ohlcv():
    np.random.seed(42)
    n = 40
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    return pd.DataFrame({
        "date": dates,
        "open": close + np.random.randn(n) * 0.5,
        "high": close + abs(np.random.randn(n)) * 2,
        "low": close - abs(np.random.randn(n)) * 2,
        "close": close,
        "volume": np.random.randint(100_000, 1_000_000, n),
    })


def test_output_columns(sample_ohlcv):
    calc = MarketSentimentCalculator()
    result = calc.calculate(sample_ohlcv)
    for col in ["volume_ratio", "volatility_ratio", "gap_direction",
                "rsi_norm", "foreign_inst_flow"]:
        assert col in result.columns


def test_value_ranges(sample_ohlcv):
    calc = MarketSentimentCalculator()
    result = calc.calculate(sample_ohlcv)
    assert (result["volume_ratio"] >= 0).all()
    assert (result["volume_ratio"] <= 1).all()
    assert (result["gap_direction"] >= -1).all()
    assert (result["gap_direction"] <= 1).all()
    assert (result["rsi_norm"] >= 0).all()
    assert (result["rsi_norm"] <= 1).all()


def test_no_nan_after_fillna(sample_ohlcv):
    calc = MarketSentimentCalculator()
    result = calc.calculate(sample_ohlcv)
    for col in calc.get_feature_names():
        assert not result[col].isna().any(), f"{col} has NaN"


def test_with_investor_data(sample_ohlcv):
    calc = MarketSentimentCalculator()
    investor_df = pd.DataFrame({
        "date": sample_ohlcv["date"],
        "foreign_net": np.random.randint(-1_000_000, 1_000_000, len(sample_ohlcv)),
        "inst_net": np.random.randint(-500_000, 500_000, len(sample_ohlcv)),
    })
    result = calc.calculate(sample_ohlcv, investor_df)
    assert not result["foreign_inst_flow"].isna().any()
    assert (result["foreign_inst_flow"] != 0).any()
