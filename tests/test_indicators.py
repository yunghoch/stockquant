import pandas as pd
import numpy as np
import pytest
from lasps.data.processors.technical_indicators import TechnicalIndicatorCalculator
from lasps.utils.constants import INDICATOR_FEATURES


@pytest.fixture
def sample_ohlcv():
    np.random.seed(42)
    n = 150
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    return pd.DataFrame({
        "date": dates,
        "open": close + np.random.randn(n) * 0.5,
        "high": close + abs(np.random.randn(n)) * 2,
        "low": close - abs(np.random.randn(n)) * 2,
        "close": close,
        "volume": np.random.randint(100_000, 1_000_000, n).astype(float),
    })


def test_all_15_indicators_present(sample_ohlcv):
    calc = TechnicalIndicatorCalculator()
    result = calc.calculate(sample_ohlcv)
    for feat in INDICATOR_FEATURES:
        assert feat in result.columns, f"Missing: {feat}"


def test_output_length(sample_ohlcv):
    calc = TechnicalIndicatorCalculator()
    result = calc.calculate(sample_ohlcv)
    assert len(result) == len(sample_ohlcv)


def test_ma_ordering(sample_ohlcv):
    calc = TechnicalIndicatorCalculator()
    result = calc.calculate(sample_ohlcv)
    valid = result.dropna()
    assert valid["ma5"].std() >= valid["ma20"].std()
