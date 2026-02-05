import pandas as pd
import numpy as np
import pytest
from lasps.data.processors.chart_generator import ChartGenerator


@pytest.fixture
def sample_ohlcv():
    np.random.seed(42)
    n = 60
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    return pd.DataFrame({
        "Open": close + np.random.randn(n) * 0.5,
        "High": close + abs(np.random.randn(n)) * 2,
        "Low": close - abs(np.random.randn(n)) * 2,
        "Close": close,
        "Volume": np.random.randint(100_000, 1_000_000, n),
    }, index=dates)


def test_generate_returns_tensor(sample_ohlcv):
    gen = ChartGenerator()
    tensor = gen.generate_tensor(sample_ohlcv)
    assert tensor.shape == (3, 224, 224)


def test_tensor_value_range(sample_ohlcv):
    gen = ChartGenerator()
    tensor = gen.generate_tensor(sample_ohlcv)
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_generate_saves_file(sample_ohlcv, tmp_path):
    gen = ChartGenerator()
    path = tmp_path / "test_chart.png"
    gen.save_chart(sample_ohlcv, str(path))
    assert path.exists()
