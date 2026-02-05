import pandas as pd
import numpy as np
from lasps.utils.helpers import compute_label, normalize_minmax
from lasps.utils.metrics import classification_metrics


def test_compute_label_buy():
    prices = pd.Series([100.0, 101, 102, 103, 104, 110])
    assert compute_label(prices, 0) == 2


def test_compute_label_sell():
    prices = pd.Series([100.0, 99, 98, 97, 96, 90])
    assert compute_label(prices, 0) == 0


def test_compute_label_hold():
    prices = pd.Series([100.0, 100, 100, 100, 100, 101])
    assert compute_label(prices, 0) == 1


def test_compute_label_insufficient_data():
    prices = pd.Series([100.0, 101, 102])
    assert compute_label(prices, 0) == -1


def test_normalize_minmax():
    s = pd.Series([0, 50, 100])
    result = normalize_minmax(s)
    assert result.iloc[0] == 0.0
    assert result.iloc[1] == 0.5
    assert result.iloc[2] == 1.0


def test_classification_metrics():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    m = classification_metrics(y_true, y_pred)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0


def test_normalize_time_series_basic():
    """Test min-max normalization produces [0,1] for features 0-19."""
    from lasps.utils.helpers import normalize_time_series
    data = np.random.rand(60, 25) * 1000  # random large-scale data
    result = normalize_time_series(data)
    # OHLCV + indicators (0-19) should be in [0, 1]
    assert result[:, :20].min() >= 0.0
    assert result[:, :20].max() <= 1.0
    # Sentiment (20-24) should be unchanged
    np.testing.assert_array_almost_equal(result[:, 20:], data[:, 20:])


def test_normalize_time_series_constant_feature():
    """Test constant features map to 0.5."""
    from lasps.utils.helpers import normalize_time_series
    data = np.ones((60, 25)) * 42.0
    result = normalize_time_series(data)
    # Constant OHLCV+indicator features should be 0.5
    assert np.allclose(result[:, :20], 0.5)
    # Sentiment unchanged
    assert np.allclose(result[:, 20:], 42.0)


def test_normalize_time_series_shape_preserved():
    """Test output shape matches input."""
    from lasps.utils.helpers import normalize_time_series
    data = np.random.rand(60, 25).astype(np.float32)
    result = normalize_time_series(data)
    assert result.shape == (60, 25)
    assert result.dtype == np.float32
