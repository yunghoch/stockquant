"""Tests for Alpha101 implementation."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lasps.data.processors.alpha101 import Alpha101Calculator
from lasps.data.processors.alpha101.operators import (
    rank, scale, delta, delay, ts_min, ts_max, ts_argmin, ts_argmax,
    ts_rank, ts_sum, stddev, correlation, decay_linear, sma, returns
)


@pytest.fixture
def sample_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n_dates = 100
    n_stocks = 10

    dates = pd.date_range('2024-01-01', periods=n_dates, freq='B')
    stocks = [f'STOCK_{i:02d}' for i in range(n_stocks)]

    # Generate random walk prices
    base_price = 100
    returns_data = np.random.randn(n_dates, n_stocks) * 0.02

    close = pd.DataFrame(
        base_price * np.exp(np.cumsum(returns_data, axis=0)),
        index=dates,
        columns=stocks
    )

    # Generate OHLV from close
    high = close * (1 + np.abs(np.random.randn(n_dates, n_stocks) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n_dates, n_stocks) * 0.01))
    open_ = close.shift(1).fillna(close)

    volume = pd.DataFrame(
        np.random.randint(1000000, 10000000, size=(n_dates, n_stocks)),
        index=dates,
        columns=stocks
    ).astype(float)

    return open_, high, low, close, volume


@pytest.fixture
def calculator(sample_data):
    """Create Alpha101Calculator with sample data."""
    open_, high, low, close, volume = sample_data
    return Alpha101Calculator(
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


class TestOperators:
    """Test individual operators."""

    def test_rank(self, sample_data):
        """Test rank operator scales to [0, 1]."""
        _, _, _, close, _ = sample_data
        ranked = rank(close)

        # All values should be between 0 and 1
        assert ranked.min().min() >= 0
        assert ranked.max().max() <= 1

        # Each row should have unique ranks
        for _, row in ranked.iterrows():
            assert len(row.dropna().unique()) == len(row.dropna())

    def test_delta(self, sample_data):
        """Test delta operator computes differences."""
        _, _, _, close, _ = sample_data
        d = delta(close, 5)

        # First 5 rows should be NaN
        assert d.iloc[:5].isna().all().all()

        # Check a specific value (values match, ignore name)
        expected = close.iloc[10] - close.iloc[5]
        np.testing.assert_array_almost_equal(d.iloc[10].values, expected.values)

    def test_delay(self, sample_data):
        """Test delay operator shifts values."""
        _, _, _, close, _ = sample_data
        delayed = delay(close, 3)

        # First 3 rows should be NaN
        assert delayed.iloc[:3].isna().all().all()

        # Check values match
        pd.testing.assert_frame_equal(delayed.iloc[3:], close.iloc[:-3].reset_index(drop=True).set_index(close.index[3:]))

    def test_ts_sum(self, sample_data):
        """Test ts_sum operator computes rolling sum."""
        _, _, _, close, _ = sample_data
        ts_s = ts_sum(close, 5)

        # First 4 rows should be NaN
        assert ts_s.iloc[:4].isna().all().all()

        # Check a specific value (values match, ignore name)
        expected = close.iloc[5:10].sum()
        np.testing.assert_array_almost_equal(ts_s.iloc[9].values, expected.values)

    def test_stddev(self, sample_data):
        """Test stddev operator computes rolling std."""
        _, _, _, close, _ = sample_data
        std = stddev(close, 20)

        # First 19 rows should be NaN
        assert std.iloc[:19].isna().all().all()

        # All values should be positive
        assert (std.iloc[19:] >= 0).all().all()

    def test_decay_linear(self, sample_data):
        """Test decay_linear operator computes weighted average."""
        _, _, _, close, _ = sample_data
        decayed = decay_linear(close, 5)

        # First 4 rows should be NaN
        assert decayed.iloc[:4].isna().all().all()

        # Manual check for specific row
        weights = np.array([1, 2, 3, 4, 5]) / 15
        expected = (close.iloc[0:5] * weights.reshape(-1, 1)).sum()
        pd.testing.assert_series_equal(decayed.iloc[4], expected, check_names=False)

    def test_ts_argmax(self, sample_data):
        """Test ts_argmax returns correct day."""
        _, _, _, close, _ = sample_data
        argmax = ts_argmax(close, 10)

        # First 9 rows should be NaN
        assert argmax.iloc[:9].isna().all().all()

        # Values should be between 1 and 10
        valid = argmax.iloc[9:].dropna()
        assert (valid >= 1).all().all()
        assert (valid <= 10).all().all()


class TestSimpleAlphas:
    """Test simple alphas (no industry neutralization)."""

    def test_alpha001(self, calculator):
        """Test Alpha #1 computes without error."""
        alpha = calculator.compute(1)
        assert alpha.shape == calculator.data.close.shape
        assert not alpha.isna().all().all()

    def test_alpha002(self, calculator):
        """Test Alpha #2 computes without error."""
        alpha = calculator.compute(2)
        assert alpha.shape == calculator.data.close.shape

    def test_alpha003(self, calculator):
        """Test Alpha #3 computes without error."""
        alpha = calculator.compute(3)
        assert alpha.shape == calculator.data.close.shape

    def test_alpha101(self, calculator):
        """Test Alpha #101 formula."""
        alpha = calculator.compute(101)

        # Manual calculation
        close = calculator.data.close
        open_ = calculator.data.open_
        high = calculator.data.high
        low = calculator.data.low

        expected = (close - open_) / ((high - low) + 0.001)

        pd.testing.assert_frame_equal(alpha, expected)

    def test_all_simple_alphas_compute(self, calculator):
        """Test all simple alphas compute without error."""
        results = calculator.compute_simple(skip_errors=True)  # Allow some to fail

        # Should have most simple alphas (some may fail due to insufficient data)
        assert len(results) >= 70

        # All returned should be DataFrames
        for alpha_id, alpha_df in results.items():
            if alpha_df is not None:
                assert isinstance(alpha_df, pd.DataFrame), f"Alpha {alpha_id} is not DataFrame"
                assert alpha_df.shape == calculator.data.close.shape


class TestAlpha101Calculator:
    """Test the main calculator class."""

    def test_init(self, sample_data):
        """Test calculator initialization."""
        open_, high, low, close, volume = sample_data
        calc = Alpha101Calculator(
            open_=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
        )
        assert calc.simple_alphas is not None
        assert calc.industry_alphas is None  # No industry data

    def test_compute_caching(self, calculator):
        """Test that results are cached."""
        alpha1_first = calculator.compute(1)
        alpha1_second = calculator.compute(1)

        # Should be the same object (cached)
        assert alpha1_first is alpha1_second

    def test_clear_cache(self, calculator):
        """Test cache clearing."""
        calculator.compute(1)
        assert 1 in calculator._cache

        calculator.clear_cache()
        assert len(calculator._cache) == 0

    def test_to_features(self, calculator):
        """Test feature matrix generation."""
        calculator.compute_batch([1, 2, 3, 101])
        features = calculator.to_features()

        # Should have 4 columns
        assert len(features.columns) == 4
        assert 'alpha_001' in features.columns
        assert 'alpha_101' in features.columns

    def test_from_dataframe(self):
        """Test creation from long-format DataFrame."""
        np.random.seed(42)

        # Create long-format data
        dates = pd.date_range('2024-01-01', periods=50, freq='B')
        stocks = ['A', 'B', 'C']

        rows = []
        for date in dates:
            for stock in stocks:
                rows.append({
                    'date': date,
                    'stock_code': stock,
                    'open': 100 + np.random.randn() * 5,
                    'high': 105 + np.random.randn() * 5,
                    'low': 95 + np.random.randn() * 5,
                    'close': 100 + np.random.randn() * 5,
                    'volume': 1000000 + np.random.randint(0, 500000),
                })

        df = pd.DataFrame(rows)

        calc = Alpha101Calculator.from_dataframe(df)
        assert calc.data.close.shape == (50, 3)

        # Should be able to compute alphas
        alpha101 = calc.compute(101)
        assert alpha101.shape == (50, 3)


class TestAlphaValues:
    """Test that alpha values are reasonable."""

    def test_alpha_values_finite(self, calculator):
        """Test that alpha values are mostly finite."""
        # Only test alphas that don't need long history
        short_history_alphas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 101]
        results = calculator.compute_batch(short_history_alphas)

        for alpha_id, alpha_df in results.items():
            if alpha_df is None:
                continue

            # Allow some NaN due to warmup, but most should be finite
            valid_pct = alpha_df.notna().sum().sum() / alpha_df.size
            assert valid_pct > 0.3, f"Alpha {alpha_id} has too many NaN: {1-valid_pct:.1%}"

    def test_ranked_alphas_bounded(self, calculator):
        """Test that rank-based alphas are bounded."""
        # Alpha 1 uses rank
        alpha = calculator.compute(1)
        valid = alpha.dropna()

        if len(valid) > 0:
            # Filter out infinite values (can happen due to division)
            finite_vals = valid.stack()
            finite_vals = finite_vals[np.isfinite(finite_vals)]
            if len(finite_vals) > 0:
                # Should be roughly bounded
                assert finite_vals.abs().max() < 100, \
                    f"Alpha 1 has unusually large finite values"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
