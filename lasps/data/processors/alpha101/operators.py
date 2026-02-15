"""Alpha101 Operators - All 22 operators from the paper.

Based on: https://arxiv.org/abs/1601.00991
Section 2: Definitions and Notation

All operators work with pandas DataFrames where:
- Index: dates
- Columns: stock codes
"""

import numpy as np
import pandas as pd
from typing import Union
from scipy.stats import rankdata


# =============================================================================
# Cross-Sectional Operators (operate across stocks at each time point)
# =============================================================================

def rank(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank, scaled to [0, 1].

    For each row (date), ranks all stock values and scales to [0, 1].
    NaN values remain NaN.
    """
    return x.rank(axis=1, pct=True)


def scale(x: pd.DataFrame, a: float = 1.0) -> pd.DataFrame:
    """Scale to sum of absolute values equals a.

    scale(x, a) = x / sum(|x|) * a
    """
    abs_sum = x.abs().sum(axis=1).replace(0, np.nan)
    return x.div(abs_sum, axis=0) * a


def signedpower(x: Union[pd.DataFrame, pd.Series], a: float) -> Union[pd.DataFrame, pd.Series]:
    """Signed power function.

    signedpower(x, a) = sign(x) * |x|^a
    """
    return np.sign(x) * np.power(np.abs(x), a)


def log(x: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """Natural logarithm. Returns NaN for non-positive values."""
    return np.log(x.where(x > 0))


def sign(x: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """Sign function: -1, 0, or +1."""
    return np.sign(x)


def abs_(x: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """Absolute value."""
    return np.abs(x)


def max_(x: Union[pd.DataFrame, pd.Series], y: Union[pd.DataFrame, pd.Series, float]) -> Union[pd.DataFrame, pd.Series]:
    """Element-wise maximum."""
    if isinstance(y, (int, float)):
        return x.clip(lower=y)
    return np.maximum(x, y)


def min_(x: Union[pd.DataFrame, pd.Series], y: Union[pd.DataFrame, pd.Series, float]) -> Union[pd.DataFrame, pd.Series]:
    """Element-wise minimum."""
    if isinstance(y, (int, float)):
        return x.clip(upper=y)
    return np.minimum(x, y)


# =============================================================================
# Time-Series Operators (operate across time for each stock)
# =============================================================================

def delta(x: pd.DataFrame, d: int = 1) -> pd.DataFrame:
    """Time-series difference.

    delta(x, d) = x(t) - x(t-d)
    """
    return x.diff(periods=d)


def delay(x: pd.DataFrame, d: int = 1) -> pd.DataFrame:
    """Time-series lag.

    delay(x, d) = x(t-d)
    """
    return x.shift(periods=d)


def ts_min(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Time-series minimum over d days."""
    return x.rolling(window=d, min_periods=d).min()


def ts_max(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Time-series maximum over d days."""
    return x.rolling(window=d, min_periods=d).max()


def ts_argmin(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Day of ts_min (1 to d, where d is most recent)."""
    def argmin_func(arr):
        if np.isnan(arr).all():
            return np.nan
        return d - np.nanargmin(arr)
    return x.rolling(window=d, min_periods=d).apply(argmin_func, raw=True)


def ts_argmax(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Day of ts_max (1 to d, where d is most recent)."""
    def argmax_func(arr):
        if np.isnan(arr).all():
            return np.nan
        return d - np.nanargmax(arr)
    return x.rolling(window=d, min_periods=d).apply(argmax_func, raw=True)


def ts_rank(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Time-series rank over d days, scaled to [0, 1].

    Ranks the current value against the past d-1 values.
    """
    def rank_func(arr):
        if np.isnan(arr).all():
            return np.nan
        valid = ~np.isnan(arr)
        if valid.sum() < 2:
            return np.nan
        # Rank of last element among all elements
        return rankdata(arr[valid])[-1] / valid.sum()
    return x.rolling(window=d, min_periods=d).apply(rank_func, raw=True)


def ts_sum(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Time-series sum over d days."""
    return x.rolling(window=d, min_periods=d).sum()


def ts_product(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Time-series product over d days."""
    return x.rolling(window=d, min_periods=d).apply(np.prod, raw=True)


def stddev(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Time-series standard deviation over d days."""
    return x.rolling(window=d, min_periods=d).std()


def correlation(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    """Time-series correlation between x and y over d days."""
    return x.rolling(window=d, min_periods=d).corr(y)


def covariance(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    """Time-series covariance between x and y over d days."""
    return x.rolling(window=d, min_periods=d).cov(y)


def decay_linear(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Weighted moving average with linearly decaying weights.

    Weights: [1, 2, 3, ..., d] normalized to sum to 1.
    Most recent day has highest weight (d).
    """
    weights = np.arange(1, d + 1, dtype=float)
    weights = weights / weights.sum()

    def wma(arr):
        if np.isnan(arr).any():
            return np.nan
        return np.dot(arr, weights)

    return x.rolling(window=d, min_periods=d).apply(wma, raw=True)


def sma(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Simple moving average over d days."""
    return x.rolling(window=d, min_periods=d).mean()


def returns(close: pd.DataFrame) -> pd.DataFrame:
    """Daily returns: (close - close_prev) / close_prev."""
    return close.pct_change()


# =============================================================================
# Industry Neutralization
# =============================================================================

def indneutralize(x: pd.DataFrame, industry: pd.Series) -> pd.DataFrame:
    """Industry-neutralize: subtract industry mean from each stock.

    Args:
        x: DataFrame with index=dates, columns=stock_codes
        industry: Series mapping stock_code to industry_code

    Returns:
        Industry-neutralized DataFrame
    """
    result = x.copy()

    for date in x.index:
        row = x.loc[date]
        # Group by industry
        for ind_code in industry.unique():
            mask = industry == ind_code
            stocks_in_ind = industry[mask].index
            stocks_in_row = row.index.intersection(stocks_in_ind)

            if len(stocks_in_row) > 0:
                ind_mean = row[stocks_in_row].mean()
                result.loc[date, stocks_in_row] = row[stocks_in_row] - ind_mean

    return result


# =============================================================================
# Utility Functions
# =============================================================================

def where(condition: pd.DataFrame, x: Union[pd.DataFrame, pd.Series, float, int], y: Union[pd.DataFrame, pd.Series, float, int]) -> pd.DataFrame:
    """Conditional selection: where(cond, x, y) = x if cond else y."""
    # Handle scalar x
    if isinstance(x, (int, float)):
        x = pd.DataFrame(x, index=condition.index, columns=condition.columns)

    # Handle scalar y
    if isinstance(y, (int, float)):
        y = pd.DataFrame(y, index=condition.index, columns=condition.columns)

    return x.where(condition, y)


def adv(volume: pd.DataFrame, d: int) -> pd.DataFrame:
    """Average daily volume over d days."""
    return sma(volume, d)


def vwap(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Volume-weighted average price (approximation using close * volume)."""
    # True VWAP requires intraday data; this is a daily approximation
    return close  # Simplified: just use close price


def cap(market_cap: pd.DataFrame) -> pd.DataFrame:
    """Market capitalization (pass-through)."""
    return market_cap
