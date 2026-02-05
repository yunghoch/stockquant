import numpy as np
import pandas as pd
from lasps.utils.constants import (
    PREDICTION_HORIZON, LABEL_THRESHOLD,
    OHLCV_INDICES, INDICATOR_INDICES,
)


def compute_label(close_prices: pd.Series, index: int) -> int:
    if index + PREDICTION_HORIZON >= len(close_prices):
        return -1
    current = close_prices.iloc[index]
    future = close_prices.iloc[index + PREDICTION_HORIZON]
    ret = (future - current) / current
    if ret >= LABEL_THRESHOLD:
        return 2
    elif ret <= -LABEL_THRESHOLD:
        return 0
    return 1


def normalize_time_series(data: np.ndarray) -> np.ndarray:
    """Per-stock min-max normalization for OHLCV and indicator features.

    Normalizes features at indices 0-19 (OHLCV + indicators) to [0, 1] range
    using min-max scaling over the time window. Sentiment features (20-24)
    are already bounded and left unchanged.

    Args:
        data: Array of shape (T, 25) where T is time steps.

    Returns:
        Normalized array of same shape. Constant features map to 0.5.
    """
    result = data.copy()
    norm_end = INDICATOR_INDICES[1]  # 20

    for col in range(OHLCV_INDICES[0], norm_end):
        col_data = result[:, col]
        col_min = col_data.min()
        col_max = col_data.max()
        if col_max - col_min < 1e-10:
            result[:, col] = 0.5
        else:
            result[:, col] = (col_data - col_min) / (col_max - col_min)

    return result


def normalize_minmax(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx - mn < 1e-10:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)
