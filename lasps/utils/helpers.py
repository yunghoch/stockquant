import pandas as pd
from lasps.utils.constants import PREDICTION_HORIZON, LABEL_THRESHOLD


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


def normalize_minmax(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx - mn < 1e-10:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)
