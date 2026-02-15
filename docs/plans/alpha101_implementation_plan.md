# Alpha101 Implementation Plan

## Overview

This document provides a comprehensive implementation plan for the WorldQuant 101 Formulaic Alphas based on the paper "101 Formulaic Alphas" by Zura Kakushadze (arXiv:1601.00991).

### References
- Original Paper: https://arxiv.org/abs/1601.00991
- GitHub Implementation 1: https://github.com/yli188/WorldQuant_alpha101_code
- GitHub Implementation 2: https://github.com/Harvey-Sun/World_Quant_Alphas
- DolphinDB Documentation: https://docs.dolphindb.com/en/Tutorials/wq101alpha.html

---

## 1. Data Requirements

### 1.1 Primary Input Data (OHLCV)

| Field | Description | Source |
|-------|-------------|--------|
| `open` | Opening price | DailyPrice table |
| `high` | Highest price | DailyPrice table |
| `low` | Lowest price | DailyPrice table |
| `close` | Closing price | DailyPrice table |
| `volume` | Trading volume | DailyPrice table |

### 1.2 Derived Data

| Field | Formula | Description |
|-------|---------|-------------|
| `returns` | `close / delay(close, 1) - 1` | Daily returns |
| `vwap` | `sum(price * volume) / sum(volume)` | Volume-weighted average price |
| `adv{d}` | `sma(volume, d)` | d-day average daily volume (e.g., adv5, adv10, adv20, adv60, adv120, adv180) |

### 1.3 Optional Data (for certain alphas)

| Field | Description | Required By |
|-------|-------------|-------------|
| `cap` | Market capitalization | Alpha #56 |
| `IndClass.sector` | Sector classification | Alphas #48, 58, 63, 67, 69, 70, 76, 79, 80, 82, 87, 89, 90, 91, 93, 97, 100 |
| `IndClass.industry` | Industry classification | Same as above |
| `IndClass.subindustry` | Sub-industry classification | Same as above |

---

## 2. Operator Definitions

### 2.1 Time-Series Operators

```python
def delay(x: pd.Series, d: int) -> pd.Series:
    """Lag operator: returns value d days ago.

    Formula: delay(x, d) = x[t-d]
    """
    return x.shift(d)


def delta(x: pd.Series, d: int) -> pd.Series:
    """Difference operator: returns change over d days.

    Formula: delta(x, d) = x[t] - x[t-d]
    """
    return x - x.shift(d)


def ts_sum(x: pd.Series, d: int) -> pd.Series:
    """Rolling sum over d days.

    Formula: ts_sum(x, d) = sum(x[t-d+1:t+1])
    """
    return x.rolling(window=d, min_periods=d).sum()


def sma(x: pd.Series, d: int) -> pd.Series:
    """Simple moving average over d days.

    Formula: sma(x, d) = ts_sum(x, d) / d
    """
    return x.rolling(window=d, min_periods=d).mean()


def stddev(x: pd.Series, d: int) -> pd.Series:
    """Rolling standard deviation over d days.

    Formula: stddev(x, d) = sqrt(variance(x[t-d+1:t+1]))
    """
    return x.rolling(window=d, min_periods=d).std()


def ts_min(x: pd.Series, d: int) -> pd.Series:
    """Rolling minimum over d days.

    Formula: ts_min(x, d) = min(x[t-d+1:t+1])
    """
    return x.rolling(window=d, min_periods=d).min()


def ts_max(x: pd.Series, d: int) -> pd.Series:
    """Rolling maximum over d days.

    Formula: ts_max(x, d) = max(x[t-d+1:t+1])
    """
    return x.rolling(window=d, min_periods=d).max()


def ts_argmax(x: pd.Series, d: int) -> pd.Series:
    """Rolling argmax: returns the day on which ts_max occurred.

    Returns position (1 to d) where 1 is oldest, d is most recent.
    Formula: ts_argmax(x, d) = argmax(x[t-d+1:t+1]) + 1
    """
    return x.rolling(window=d, min_periods=d).apply(
        lambda arr: arr.argmax() + 1, raw=True
    )


def ts_argmin(x: pd.Series, d: int) -> pd.Series:
    """Rolling argmin: returns the day on which ts_min occurred.

    Returns position (1 to d) where 1 is oldest, d is most recent.
    Formula: ts_argmin(x, d) = argmin(x[t-d+1:t+1]) + 1
    """
    return x.rolling(window=d, min_periods=d).apply(
        lambda arr: arr.argmin() + 1, raw=True
    )


def ts_rank(x: pd.Series, d: int) -> pd.Series:
    """Rolling rank of current value within the past d values.

    Formula: ts_rank(x, d) = rank(x[t]) within x[t-d+1:t+1], scaled to [0, 1]
    """
    def _rank(arr):
        return pd.Series(arr).rank(pct=True).iloc[-1]
    return x.rolling(window=d, min_periods=d).apply(_rank, raw=True)


def product(x: pd.Series, d: int) -> pd.Series:
    """Rolling product over d days.

    Formula: product(x, d) = prod(x[t-d+1:t+1])
    """
    return x.rolling(window=d, min_periods=d).apply(np.prod, raw=True)


def decay_linear(x: pd.Series, d: int) -> pd.Series:
    """Weighted moving average with linearly decaying weights.

    Weights: d, d-1, ..., 2, 1 (oldest to newest, normalized)
    Formula: decay_linear(x, d) = sum(w[i] * x[t-d+i+1]) / sum(w)
    where w = [d, d-1, ..., 1]
    """
    weights = np.arange(1, d + 1, dtype=float)  # [1, 2, ..., d]
    weights = weights / weights.sum()

    def _weighted_mean(arr):
        return np.dot(arr, weights)

    return x.rolling(window=d, min_periods=d).apply(_weighted_mean, raw=True)
```

### 2.2 Cross-Sectional Operators

```python
def rank(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank, scaled to [0, 1].

    Formula: rank(x) = percentile_rank(x) across all stocks at time t
    """
    return x.rank(axis=1, pct=True)


def scale(x: pd.Series, a: float = 1.0) -> pd.Series:
    """Scale values so that sum(abs(x)) = a.

    Formula: scale(x, a) = x * a / sum(|x|)
    """
    return x * a / x.abs().sum()


def indneutralize(x: pd.DataFrame, group: pd.Series) -> pd.DataFrame:
    """Industry neutralize: demean within each industry group.

    Formula: indneutralize(x, g) = x - mean(x) for each group g
    """
    return x.groupby(group).transform(lambda g: g - g.mean())
```

### 2.3 Statistical Operators

```python
def correlation(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    """Rolling Pearson correlation between x and y over d days.

    Formula: correlation(x, y, d) = corr(x[t-d+1:t+1], y[t-d+1:t+1])
    """
    return x.rolling(window=d, min_periods=d).corr(y)


def covariance(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    """Rolling covariance between x and y over d days.

    Formula: covariance(x, y, d) = cov(x[t-d+1:t+1], y[t-d+1:t+1])
    """
    return x.rolling(window=d, min_periods=d).cov(y)
```

### 2.4 Mathematical Operators

```python
def sign(x: pd.Series) -> pd.Series:
    """Sign function: returns -1, 0, or 1."""
    return np.sign(x)


def signedpower(x: pd.Series, a: float) -> pd.Series:
    """Signed power: preserves sign while raising to power.

    Formula: signedpower(x, a) = sign(x) * |x|^a
    """
    return np.sign(x) * (np.abs(x) ** a)


def log(x: pd.Series) -> pd.Series:
    """Natural logarithm."""
    return np.log(x)


def abs_val(x: pd.Series) -> pd.Series:
    """Absolute value."""
    return np.abs(x)
```

---

## 3. All 101 Alpha Formulas

### Group 1: Simple Price-Volume Alphas (No Industry Data Required)

#### Alpha #1
```
Formula: rank(ts_argmax(signedpower(((returns < 0) ? stddev(returns, 20) : close), 2), 5)) - 0.5

Description: Ranks the position of max squared value within 5 days.
If returns negative, use stddev; otherwise use close price.
```

#### Alpha #2
```
Formula: -1 * correlation(rank(delta(log(volume), 2)), rank((close - open) / open), 6)

Description: Negative correlation between volume changes and intraday returns.
```

#### Alpha #3
```
Formula: -1 * correlation(rank(open), rank(volume), 10)

Description: Negative correlation between ranked open prices and ranked volume.
```

#### Alpha #4
```
Formula: -1 * ts_rank(rank(low), 9)

Description: Negative time-series rank of cross-sectional ranked low prices.
```

#### Alpha #5
```
Formula: rank(open - (ts_sum(vwap, 10) / 10)) * (-1 * abs(rank(close - vwap)))

Description: Combines open-VWAP deviation with close-VWAP deviation.
```

#### Alpha #6
```
Formula: -1 * correlation(open, volume, 10)

Description: Negative correlation between open price and volume.
```

#### Alpha #7
```
Formula: (adv20 < volume) ? (-1 * ts_rank(abs(delta(close, 7)), 60) * sign(delta(close, 7))) : -1

Description: Conditional alpha based on volume vs average volume.
```

#### Alpha #8
```
Formula: -1 * rank((ts_sum(open, 5) * ts_sum(returns, 5)) - delay((ts_sum(open, 5) * ts_sum(returns, 5)), 10))

Description: Change in product of cumulative open prices and returns.
```

#### Alpha #9
```
Formula: (0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : -1 * delta(close, 1))

Description: Momentum reversal based on recent price changes.
```

#### Alpha #10
```
Formula: rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : -1 * delta(close, 1))))

Description: Ranked version of Alpha #9 with 4-day window.
```

#### Alpha #11
```
Formula: (rank(ts_max(vwap - close, 3)) + rank(ts_min(vwap - close, 3))) * rank(delta(volume, 3))

Description: VWAP spread combined with volume momentum.
```

#### Alpha #12
```
Formula: sign(delta(volume, 1)) * (-1 * delta(close, 1))

Description: Price reversal based on volume direction.
```

#### Alpha #13
```
Formula: -1 * rank(covariance(rank(close), rank(volume), 5))

Description: Negative ranked covariance of ranked close and volume.
```

#### Alpha #14
```
Formula: -1 * rank(delta(returns, 3)) * correlation(open, volume, 10)

Description: Return momentum times open-volume correlation.
```

#### Alpha #15
```
Formula: -1 * ts_sum(rank(correlation(rank(high), rank(volume), 3)), 3)

Description: Sum of ranked correlations between high prices and volume.
```

#### Alpha #16
```
Formula: -1 * rank(covariance(rank(high), rank(volume), 5))

Description: Negative ranked covariance of ranked high and volume.
```

#### Alpha #17
```
Formula: -1 * rank(ts_rank(close, 10)) * rank(delta(delta(close, 1), 1)) * rank(ts_rank(volume / adv20, 5))

Description: Triple factor: price momentum, acceleration, volume momentum.
```

#### Alpha #18
```
Formula: -1 * rank((stddev(abs(close - open), 5) + (close - open)) + correlation(close, open, 10))

Description: Intraday volatility plus gap plus close-open correlation.
```

#### Alpha #19
```
Formula: -1 * sign((close - delay(close, 7)) + delta(close, 7)) * (1 + rank(1 + ts_sum(returns, 250)))

Description: Weekly momentum with long-term return ranking.
```

#### Alpha #20
```
Formula: -1 * rank(open - delay(high, 1)) * rank(open - delay(close, 1)) * rank(open - delay(low, 1))

Description: Gap analysis from previous day's price range.
```

#### Alpha #21
```
Formula: ((sma(close, 8) + stddev(close, 8)) < sma(close, 2)) ? -1 : (((sma(volume, 20) / volume) < 1) ? -1 : 1)

Description: Conditional based on price trend and volume.
```

#### Alpha #22
```
Formula: -1 * delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))

Description: Change in high-volume correlation times volatility rank.
```

#### Alpha #23
```
Formula: (sma(high, 20) < high) ? -1 * delta(high, 2) : 0

Description: Conditional high price momentum.
```

#### Alpha #24
```
Formula: ((delta(sma(close, 100), 100) / delay(close, 100)) <= 0.05) ? -1 * (close - ts_min(close, 100)) : -1 * delta(close, 3)

Description: Long-term trend versus short-term momentum.
```

#### Alpha #25
```
Formula: rank(-1 * returns * adv20 * vwap * (high - close))

Description: Multi-factor combination of returns, volume, VWAP, range.
```

#### Alpha #26
```
Formula: -1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)

Description: Max correlation of ranked volume and high prices.
```

#### Alpha #27
```
Formula: (0.5 < rank(sma(correlation(rank(volume), rank(vwap), 6), 2) / 2.0)) ? -1 : 1

Description: Binary signal based on volume-VWAP correlation.
```

#### Alpha #28
```
Formula: scale(correlation(adv20, low, 5) + ((high + low) / 2) - close)

Description: Scaled combination of ADV correlation and midpoint deviation.
```

#### Alpha #29
```
Formula: ts_min(rank(rank(scale(log(ts_sum(rank(rank(-1 * rank(delta(close - 1, 5)))), 2))))), 5) + ts_rank(delay(-1 * returns, 6), 5)

Description: Complex nested ranking with lagged returns.
```

#### Alpha #30
```
Formula: (1.0 - rank(sign(delta(close, 1)) + sign(delay(delta(close, 1), 1)) + sign(delay(delta(close, 1), 2)))) * ts_sum(volume, 5) / ts_sum(volume, 20)

Description: Trend consistency times volume ratio.
```

#### Alpha #31
```
Formula: rank(rank(rank(decay_linear(-1 * rank(rank(delta(close, 10))), 10)))) + rank(-1 * delta(close, 3)) + sign(scale(correlation(adv20, low, 12)))

Description: Complex decay with delta and correlation.
```

#### Alpha #32
```
Formula: scale(sma(close, 7) / 7 - close) + 20 * scale(correlation(vwap, delay(close, 5), 230))

Description: Short-term mean reversion plus long-term VWAP correlation.
```

#### Alpha #33
```
Formula: rank(-1 + (open / close))

Description: Ranked intraday return (negative).
```

#### Alpha #34
```
Formula: rank(2 - rank(stddev(returns, 2) / stddev(returns, 5)) - rank(delta(close, 1)))

Description: Volatility ratio combined with price momentum.
```

#### Alpha #35
```
Formula: ts_rank(volume, 32) * (1 - ts_rank(close + high - low, 16)) * (1 - ts_rank(returns, 32))

Description: Triple factor with volume, range, and returns.
```

#### Alpha #36
```
Formula: 2.21 * rank(correlation(close - open, delay(volume, 1), 15)) + 0.7 * rank(open - close) + 0.73 * rank(ts_rank(delay(-1 * returns, 6), 5)) + rank(abs(correlation(vwap, adv20, 6))) + 0.6 * rank((sma(close, 200) / 200 - open) * (close - open))

Description: Weighted combination of 5 factors.
```

#### Alpha #37
```
Formula: rank(correlation(delay(open - close, 1), close, 200)) + rank(open - close)

Description: Long-term gap correlation plus current gap.
```

#### Alpha #38
```
Formula: -1 * rank(ts_rank(open, 10)) * rank(close / open)

Description: Open price momentum times intraday return.
```

#### Alpha #39
```
Formula: -1 * rank(delta(close, 7) * (1 - rank(decay_linear(volume / adv20, 9)))) * (1 + rank(sma(returns, 250)))

Description: Weekly momentum with volume decay and long-term returns.
```

#### Alpha #40
```
Formula: -1 * rank(stddev(high, 10)) * correlation(high, volume, 10)

Description: High price volatility times high-volume correlation.
```

#### Alpha #41
```
Formula: sqrt(high * low) - vwap

Description: Geometric mean of high-low minus VWAP.
```

#### Alpha #42
```
Formula: rank(vwap - close) / rank(vwap + close)

Description: VWAP deviation ratio.
Note: Delay-0 alpha (traded at close).
```

#### Alpha #43
```
Formula: ts_rank(volume / adv20, 20) * ts_rank(-1 * delta(close, 7), 8)

Description: Normalized volume momentum times price reversal.
```

#### Alpha #44
```
Formula: -1 * correlation(high, rank(volume), 5)

Description: Negative correlation of high prices and ranked volume.
```

#### Alpha #45
```
Formula: -1 * rank(sma(delay(close, 5), 20)) * correlation(close, volume, 2) * rank(correlation(ts_sum(close, 5), ts_sum(close, 20), 2))

Description: Lagged price momentum with volume correlation.
```

#### Alpha #46
```
Formula: ((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0.25) ? (inner < 0 ? 1 : -1 * delta(close)) : -1 * delta(close)

Description: Momentum acceleration conditional.
```

#### Alpha #47
```
Formula: ((rank(1 / close) * volume / adv20) * ((high * rank(high - close)) / (sma(high, 5) / 5))) - rank(vwap - delay(vwap, 5))

Description: Complex price-volume-VWAP combination.
```

#### Alpha #49
```
Formula: ((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < -0.1) ? 1 : -1 * delta(close)

Description: Momentum acceleration conditional (threshold -0.1).
```

#### Alpha #50
```
Formula: -1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5)

Description: Max correlation of ranked volume and VWAP.
```

#### Alpha #51
```
Formula: ((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < -0.05) ? 1 : -1 * delta(close)

Description: Momentum acceleration conditional (threshold -0.05).
```

#### Alpha #52
```
Formula: (-1 * delta(ts_min(low, 5), 5)) * rank((ts_sum(returns, 240) - ts_sum(returns, 20)) / 220) * ts_rank(volume, 5)

Description: Low price momentum with long-term return differential.
```

#### Alpha #53
```
Formula: -1 * delta(((close - low) - (high - close)) / (close - low), 9)

Description: Change in intraday price position.
Note: Delay-0 alpha (traded at close).
```

#### Alpha #54
```
Formula: -1 * (low - close) * (open ^ 5) / ((low - high) * (close ^ 5))

Description: Price range ratio with power scaling.
Note: Delay-0 alpha (traded at close).
```

#### Alpha #55
```
Formula: -1 * correlation(rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))), rank(volume), 6)

Description: Correlation of normalized price position and volume.
```

#### Alpha #57
```
Formula: -1 * (close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2)

Description: VWAP deviation normalized by decayed argmax.
```

#### Alpha #60
```
Formula: -1 * ((2 * scale(rank(((close - low) - (high - close)) / (high - low) * volume))) - scale(rank(ts_argmax(close, 10))))

Description: Volume-weighted price position minus price momentum.
```

#### Alpha #61
```
Formula: rank(vwap - ts_min(vwap, 16)) < rank(correlation(vwap, adv180, 18))

Description: Boolean comparison of VWAP momentum and correlation.
```

#### Alpha #62
```
Formula: (rank(correlation(vwap, sma(adv20, 22), 10)) < rank(((rank(open) + rank(open)) < (rank((high + low) / 2) + rank(high))))) * -1

Description: Complex correlation vs price comparison.
```

#### Alpha #64
```
Formula: (rank(correlation(sma((open * 0.178404 + low * (1 - 0.178404)), 13), sma(adv120, 13), 17)) < rank(delta(((high + low) / 2 * 0.178404 + vwap * (1 - 0.178404)), 3.69741))) * -1

Description: Weighted price correlation vs delta.
```

#### Alpha #65
```
Formula: (rank(correlation((open * 0.00817205 + vwap * (1 - 0.00817205)), sma(adv60, 9), 6)) < rank(open - ts_min(open, 14))) * -1

Description: Weighted price correlation vs open momentum.
```

#### Alpha #66
```
Formula: (rank(decay_linear(delta(vwap, 4), 7)) + ts_rank(decay_linear(((low * 0.96633 + low * (1 - 0.96633)) - vwap) / (open - (high + low) / 2), 11), 7)) * -1

Description: VWAP momentum with complex ratio.
```

#### Alpha #68
```
Formula: (ts_rank(correlation(rank(high), rank(adv15), 9), 14) < rank(delta((close * 0.518371 + low * (1 - 0.518371)), 1.06157))) * -1

Description: High-ADV correlation vs price delta.
```

#### Alpha #71
```
Formula: max(ts_rank(decay_linear(correlation(ts_rank(close, 3), ts_rank(adv180, 12), 18), 4), 16), ts_rank(decay_linear(rank(((low + open) - (vwap + vwap))).pow(2), 16), 4))

Description: Max of two complex decay correlations.
```

#### Alpha #72
```
Formula: rank(decay_linear(correlation((high + low) / 2, adv40, 9), 10)) / rank(decay_linear(correlation(ts_rank(vwap, 4), ts_rank(volume, 19), 7), 3))

Description: Ratio of midpoint-ADV and VWAP-volume correlations.
```

#### Alpha #73
```
Formula: -1 * max(rank(decay_linear(delta(vwap, 5), 3)), ts_rank(decay_linear((-1 * delta((open * 0.147155 + low * (1 - 0.147155)), 2) / (open * 0.147155 + low * (1 - 0.147155))), 3), 17))

Description: Max of VWAP momentum and weighted return.
```

#### Alpha #74
```
Formula: (rank(correlation(close, sma(adv30, 37), 15)) < rank(correlation(rank(high * 0.0261661 + vwap * (1 - 0.0261661)), rank(volume), 11))) * -1

Description: Close-ADV correlation vs weighted VWAP-volume correlation.
```

#### Alpha #75
```
Formula: rank(correlation(vwap, volume, 4)) < rank(correlation(rank(low), rank(adv50), 12))

Description: VWAP-volume vs low-ADV correlation comparison.
```

#### Alpha #77
```
Formula: min(rank(decay_linear(((high + low) / 2 + high) - (vwap + high), 20)), rank(decay_linear(correlation((high + low) / 2, adv40, 3), 6)))

Description: Min of price deviation and midpoint-ADV correlation.
```

#### Alpha #78
```
Formula: rank(correlation(ts_sum((low * 0.352233 + vwap * (1 - 0.352233)), 20), ts_sum(adv40, 20), 7)).pow(rank(correlation(rank(vwap), rank(volume), 6)))

Description: Power of weighted price-ADV correlation.
```

#### Alpha #81
```
Formula: (rank(log(product(rank(rank(correlation(vwap, ts_sum(adv10, 50), 8)).pow(4)), 15))) < rank(correlation(rank(vwap), rank(volume), 5))) * -1

Description: Log product of correlations vs VWAP-volume correlation.
```

#### Alpha #83
```
Formula: (rank(delay((high - low) / (ts_sum(close, 5) / 5), 2)) * rank(rank(volume))) / (((high - low) / (ts_sum(close, 5) / 5)) / (vwap - close))

Description: Complex ratio involving range, volume, and VWAP.
```

#### Alpha #84
```
Formula: signedpower(ts_rank(vwap - ts_max(vwap, 15), 21), delta(close, 5))

Description: VWAP momentum raised to price change power.
```

#### Alpha #85
```
Formula: rank(correlation((high * 0.876703 + close * (1 - 0.876703)), adv30, 10)).pow(rank(correlation(ts_rank((high + low) / 2, 4), ts_rank(volume, 10), 7)))

Description: Weighted high-ADV correlation power.
```

#### Alpha #86
```
Formula: (ts_rank(correlation(close, sma(adv20, 15), 6), 20) < rank((open + close) - (vwap + open))) * -1

Description: Close-ADV correlation vs price deviation.
```

#### Alpha #88
```
Formula: min(rank(decay_linear((rank(open) + rank(low)) - (rank(high) + rank(close)), 8)), ts_rank(decay_linear(correlation(ts_rank(close, 8), ts_rank(adv60, 21), 8), 7), 3))

Description: Min of ranked price spread and decay correlation.
```

#### Alpha #92
```
Formula: min(ts_rank(decay_linear(((high + low) / 2 + close) < (low + open), 15), 19), ts_rank(decay_linear(correlation(rank(low), rank(adv30), 8), 7), 7))

Description: Min of price condition and low-ADV correlation.
```

#### Alpha #94
```
Formula: (rank(vwap - ts_min(vwap, 12)).pow(ts_rank(correlation(ts_rank(vwap, 20), ts_rank(adv60, 4), 18), 3))) * -1

Description: VWAP momentum power of correlation rank.
```

#### Alpha #95
```
Formula: rank(open - ts_min(open, 12)) < ts_rank(rank(correlation(sma((high + low) / 2, 19), sma(adv40, 19), 13)).pow(5), 12)

Description: Open momentum vs midpoint-ADV correlation.
```

#### Alpha #96
```
Formula: -1 * max(ts_rank(decay_linear(correlation(rank(vwap), rank(volume), 4), 4), 8), ts_rank(decay_linear(ts_argmax(correlation(ts_rank(close, 7), ts_rank(adv60, 4), 4), 13), 14), 13))

Description: Max of VWAP-volume and argmax correlations.
```

#### Alpha #98
```
Formula: rank(decay_linear(correlation(vwap, sma(adv5, 26), 5), 7)) - rank(decay_linear(ts_rank(ts_argmin(correlation(rank(open), rank(adv15), 21), 9), 7), 8))

Description: VWAP-ADV correlation minus argmin rank.
```

#### Alpha #99
```
Formula: (rank(correlation(ts_sum((high + low) / 2, 20), ts_sum(adv60, 20), 9)) < rank(correlation(low, volume, 6))) * -1

Description: Midpoint-ADV correlation vs low-volume correlation.
```

#### Alpha #101
```
Formula: (close - open) / ((high - low) + 0.001)

Description: Intraday return normalized by range.
```

---

### Group 2: Alphas Requiring Industry Classification (IndNeutralize)

These alphas require sector/industry classification data and use `indneutralize()` to demean values within industry groups.

#### Alpha #48
```
Formula: indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250)

Description: Industry-neutralized momentum correlation.
Note: Delay-0 alpha (traded at close).
```

#### Alpha #56
```
Formula: -1 * (rank((ts_sum(returns, 10) / ts_sum(ts_sum(returns, 2), 3))) * rank(returns * cap))

Description: Return ratio times market-cap weighted returns.
Requires: Market capitalization (cap)
```

#### Alpha #58
```
Formula: -1 * ts_rank(decay_linear(correlation(indneutralize(vwap, IndClass.sector), volume, 3.92795), 7.89291), 5.50322)

Description: Decay correlation of sector-neutralized VWAP and volume.
```

#### Alpha #59
```
Formula: -1 * ts_rank(decay_linear(correlation(indneutralize((vwap * 0.728317 + vwap * (1 - 0.728317)), IndClass.industry), volume, 4.25197), 16.2289), 8.19648)

Description: Industry-neutralized VWAP-volume correlation.
```

#### Alpha #63
```
Formula: (rank(decay_linear(delta(indneutralize(close, IndClass.industry), 2.25164), 8.22237)) - rank(decay_linear(correlation((vwap * 0.318108 + open * (1 - 0.318108)), ts_sum(adv180, 37.2467), 13.557), 12.2883))) * -1

Description: Industry-neutralized close momentum vs weighted correlation.
```

#### Alpha #67
```
Formula: (rank(high - ts_min(high, 2.14593)).pow(rank(correlation(indneutralize(vwap, IndClass.sector), indneutralize(adv20, IndClass.subindustry), 6.02936)))) * -1

Description: High momentum power of cross-neutralized correlation.
```

#### Alpha #69
```
Formula: (rank(ts_max(delta(indneutralize(vwap, IndClass.industry), 2.72412), 4.79344)).pow(ts_rank(correlation((close * 0.490655 + vwap * (1 - 0.490655)), adv20, 4.92416), 9.0615))) * -1

Description: Industry-neutralized VWAP momentum power.
```

#### Alpha #70
```
Formula: (rank(delta(vwap, 1.29456)).pow(ts_rank(correlation(indneutralize(close, IndClass.industry), adv50, 17.8256), 17.9171))) * -1

Description: VWAP momentum power of industry-neutralized correlation.
```

#### Alpha #76
```
Formula: -1 * max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)), ts_rank(decay_linear(ts_rank(correlation(indneutralize(low, IndClass.sector), adv81, 8.14941), 19.569), 17.1543), 19.383))

Description: Max of VWAP decay and sector-neutralized low-ADV correlation.
```

#### Alpha #79
```
Formula: rank(delta(indneutralize((close * 0.60733 + open * (1 - 0.60733)), IndClass.sector), 1.23438)) < rank(correlation(ts_rank(vwap, 3.60973), ts_rank(adv150, 9.18637), 14.6644))

Description: Sector-neutralized price momentum vs VWAP-ADV correlation.
```

#### Alpha #80
```
Formula: (rank(sign(delta(indneutralize((open * 0.868128 + high * (1 - 0.868128)), IndClass.industry), 4.04545))).pow(ts_rank(correlation(high, adv10, 5.11456), 5.53756))) * -1

Description: Industry-neutralized price sign power.
```

#### Alpha #82
```
Formula: -1 * min(rank(decay_linear(delta(open, 1.46063), 14.8717)), ts_rank(decay_linear(correlation(indneutralize(volume, IndClass.sector), (open * 0.634196 + open * (1 - 0.634196)), 17.4842), 6.92131), 13.4283))

Description: Min of open decay and sector-neutralized volume correlation.
```

#### Alpha #87
```
Formula: -1 * max(rank(decay_linear(delta((close * 0.369701 + vwap * (1 - 0.369701)), 1.91233), 2.65461)), ts_rank(decay_linear(abs(correlation(indneutralize(adv81, IndClass.industry), close, 13.4132)), 4.89768), 14.4535))

Description: Max of weighted price decay and industry-neutralized ADV correlation.
```

#### Alpha #89
```
Formula: ts_rank(decay_linear(correlation((low * 0.967285 + low * (1 - 0.967285)), adv10, 6.94279), 5.51607), 3.79744) - ts_rank(decay_linear(delta(indneutralize(vwap, IndClass.industry), 3.48158), 10.1466), 15.3012)

Description: Low-ADV correlation minus industry-neutralized VWAP momentum.
```

#### Alpha #90
```
Formula: (rank(close - ts_max(close, 4.66719)).pow(ts_rank(correlation(indneutralize(adv40, IndClass.subindustry), low, 5.38375), 3.21856))) * -1

Description: Close momentum power of subindustry-neutralized ADV-low correlation.
```

#### Alpha #91
```
Formula: (ts_rank(decay_linear(decay_linear(correlation(indneutralize(close, IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) - rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1

Description: Double decay industry-neutralized correlation minus VWAP-ADV.
```

#### Alpha #93
```
Formula: ts_rank(decay_linear(correlation(indneutralize(vwap, IndClass.industry), adv81, 17.4193), 19.848), 7.54455) / rank(decay_linear(delta((close * 0.524434 + vwap * (1 - 0.524434)), 2.77377), 16.2664))

Description: Ratio of industry-neutralized VWAP-ADV correlation to weighted price momentum.
```

#### Alpha #97
```
Formula: (rank(decay_linear(delta(indneutralize((low * 0.721001 + vwap * (1 - 0.721001)), IndClass.industry), 3.3705), 20.4523)) - ts_rank(decay_linear(ts_rank(correlation(ts_rank(low, 7.87871), ts_rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1

Description: Industry-neutralized weighted low-VWAP momentum.
```

#### Alpha #100
```
Formula: -1 * (((1.5 * scale(indneutralize(indneutralize(rank(((close - low) - (high - close)) / (high - low) * volume), IndClass.subindustry), IndClass.subindustry))) - scale(indneutralize(correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30)), IndClass.subindustry))) * (volume / adv20))

Description: Double subindustry-neutralized complex factor.
```

---

## 4. Python Implementation Structure

### 4.1 Project Structure

```
lasps/
├── data/
│   └── processors/
│       └── alpha101/
│           ├── __init__.py
│           ├── operators.py       # All operator definitions
│           ├── alpha_base.py      # Base class for alphas
│           ├── simple_alphas.py   # Alphas 1-55, 57, 60-66, 68, 71-78, 81, 83-86, 88, 92, 94-96, 98-99, 101
│           ├── industry_alphas.py # Alphas 48, 56, 58-59, 63, 67, 69-70, 76, 79-80, 82, 87, 89-91, 93, 97, 100
│           └── calculator.py      # Main calculator class
```

### 4.2 Base Class

```python
# lasps/data/processors/alpha101/alpha_base.py
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Optional


class Alpha101Base(ABC):
    """Base class for Alpha101 factor calculation."""

    def __init__(
        self,
        open_: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        vwap: Optional[pd.DataFrame] = None,
        cap: Optional[pd.DataFrame] = None,
        sector: Optional[pd.Series] = None,
        industry: Optional[pd.Series] = None,
        subindustry: Optional[pd.Series] = None,
    ):
        """Initialize with price-volume data.

        Args:
            open_: Open prices DataFrame (date x stocks)
            high: High prices DataFrame (date x stocks)
            low: Low prices DataFrame (date x stocks)
            close: Close prices DataFrame (date x stocks)
            volume: Volume DataFrame (date x stocks)
            vwap: VWAP DataFrame (optional, will be calculated if not provided)
            cap: Market cap DataFrame (optional, required for alpha56)
            sector: Sector classification Series (optional, for industry alphas)
            industry: Industry classification Series (optional)
            subindustry: Sub-industry classification Series (optional)
        """
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

        # Calculate derived data
        self.returns = close.pct_change()
        self.vwap = vwap if vwap is not None else self._calculate_vwap()

        # Optional data
        self.cap = cap
        self.sector = sector
        self.industry = industry
        self.subindustry = subindustry

        # Pre-calculate ADV at various windows
        self._adv_cache = {}

    def _calculate_vwap(self) -> pd.DataFrame:
        """Calculate VWAP as (high + low + close) / 3 * volume / volume."""
        # Simplified VWAP approximation
        typical_price = (self.high + self.low + self.close) / 3
        return typical_price

    def adv(self, d: int) -> pd.DataFrame:
        """Get d-day average daily volume with caching."""
        if d not in self._adv_cache:
            self._adv_cache[d] = self.volume.rolling(window=d).mean()
        return self._adv_cache[d]

    @abstractmethod
    def calculate(self) -> pd.DataFrame:
        """Calculate the alpha factor. Must be implemented by subclasses."""
        pass
```

### 4.3 Main Calculator Class

```python
# lasps/data/processors/alpha101/calculator.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger

from .alpha_base import Alpha101Base
from .simple_alphas import SimpleAlphas
from .industry_alphas import IndustryAlphas


class Alpha101Calculator:
    """Calculate WorldQuant 101 Alpha factors."""

    SIMPLE_ALPHAS = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
        39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53, 54, 55, 57, 60,
        61, 62, 64, 65, 66, 68, 71, 72, 73, 74, 75, 77, 78, 81, 83, 84, 85, 86,
        88, 92, 94, 95, 96, 98, 99, 101
    ]

    INDUSTRY_ALPHAS = [48, 56, 58, 59, 63, 67, 69, 70, 76, 79, 80, 82, 87, 89, 90, 91, 93, 97, 100]

    def __init__(
        self,
        open_: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        vwap: Optional[pd.DataFrame] = None,
        cap: Optional[pd.DataFrame] = None,
        sector: Optional[pd.Series] = None,
        industry: Optional[pd.Series] = None,
        subindustry: Optional[pd.Series] = None,
    ):
        """Initialize calculator with market data."""
        self.simple_calculator = SimpleAlphas(
            open_, high, low, close, volume, vwap
        )

        self.industry_calculator = None
        if sector is not None or industry is not None or subindustry is not None:
            self.industry_calculator = IndustryAlphas(
                open_, high, low, close, volume, vwap,
                cap, sector, industry, subindustry
            )

    def calculate_alpha(self, alpha_num: int) -> pd.DataFrame:
        """Calculate a specific alpha.

        Args:
            alpha_num: Alpha number (1-101)

        Returns:
            DataFrame with alpha values

        Raises:
            ValueError: If alpha requires industry data but not provided
        """
        method_name = f"alpha{alpha_num:03d}"

        if alpha_num in self.SIMPLE_ALPHAS:
            method = getattr(self.simple_calculator, method_name, None)
            if method is None:
                raise NotImplementedError(f"Alpha #{alpha_num} not implemented")
            return method()

        elif alpha_num in self.INDUSTRY_ALPHAS:
            if self.industry_calculator is None:
                raise ValueError(f"Alpha #{alpha_num} requires industry classification data")
            method = getattr(self.industry_calculator, method_name, None)
            if method is None:
                raise NotImplementedError(f"Alpha #{alpha_num} not implemented")
            return method()

        else:
            raise ValueError(f"Invalid alpha number: {alpha_num}")

    def calculate_all(
        self,
        include_industry: bool = False,
        alpha_list: Optional[List[int]] = None
    ) -> Dict[int, pd.DataFrame]:
        """Calculate multiple alphas.

        Args:
            include_industry: Whether to include industry-neutralized alphas
            alpha_list: Specific list of alphas to calculate (default: all simple)

        Returns:
            Dictionary mapping alpha number to DataFrame
        """
        results = {}

        if alpha_list is None:
            alpha_list = self.SIMPLE_ALPHAS.copy()
            if include_industry and self.industry_calculator is not None:
                alpha_list.extend(self.INDUSTRY_ALPHAS)

        for alpha_num in alpha_list:
            try:
                results[alpha_num] = self.calculate_alpha(alpha_num)
                logger.debug(f"Calculated Alpha #{alpha_num}")
            except Exception as e:
                logger.warning(f"Failed to calculate Alpha #{alpha_num}: {e}")

        return results
```

### 4.4 Example Alpha Implementations

```python
# lasps/data/processors/alpha101/simple_alphas.py
import pandas as pd
import numpy as np
from typing import Optional

from .alpha_base import Alpha101Base
from .operators import (
    rank, delay, delta, ts_sum, sma, stddev, correlation, covariance,
    ts_min, ts_max, ts_argmax, ts_argmin, ts_rank, product, decay_linear,
    scale, sign, signedpower, log, abs_val
)


class SimpleAlphas(Alpha101Base):
    """Implementation of simple Alpha101 factors (no industry data required)."""

    def alpha001(self) -> pd.DataFrame:
        """
        rank(ts_argmax(signedpower(((returns < 0) ? stddev(returns, 20) : close), 2), 5)) - 0.5
        """
        inner = self.close.copy()
        cond = self.returns < 0
        inner[cond] = stddev(self.returns, 20)[cond]
        return rank(ts_argmax(inner ** 2, 5)) - 0.5

    def alpha002(self) -> pd.DataFrame:
        """
        -1 * correlation(rank(delta(log(volume), 2)), rank((close - open) / open), 6)
        """
        df = -1 * correlation(
            rank(delta(log(self.volume), 2)),
            rank((self.close - self.open) / self.open),
            6
        )
        return df.replace([np.inf, -np.inf], 0).fillna(0)

    def alpha003(self) -> pd.DataFrame:
        """
        -1 * correlation(rank(open), rank(volume), 10)
        """
        df = -1 * correlation(rank(self.open), rank(self.volume), 10)
        return df.replace([np.inf, -np.inf], 0).fillna(0)

    def alpha004(self) -> pd.DataFrame:
        """
        -1 * ts_rank(rank(low), 9)
        """
        return -1 * ts_rank(rank(self.low), 9)

    def alpha005(self) -> pd.DataFrame:
        """
        rank(open - (ts_sum(vwap, 10) / 10)) * (-1 * abs(rank(close - vwap)))
        """
        return rank(self.open - sma(self.vwap, 10)) * (-1 * abs_val(rank(self.close - self.vwap)))

    def alpha006(self) -> pd.DataFrame:
        """
        -1 * correlation(open, volume, 10)
        """
        df = -1 * correlation(self.open, self.volume, 10)
        return df.replace([np.inf, -np.inf], 0).fillna(0)

    # ... (remaining alpha implementations)

    def alpha101(self) -> pd.DataFrame:
        """
        (close - open) / ((high - low) + 0.001)
        """
        return (self.close - self.open) / ((self.high - self.low) + 0.001)
```

---

## 5. Integration with Stockquant Project

### 5.1 Database Schema Extension

Add a new table for storing alpha factor values:

```python
# lasps/db/models/alpha_factor.py
from sqlalchemy import Column, Date, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from lasps.db.base import Base, TimestampMixin


class AlphaFactor(TimestampMixin, Base):
    """Alpha101 factor values per stock per date."""

    __tablename__ = "alpha_factors"

    id = Column(Integer, primary_key=True)
    stock_code = Column(String(6), ForeignKey("stocks.code"), nullable=False)
    date = Column(Date, nullable=False)
    alpha_num = Column(Integer, nullable=False)  # 1-101
    value = Column(Float, nullable=True)

    stock = relationship("Stock", backref="alpha_factors")

    __table_args__ = (
        Index("idx_alpha_stock_date", "stock_code", "date", "alpha_num"),
    )
```

### 5.2 Repository Pattern

```python
# lasps/db/repositories/alpha_repo.py
from typing import Dict, List, Optional
from datetime import date
import pandas as pd
from sqlalchemy.orm import Session

from lasps.db.repositories.base_repository import BaseRepository
from lasps.db.models.alpha_factor import AlphaFactor


class AlphaRepository(BaseRepository[AlphaFactor]):
    """Repository for Alpha101 factor storage and retrieval."""

    def __init__(self, session: Session):
        super().__init__(AlphaFactor, session)

    def save_alphas(
        self,
        stock_code: str,
        dt: date,
        alpha_values: Dict[int, float]
    ) -> None:
        """Save multiple alpha values for a stock on a date."""
        for alpha_num, value in alpha_values.items():
            self.upsert(
                stock_code=stock_code,
                date=dt,
                alpha_num=alpha_num,
                value=value
            )

    def get_alpha_series(
        self,
        alpha_num: int,
        start_date: date,
        end_date: date,
        stock_codes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Get alpha values as a DataFrame (date x stocks)."""
        query = self.session.query(AlphaFactor).filter(
            AlphaFactor.alpha_num == alpha_num,
            AlphaFactor.date >= start_date,
            AlphaFactor.date <= end_date
        )
        if stock_codes:
            query = query.filter(AlphaFactor.stock_code.in_(stock_codes))

        records = query.all()
        df = pd.DataFrame([
            {"date": r.date, "stock_code": r.stock_code, "value": r.value}
            for r in records
        ])

        if df.empty:
            return pd.DataFrame()

        return df.pivot(index="date", columns="stock_code", values="value")
```

### 5.3 Script for Batch Calculation

```python
# scripts/calculate_alpha101.py
"""Calculate Alpha101 factors for all stocks."""
import argparse
from datetime import date, timedelta
from loguru import logger
import pandas as pd

from lasps.db.engine import get_session
from lasps.db.repositories.price_repo import PriceRepository
from lasps.db.repositories.alpha_repo import AlphaRepository
from lasps.data.processors.alpha101 import Alpha101Calculator


def main():
    parser = argparse.ArgumentParser(description="Calculate Alpha101 factors")
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--alphas", type=str, default="all", help="Comma-separated alpha numbers or 'all'")
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date) if args.end_date else date.today()

    # Parse alpha list
    if args.alphas == "all":
        alpha_list = Alpha101Calculator.SIMPLE_ALPHAS
    else:
        alpha_list = [int(x) for x in args.alphas.split(",")]

    with get_session() as session:
        price_repo = PriceRepository(session)
        alpha_repo = AlphaRepository(session)

        # Load OHLCV data with lookback for indicator calculation
        lookback_start = start_date - timedelta(days=300)  # 300 days for longest indicator

        logger.info(f"Loading price data from {lookback_start} to {end_date}")
        prices = price_repo.get_all_prices(lookback_start, end_date)

        # Pivot to DataFrames (date x stocks)
        open_df = prices.pivot(index="date", columns="stock_code", values="open")
        high_df = prices.pivot(index="date", columns="stock_code", values="high")
        low_df = prices.pivot(index="date", columns="stock_code", values="low")
        close_df = prices.pivot(index="date", columns="stock_code", values="close")
        volume_df = prices.pivot(index="date", columns="stock_code", values="volume")

        # Calculate alphas
        calculator = Alpha101Calculator(
            open_df, high_df, low_df, close_df, volume_df
        )

        alphas = calculator.calculate_all(alpha_list=alpha_list)

        # Save results (only for target date range)
        for alpha_num, alpha_df in alphas.items():
            alpha_df = alpha_df.loc[start_date:end_date]
            for dt in alpha_df.index:
                for stock_code in alpha_df.columns:
                    value = alpha_df.loc[dt, stock_code]
                    if pd.notna(value):
                        alpha_repo.save_alphas(stock_code, dt, {alpha_num: value})

        session.commit()
        logger.info(f"Saved {len(alphas)} alpha factors")


if __name__ == "__main__":
    main()
```

---

## 6. Testing Strategy

### 6.1 Unit Tests for Operators

```python
# tests/test_alpha101_operators.py
import pytest
import pandas as pd
import numpy as np
from lasps.data.processors.alpha101.operators import (
    delay, delta, ts_sum, ts_min, ts_max, ts_argmax, ts_rank,
    decay_linear, rank, correlation, stddev
)


class TestOperators:

    def test_delay(self):
        s = pd.Series([1, 2, 3, 4, 5])
        result = delay(s, 2)
        expected = pd.Series([np.nan, np.nan, 1, 2, 3])
        pd.testing.assert_series_equal(result, expected)

    def test_delta(self):
        s = pd.Series([1, 2, 4, 7, 11])
        result = delta(s, 1)
        expected = pd.Series([np.nan, 1, 2, 3, 4])
        pd.testing.assert_series_equal(result, expected)

    def test_ts_argmax(self):
        s = pd.Series([1, 3, 2, 5, 4])
        result = ts_argmax(s, 3)
        # Window [1,3,2]: max at idx 1 (position 2)
        # Window [3,2,5]: max at idx 2 (position 3)
        # Window [2,5,4]: max at idx 1 (position 2)
        expected = pd.Series([np.nan, np.nan, 2.0, 3.0, 2.0])
        pd.testing.assert_series_equal(result, expected)

    def test_decay_linear(self):
        s = pd.Series([1.0, 2.0, 3.0])
        result = decay_linear(s, 3)
        # weights = [1, 2, 3] / 6 = [1/6, 2/6, 3/6]
        # result[2] = 1*1/6 + 2*2/6 + 3*3/6 = 14/6 = 2.333...
        assert result.iloc[-1] == pytest.approx(14/6, rel=1e-6)

    def test_rank_cross_sectional(self):
        df = pd.DataFrame({
            'A': [1, 4, 2],
            'B': [3, 2, 5],
            'C': [2, 3, 1]
        })
        result = rank(df)
        # Row 0: A=1(rank 0.33), B=3(rank 1.0), C=2(rank 0.67)
        assert result.loc[0, 'A'] < result.loc[0, 'C'] < result.loc[0, 'B']
```

### 6.2 Validation Against Reference Implementation

```python
# tests/test_alpha101_validation.py
import pytest
import pandas as pd
import numpy as np
from lasps.data.processors.alpha101 import Alpha101Calculator


class TestAlphaValidation:
    """Validate alpha calculations against known reference values."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        stocks = ["A", "B", "C", "D", "E"]

        data = {}
        for stock in stocks:
            base_price = 100 + np.random.randn() * 20
            returns = np.random.randn(100) * 0.02
            close = base_price * np.cumprod(1 + returns)
            high = close * (1 + np.abs(np.random.randn(100) * 0.01))
            low = close * (1 - np.abs(np.random.randn(100) * 0.01))
            open_ = low + (high - low) * np.random.rand(100)
            volume = np.random.randint(1000000, 10000000, 100)

            data[stock] = {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume
            }

        open_df = pd.DataFrame({s: data[s]["open"] for s in stocks}, index=dates)
        high_df = pd.DataFrame({s: data[s]["high"] for s in stocks}, index=dates)
        low_df = pd.DataFrame({s: data[s]["low"] for s in stocks}, index=dates)
        close_df = pd.DataFrame({s: data[s]["close"] for s in stocks}, index=dates)
        volume_df = pd.DataFrame({s: data[s]["volume"] for s in stocks}, index=dates)

        return open_df, high_df, low_df, close_df, volume_df

    def test_alpha101_simple(self, sample_data):
        """Test Alpha #101: (close - open) / ((high - low) + 0.001)"""
        open_df, high_df, low_df, close_df, volume_df = sample_data

        calculator = Alpha101Calculator(open_df, high_df, low_df, close_df, volume_df)
        result = calculator.calculate_alpha(101)

        # Manual calculation
        expected = (close_df - open_df) / ((high_df - low_df) + 0.001)

        pd.testing.assert_frame_equal(result, expected)

    def test_alpha_values_bounded(self, sample_data):
        """Test that alpha values are within reasonable bounds."""
        open_df, high_df, low_df, close_df, volume_df = sample_data
        calculator = Alpha101Calculator(open_df, high_df, low_df, close_df, volume_df)

        for alpha_num in [1, 2, 3, 4, 5, 6]:
            result = calculator.calculate_alpha(alpha_num)
            # Most alphas should be bounded after rank transformation
            assert result.dropna().abs().max().max() < 100, f"Alpha #{alpha_num} has extreme values"
```

---

## 7. Implementation Timeline

### Phase 1: Core Infrastructure (Week 1)
- [ ] Implement all operator functions in `operators.py`
- [ ] Implement `Alpha101Base` class
- [ ] Set up unit tests for operators
- [ ] Create database schema for alpha storage

### Phase 2: Simple Alphas (Week 2-3)
- [ ] Implement Alphas 1-30
- [ ] Implement Alphas 31-60 (excluding industry alphas)
- [ ] Implement Alphas 61-101 (excluding industry alphas)
- [ ] Validate against reference implementations

### Phase 3: Industry Alphas (Week 4)
- [ ] Implement `indneutralize` operator
- [ ] Implement industry-neutralized alphas (48, 56, 58-59, 63, 67, 69-70, 76, 79-80, 82, 87, 89-91, 93, 97, 100)
- [ ] Add sector classification mapping

### Phase 4: Integration (Week 5)
- [ ] Create batch calculation script
- [ ] Integrate with existing data pipeline
- [ ] Performance optimization
- [ ] Documentation and examples

---

## 8. Notes and Considerations

### 8.1 Delay-0 Alphas
Alphas #42, #48, #53, and #54 are delay-0 alphas. They should be traded at or as close as possible to the close of the trading day.

### 8.2 Handling Edge Cases
- Division by zero: Use `+ 0.001` or `replace(0, np.nan)`
- Infinite values: Replace with 0 or NaN
- Missing data: Use `fillna()` or `dropna()` appropriately

### 8.3 Performance Optimization
- Pre-calculate ADV at common windows (5, 10, 15, 20, 30, 40, 50, 60, 81, 120, 150, 180)
- Use vectorized operations with pandas/numpy
- Consider using numba for time-series operations
- Cache intermediate calculations

### 8.4 Korean Market Adaptation
- Sector classification: Map KOSPI/KOSDAQ sectors to IndClass levels
- Trading hours: Adjust for Korean market hours
- Volume: Consider Korean market liquidity characteristics

---

## Appendix: Complete Alpha Formula Reference

See individual alpha formulas in Section 3 above. All 101 alphas are documented with:
- Original formula from the paper
- Description of the factor
- Required data inputs
- Implementation notes
