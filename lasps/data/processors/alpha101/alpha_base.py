"""Alpha101 Base Class - Abstract base for alpha calculations.

Provides common infrastructure for computing alphas.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional
from dataclasses import dataclass

from .operators import (
    rank, scale, signedpower, log, sign, abs_, max_, min_,
    delta, delay, ts_min, ts_max, ts_argmin, ts_argmax,
    ts_rank, ts_sum, ts_product, stddev, correlation,
    covariance, decay_linear, sma, returns, indneutralize,
    where, adv, vwap, cap
)


@dataclass
class MarketData:
    """Container for market data required by Alpha101.

    All DataFrames have:
    - Index: DatetimeIndex (trading dates)
    - Columns: stock codes

    Required fields:
    - open_: Opening price
    - high: High price
    - low: Low price
    - close: Closing price
    - volume: Trading volume

    Optional fields:
    - vwap: Volume-weighted average price
    - cap: Market capitalization
    - industry: Series mapping stock_code to industry_code
    """
    open_: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    vwap: Optional[pd.DataFrame] = None
    cap: Optional[pd.DataFrame] = None
    industry: Optional[pd.Series] = None

    def __post_init__(self):
        """Initialize derived fields."""
        # Pre-compute common values
        self._returns = None
        self._adv5 = None
        self._adv10 = None
        self._adv15 = None
        self._adv20 = None
        self._adv30 = None
        self._adv40 = None
        self._adv50 = None
        self._adv60 = None
        self._adv81 = None
        self._adv120 = None
        self._adv150 = None
        self._adv180 = None

        # Use close as vwap if not provided
        if self.vwap is None:
            self.vwap = self.close.copy()

    @property
    def returns_(self) -> pd.DataFrame:
        """Cached daily returns."""
        if self._returns is None:
            self._returns = returns(self.close)
        return self._returns

    def get_adv(self, d: int) -> pd.DataFrame:
        """Get average daily volume (cached)."""
        cache_name = f'_adv{d}'
        if hasattr(self, cache_name):
            cached = getattr(self, cache_name)
            if cached is not None:
                return cached

        result = adv(self.volume, d)

        if hasattr(self, cache_name):
            setattr(self, cache_name, result)

        return result


class AlphaBase(ABC):
    """Abstract base class for Alpha calculations."""

    def __init__(self, data: MarketData):
        """Initialize with market data.

        Args:
            data: MarketData containing OHLCV and optional fields
        """
        self.data = data

        # Shortcuts for common data
        self.open = data.open_
        self.high = data.high
        self.low = data.low
        self.close = data.close
        self.volume = data.volume
        self.vwap = data.vwap
        self.cap = data.cap
        self.industry = data.industry
        self.returns_ = data.returns_

    def adv(self, d: int) -> pd.DataFrame:
        """Average daily volume over d days."""
        return self.data.get_adv(d)

    @abstractmethod
    def compute(self, alpha_id: int) -> pd.DataFrame:
        """Compute a specific alpha.

        Args:
            alpha_id: Alpha number (1-101)

        Returns:
            DataFrame with alpha values (index=dates, columns=stocks)
        """
        pass

    def compute_all(self, alpha_ids: Optional[list] = None) -> Dict[int, pd.DataFrame]:
        """Compute multiple alphas.

        Args:
            alpha_ids: List of alpha numbers to compute (default: all implemented)

        Returns:
            Dict mapping alpha_id to DataFrame
        """
        if alpha_ids is None:
            alpha_ids = self.get_implemented_alphas()

        results = {}
        for alpha_id in alpha_ids:
            try:
                results[alpha_id] = self.compute(alpha_id)
            except Exception as e:
                print(f"Alpha {alpha_id} failed: {e}")
                results[alpha_id] = None

        return results

    @abstractmethod
    def get_implemented_alphas(self) -> list:
        """Return list of implemented alpha IDs."""
        pass
