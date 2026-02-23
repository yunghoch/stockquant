"""Alpha101 Calculator - Main interface for computing all 101 alphas.

Combines SimpleAlphas and IndustryAlphas into a unified interface.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from pathlib import Path
from loguru import logger

from .alpha_base import MarketData
from .simple_alphas import SimpleAlphas
from .industry_alphas import IndustryAlphas


class Alpha101Calculator:
    """Main calculator for WorldQuant's 101 Formulaic Alphas.

    Usage:
        >>> from lasps.data.processors.alpha101 import Alpha101Calculator
        >>> calc = Alpha101Calculator(
        ...     open_=df_open,
        ...     high=df_high,
        ...     low=df_low,
        ...     close=df_close,
        ...     volume=df_volume,
        ... )
        >>> alphas = calc.compute_all()
        >>> alpha1 = calc.compute(1)
    """

    def __init__(
        self,
        open_: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        vwap: Optional[pd.DataFrame] = None,
        cap: Optional[pd.DataFrame] = None,
        industry: Optional[pd.Series] = None,
    ):
        """Initialize with market data.

        Args:
            open_: DataFrame with open prices (index=dates, columns=stocks)
            high: DataFrame with high prices
            low: DataFrame with low prices
            close: DataFrame with close prices
            volume: DataFrame with trading volumes
            vwap: Optional VWAP (VWAP-dependent alphas will fail if not provided)
            cap: Optional market capitalization
            industry: Optional Series mapping stock_code to industry_code

        All DataFrames must have the same shape and aligned indices/columns.
        """
        # Validate inputs
        assert open_.shape == high.shape == low.shape == close.shape == volume.shape, \
            "All price DataFrames must have the same shape"

        self.data = MarketData(
            open_=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            vwap=vwap,
            cap=cap,
            industry=industry,
        )

        # Initialize alpha calculators
        self.simple_alphas = SimpleAlphas(self.data)
        self.industry_alphas = None
        if industry is not None:
            try:
                self.industry_alphas = IndustryAlphas(self.data)
            except ValueError as e:
                logger.warning(f"Industry alphas not available: {e}")

        # Cache computed alphas
        self._cache: Dict[int, pd.DataFrame] = {}

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        date_col: str = 'date',
        stock_col: str = 'stock_code',
        open_col: str = 'open',
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close',
        volume_col: str = 'volume',
        vwap_col: Optional[str] = None,
        cap_col: Optional[str] = None,
        industry_col: Optional[str] = None,
    ) -> 'Alpha101Calculator':
        """Create calculator from a long-format DataFrame.

        Args:
            df: DataFrame with columns for date, stock, and OHLCV
            date_col: Column name for dates
            stock_col: Column name for stock codes
            open_col, high_col, low_col, close_col, volume_col: OHLCV column names
            vwap_col: Optional VWAP column
            cap_col: Optional market cap column
            industry_col: Optional industry classification column

        Returns:
            Alpha101Calculator instance
        """
        # Pivot to wide format
        pivot_cols = [open_col, high_col, low_col, close_col, volume_col]
        if vwap_col:
            pivot_cols.append(vwap_col)
        if cap_col:
            pivot_cols.append(cap_col)

        wide_data = {}
        for col in pivot_cols:
            wide_data[col] = df.pivot(index=date_col, columns=stock_col, values=col)

        # Get industry mapping if available
        industry = None
        if industry_col and industry_col in df.columns:
            industry_df = df[[stock_col, industry_col]].drop_duplicates()
            industry = industry_df.set_index(stock_col)[industry_col]

        return cls(
            open_=wide_data[open_col],
            high=wide_data[high_col],
            low=wide_data[low_col],
            close=wide_data[close_col],
            volume=wide_data[volume_col],
            vwap=wide_data.get(vwap_col),
            cap=wide_data.get(cap_col),
            industry=industry,
        )

    def compute(self, alpha_id: int, use_cache: bool = True) -> pd.DataFrame:
        """Compute a single alpha.

        Args:
            alpha_id: Alpha number (1-101)
            use_cache: Whether to use cached result if available

        Returns:
            DataFrame with alpha values (index=dates, columns=stocks)

        Raises:
            NotImplementedError: If alpha_id is not implemented
        """
        if use_cache and alpha_id in self._cache:
            return self._cache[alpha_id]

        # Determine which calculator to use
        if alpha_id in SimpleAlphas.INDUSTRY_ALPHAS:
            if self.industry_alphas is None:
                raise ValueError(f"Alpha {alpha_id} requires industry data")
            result = self.industry_alphas.compute(alpha_id)
        else:
            result = self.simple_alphas.compute(alpha_id)

        self._cache[alpha_id] = result
        return result

    def compute_batch(
        self,
        alpha_ids: List[int],
        use_cache: bool = True,
        skip_errors: bool = True,
    ) -> Dict[int, pd.DataFrame]:
        """Compute multiple alphas.

        Args:
            alpha_ids: List of alpha numbers to compute
            use_cache: Whether to use cached results
            skip_errors: If True, skip failed alphas; if False, raise exception

        Returns:
            Dict mapping alpha_id to DataFrame
        """
        results = {}

        for alpha_id in alpha_ids:
            try:
                results[alpha_id] = self.compute(alpha_id, use_cache=use_cache)
            except Exception as e:
                if skip_errors:
                    logger.warning(f"Alpha {alpha_id} failed: {e}")
                    results[alpha_id] = None
                else:
                    raise

        return results

    def compute_all(
        self,
        include_industry: bool = True,
        use_cache: bool = True,
        skip_errors: bool = True,
    ) -> Dict[int, pd.DataFrame]:
        """Compute all implemented alphas.

        Args:
            include_industry: Whether to include industry-neutralized alphas
            use_cache: Whether to use cached results
            skip_errors: If True, skip failed alphas

        Returns:
            Dict mapping alpha_id to DataFrame
        """
        # Get list of alphas to compute
        alpha_ids = self.simple_alphas.get_implemented_alphas()

        if include_industry and self.industry_alphas is not None:
            alpha_ids.extend(self.industry_alphas.get_implemented_alphas())

        return self.compute_batch(alpha_ids, use_cache=use_cache, skip_errors=skip_errors)

    def compute_simple(
        self,
        use_cache: bool = True,
        skip_errors: bool = True,
    ) -> Dict[int, pd.DataFrame]:
        """Compute only simple alphas (no industry neutralization).

        Args:
            use_cache: Whether to use cached results
            skip_errors: If True, skip failed alphas

        Returns:
            Dict mapping alpha_id to DataFrame
        """
        alpha_ids = self.simple_alphas.get_implemented_alphas()
        return self.compute_batch(alpha_ids, use_cache=use_cache, skip_errors=skip_errors)

    def to_features(
        self,
        alpha_ids: Optional[List[int]] = None,
        date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Convert alphas to feature matrix.

        Args:
            alpha_ids: List of alpha IDs to include (default: all computed)
            date: Specific date to extract (default: all dates)

        Returns:
            DataFrame with stocks as index and alpha_X as columns
        """
        if alpha_ids is None:
            alpha_ids = list(self._cache.keys())

        if not alpha_ids:
            raise ValueError("No alphas computed. Call compute_all() first.")

        features = {}
        for alpha_id in alpha_ids:
            if alpha_id not in self._cache:
                self.compute(alpha_id)

            alpha_df = self._cache[alpha_id]
            if date is not None:
                features[f'alpha_{alpha_id:03d}'] = alpha_df.loc[date]
            else:
                # Stack all dates
                stacked = alpha_df.stack()
                stacked.name = f'alpha_{alpha_id:03d}'
                features[f'alpha_{alpha_id:03d}'] = stacked

        return pd.DataFrame(features)

    def clear_cache(self):
        """Clear the alpha cache."""
        self._cache.clear()

    def get_computed_alphas(self) -> List[int]:
        """Return list of already computed alpha IDs."""
        return list(self._cache.keys())

    def save_alphas(self, path: Union[str, Path], format: str = 'parquet'):
        """Save computed alphas to disk.

        Args:
            path: Directory to save alphas
            format: Output format ('parquet' or 'csv')
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for alpha_id, alpha_df in self._cache.items():
            if alpha_df is None:
                continue

            filename = f'alpha_{alpha_id:03d}'
            if format == 'parquet':
                alpha_df.to_parquet(path / f'{filename}.parquet')
            elif format == 'csv':
                alpha_df.to_csv(path / f'{filename}.csv')
            else:
                raise ValueError(f"Unknown format: {format}")

        logger.info(f"Saved {len(self._cache)} alphas to {path}")

    @classmethod
    def load_alphas(cls, path: Union[str, Path]) -> Dict[int, pd.DataFrame]:
        """Load previously saved alphas.

        Args:
            path: Directory containing saved alphas

        Returns:
            Dict mapping alpha_id to DataFrame
        """
        path = Path(path)
        alphas = {}

        for file in path.glob('alpha_*.parquet'):
            alpha_id = int(file.stem.split('_')[1])
            alphas[alpha_id] = pd.read_parquet(file)

        for file in path.glob('alpha_*.csv'):
            alpha_id = int(file.stem.split('_')[1])
            if alpha_id not in alphas:
                alphas[alpha_id] = pd.read_csv(file, index_col=0, parse_dates=True)

        logger.info(f"Loaded {len(alphas)} alphas from {path}")
        return alphas
