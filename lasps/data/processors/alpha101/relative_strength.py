"""Relative Strength Custom Alphas (201-205).

Custom alpha factors measuring stock strength relative to the benchmark index (KOSPI).
These alphas complement WorldQuant Alpha101 by capturing relative performance dynamics.
"""

import numpy as np
import pandas as pd
from typing import List

from .operators import rank, sma, delay, returns, ts_max


class RelativeStrengthAlphas:
    """5 Relative Strength alphas using benchmark index data.

    Alpha 201: Relative Return (20d stock return vs index return)
    Alpha 202: RS Trend (short-term vs long-term relative strength ratio)
    Alpha 203: Down-Market Defense (excess return on index down days)
    Alpha 204: Up-Market Attack (excess return on index up days)
    Alpha 205: Drawdown Recovery Speed (stock drawdown vs index drawdown)
    """

    RS_ALPHA_IDS = [201, 202, 203, 204, 205]

    def __init__(
        self,
        stock_close: pd.DataFrame,
        stock_returns: pd.DataFrame,
        index_close: pd.Series,
    ):
        """Initialize with stock and index data.

        Args:
            stock_close: DataFrame (dates x stocks) of closing prices
            stock_returns: DataFrame (dates x stocks) of daily returns
            index_close: Series (dates -> index close price), e.g. KOSPI
        """
        # Align index to stock dates
        aligned_index = index_close.reindex(stock_close.index)

        # Broadcast index_close to DataFrame shape (dates x stocks)
        self.index_close_df = pd.DataFrame(
            np.outer(aligned_index.values, np.ones(stock_close.shape[1])),
            index=stock_close.index,
            columns=stock_close.columns,
        )
        self.index_returns_df = returns(self.index_close_df)
        self.stock_close = stock_close
        self.stock_returns = stock_returns

    def get_implemented_alphas(self) -> List[int]:
        """Return list of implemented RS alpha IDs."""
        return list(self.RS_ALPHA_IDS)

    def compute(self, alpha_id: int) -> pd.DataFrame:
        """Compute a specific RS alpha.

        Args:
            alpha_id: Alpha number (201-205)

        Returns:
            DataFrame with alpha values (index=dates, columns=stocks)

        Raises:
            NotImplementedError: If alpha_id is not in RS_ALPHA_IDS
        """
        method_name = f'alpha{alpha_id}'
        if hasattr(self, method_name):
            return getattr(self, method_name)()
        raise NotImplementedError(f"RS Alpha {alpha_id} not implemented")

    def alpha201(self) -> pd.DataFrame:
        """Alpha 201: Relative Return (상대수익률).

        rank(stock_return_20d - index_return_20d)

        Stocks outperforming the index over 20 trading days rank higher.
        """
        stock_ret_20d = self.stock_close / delay(self.stock_close, 20) - 1
        index_ret_20d = self.index_close_df / delay(self.index_close_df, 20) - 1
        return rank(stock_ret_20d - index_ret_20d)

    def alpha202(self) -> pd.DataFrame:
        """Alpha 202: RS Trend (RS 추세).

        rank(sma(stock/index, 5) - sma(stock/index, 20))

        Rising short-term RS ratio vs long-term indicates strengthening trend.
        """
        rs_ratio = self.stock_close / self.index_close_df
        return rank(sma(rs_ratio, 5) - sma(rs_ratio, 20))

    def alpha203(self) -> pd.DataFrame:
        """Alpha 203: Down-Market Defense (하락장 방어력).

        rank(mean of (stock_ret - index_ret) on index down days over 60d)

        Stocks that lose less than the index on down days rank higher.
        """
        excess = self.stock_returns - self.index_returns_df
        down_mask = self.index_returns_df < 0
        down_excess = excess.where(down_mask)
        return rank(down_excess.rolling(60, min_periods=20).mean())

    def alpha204(self) -> pd.DataFrame:
        """Alpha 204: Up-Market Attack (상승장 공격력).

        rank(mean of (stock_ret - index_ret) on index up days over 60d)

        Stocks that gain more than the index on up days rank higher.
        """
        excess = self.stock_returns - self.index_returns_df
        up_mask = self.index_returns_df > 0
        up_excess = excess.where(up_mask)
        return rank(up_excess.rolling(60, min_periods=20).mean())

    def alpha205(self) -> pd.DataFrame:
        """Alpha 205: Drawdown Recovery Speed (전고점 회복 속도).

        rank(sma(stock_drawdown_60d - index_drawdown_60d, 20))

        20-day average of relative drawdown vs index.
        Using rolling max with min_periods=20 for broader coverage,
        and averaging over 20 days to break ties among stocks at their 60d high.
        """
        stock_dd = self.stock_close / self.stock_close.rolling(60, min_periods=20).max() - 1
        index_dd = self.index_close_df / self.index_close_df.rolling(60, min_periods=20).max() - 1
        relative_dd = stock_dd - index_dd
        return rank(relative_dd.rolling(20, min_periods=10).mean())
