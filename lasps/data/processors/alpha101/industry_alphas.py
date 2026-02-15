"""Alpha101 Industry Alphas - 19 alphas requiring industry neutralization.

Based on: https://arxiv.org/abs/1601.00991
Section 3: Alphas

Implements alphas that require IndClass (industry classification).
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

from .alpha_base import AlphaBase, MarketData
from .operators import (
    rank, scale, signedpower, log, sign, abs_, max_, min_,
    delta, delay, ts_min, ts_max, ts_argmin, ts_argmax,
    ts_rank, ts_sum, ts_product, stddev, correlation,
    covariance, decay_linear, sma, returns, indneutralize, where
)


class IndustryAlphas(AlphaBase):
    """19 Industry-Neutralized Alphas."""

    # List of alphas implemented in this class
    INDUSTRY_ALPHAS = [48, 56, 58, 59, 63, 67, 69, 70, 76, 79, 80, 82, 87, 89, 90, 91, 93, 97, 100]

    def __init__(self, data: MarketData):
        """Initialize with market data.

        Raises:
            ValueError: If industry data is not provided
        """
        super().__init__(data)
        if self.industry is None:
            raise ValueError("Industry data is required for IndustryAlphas")

    def get_implemented_alphas(self) -> list:
        """Return list of implemented alpha IDs."""
        return self.INDUSTRY_ALPHAS

    def compute(self, alpha_id: int) -> pd.DataFrame:
        """Compute a specific alpha."""
        method_name = f'alpha{alpha_id:03d}'
        if hasattr(self, method_name):
            return getattr(self, method_name)()
        else:
            raise NotImplementedError(f"Alpha {alpha_id} not implemented")

    # =========================================================================
    # Industry Alpha Implementations
    # =========================================================================

    def alpha048(self) -> pd.DataFrame:
        """Alpha #48: indneutralize(-1*ts_max(correlation(rank(vwap), rank(volume), 3), 5),
                      IndClass.subindustry) / count(ts_max(correlation(rank(vwap), rank(volume), 3), 5),
                      IndClass.subindustry)"""
        inner = ts_max(correlation(rank(self.vwap), rank(self.volume), 3), 5)
        neutralized = indneutralize(-1 * inner, self.industry)
        # Note: count is approximated as rank here
        return neutralized

    def alpha056(self) -> pd.DataFrame:
        """Alpha #56: -1 * rank(ts_sum(returns, 10) / ts_sum(ts_sum(returns, 2), 3)) *
                      rank(returns * cap)"""
        part1 = rank(ts_sum(self.returns_, 10) / ts_sum(ts_sum(self.returns_, 2), 3))
        if self.cap is not None:
            part2 = rank(self.returns_ * self.cap)
        else:
            part2 = rank(self.returns_)
        return -1 * part1 * part2

    def alpha058(self) -> pd.DataFrame:
        """Alpha #58: -1 * ts_rank(decay_linear(correlation(indneutralize(vwap, IndClass.sector),
                      volume, 4), 8), 6)"""
        neutralized = indneutralize(self.vwap, self.industry)
        inner = decay_linear(correlation(neutralized, self.volume, 4), 8)
        return -1 * ts_rank(inner, 6)

    def alpha059(self) -> pd.DataFrame:
        """Alpha #59: -1 * ts_rank(decay_linear(correlation(indneutralize((vwap * 0.728317 +
                      vwap * (1 - 0.728317)), IndClass.industry), volume, 4), 16), 8)"""
        inner_vwap = self.vwap * 0.728317 + self.vwap * (1 - 0.728317)  # = vwap
        neutralized = indneutralize(inner_vwap, self.industry)
        inner = decay_linear(correlation(neutralized, self.volume, 4), 16)
        return -1 * ts_rank(inner, 8)

    def alpha063(self) -> pd.DataFrame:
        """Alpha #63: -1 * rank(decay_linear(delta(indneutralize(close, IndClass.industry), 2), 8)) -
                      rank(decay_linear(correlation(vwap * 0.318108 + open * (1-0.318108), ts_sum(adv180, 37), 14), 12))"""
        neutralized = indneutralize(self.close, self.industry)
        part1 = rank(decay_linear(delta(neutralized, 2), 8))
        inner = self.vwap * 0.318108 + self.open * (1 - 0.318108)
        part2 = rank(decay_linear(correlation(inner, ts_sum(self.adv(180), 37), 14), 12))
        return -1 * part1 - part2

    def alpha067(self) -> pd.DataFrame:
        """Alpha #67: rank(rank(high - ts_min(high, 2))) ** rank(correlation(indneutralize(vwap,
                      IndClass.sector), indneutralize(adv20, IndClass.subindustry), 6))"""
        part1 = rank(rank(self.high - ts_min(self.high, 2)))
        neut_vwap = indneutralize(self.vwap, self.industry)
        neut_adv = indneutralize(self.adv(20), self.industry)
        part2 = rank(correlation(neut_vwap, neut_adv, 6))
        return np.power(part1, part2)

    def alpha069(self) -> pd.DataFrame:
        """Alpha #69: rank(rank(ts_max(delta(indneutralize(vwap, IndClass.industry), 3), 5))) **
                      ts_rank(correlation(close*0.490655+vwap*(1-0.490655), adv20, 5), 9)"""
        neutralized = indneutralize(self.vwap, self.industry)
        part1 = rank(rank(ts_max(delta(neutralized, 3), 5)))
        inner = self.close * 0.490655 + self.vwap * (1 - 0.490655)
        part2 = ts_rank(correlation(inner, self.adv(20), 5), 9)
        return np.power(part1, part2)

    def alpha070(self) -> pd.DataFrame:
        """Alpha #70: rank(rank(ts_max(delta(indneutralize(vwap, IndClass.industry), 3), 5))) **
                      ts_rank(correlation(close, adv50, 18), 18)"""
        neutralized = indneutralize(self.vwap, self.industry)
        part1 = rank(rank(ts_max(delta(neutralized, 3), 5)))
        part2 = ts_rank(correlation(self.close, self.adv(50), 18), 18)
        return np.power(part1, part2)

    def alpha076(self) -> pd.DataFrame:
        """Alpha #76: max(rank(decay_linear(delta(vwap, 1), 12)),
                      ts_rank(decay_linear(ts_rank(correlation(indneutralize(low, IndClass.sector),
                      adv81, 8), 20), 17), 19)) * -1"""
        part1 = rank(decay_linear(delta(self.vwap, 1), 12))
        neut_low = indneutralize(self.low, self.industry)
        inner1 = ts_rank(correlation(neut_low, self.adv(81), 8), 20)
        part2 = ts_rank(decay_linear(inner1, 17), 19)
        return max_(part1, part2) * -1

    def alpha079(self) -> pd.DataFrame:
        """Alpha #79: rank(delta(indneutralize((close*0.60733+open*(1-0.60733)), IndClass.sector), 1)) <
                      rank(correlation(ts_rank(vwap, 4), ts_rank(adv150, 9), 15))"""
        inner = self.close * 0.60733 + self.open * (1 - 0.60733)
        neutralized = indneutralize(inner, self.industry)
        x = rank(delta(neutralized, 1))
        y = rank(correlation(ts_rank(self.vwap, 4), ts_rank(self.adv(150), 9), 15))
        return (x < y).astype(float)

    def alpha080(self) -> pd.DataFrame:
        """Alpha #80: rank(sign(delta(indneutralize((open*0.868128+high*(1-0.868128)), IndClass.industry), 4))) **
                      ts_rank(correlation(high, adv10, 5), 6)"""
        inner = self.open * 0.868128 + self.high * (1 - 0.868128)
        neutralized = indneutralize(inner, self.industry)
        part1 = rank(sign(delta(neutralized, 4)))
        part2 = ts_rank(correlation(self.high, self.adv(10), 5), 6)
        return np.power(part1, part2)

    def alpha082(self) -> pd.DataFrame:
        """Alpha #82: min(rank(decay_linear(delta(open, 1), 15)),
                      ts_rank(decay_linear(correlation(indneutralize(volume, IndClass.sector),
                      open*0.634196+open*(1-0.634196), 17), 7), 13)) * -1"""
        part1 = rank(decay_linear(delta(self.open, 1), 15))
        neut_vol = indneutralize(self.volume, self.industry)
        inner = self.open * 0.634196 + self.open * (1 - 0.634196)  # = open
        part2 = ts_rank(decay_linear(correlation(neut_vol, inner, 17), 7), 13)
        return min_(part1, part2) * -1

    def alpha087(self) -> pd.DataFrame:
        """Alpha #87: max(rank(decay_linear(delta((close*0.369701+vwap*(1-0.369701)), 2), 3)),
                      ts_rank(decay_linear(abs(correlation(indneutralize(adv81, IndClass.industry),
                      close, 13)), 5), 14)) * -1"""
        inner1 = self.close * 0.369701 + self.vwap * (1 - 0.369701)
        part1 = rank(decay_linear(delta(inner1, 2), 3))
        neut_adv = indneutralize(self.adv(81), self.industry)
        inner2 = abs_(correlation(neut_adv, self.close, 13))
        part2 = ts_rank(decay_linear(inner2, 5), 14)
        return max_(part1, part2) * -1

    def alpha089(self) -> pd.DataFrame:
        """Alpha #89: ts_rank(decay_linear(correlation(low*0.967285+low*(1-0.967285),
                      adv10, 7), 6), 4) - ts_rank(decay_linear(delta(indneutralize(vwap,
                      IndClass.industry), 3), 10), 15)"""
        inner1 = self.low * 0.967285 + self.low * (1 - 0.967285)  # = low
        part1 = ts_rank(decay_linear(correlation(inner1, self.adv(10), 7), 6), 4)
        neut_vwap = indneutralize(self.vwap, self.industry)
        part2 = ts_rank(decay_linear(delta(neut_vwap, 3), 10), 15)
        return part1 - part2

    def alpha090(self) -> pd.DataFrame:
        """Alpha #90: rank(rank(correlation(close, volume, 10))) ** rank(rank(ts_max(close, 5))) *
                      indneutralize(-1 * rank(delta(close, 1)), IndClass.subindustry)"""
        part1 = rank(rank(correlation(self.close, self.volume, 10)))
        part2 = rank(rank(ts_max(self.close, 5)))
        inner = -1 * rank(delta(self.close, 1))
        part3 = indneutralize(inner, self.industry)
        return np.power(part1, part2) * part3

    def alpha091(self) -> pd.DataFrame:
        """Alpha #91: ts_rank(decay_linear(decay_linear(correlation(indneutralize(close, IndClass.industry),
                      volume, 10), 4), 5), 3) - ts_rank(decay_linear(correlation(vwap, adv30, 4), 3), 16)"""
        neut_close = indneutralize(self.close, self.industry)
        inner1 = decay_linear(correlation(neut_close, self.volume, 10), 4)
        part1 = ts_rank(decay_linear(inner1, 5), 3)
        part2 = ts_rank(decay_linear(correlation(self.vwap, self.adv(30), 4), 3), 16)
        return part1 - part2

    def alpha093(self) -> pd.DataFrame:
        """Alpha #93: ts_rank(decay_linear(correlation(indneutralize(vwap, IndClass.industry),
                      adv81, 17), 20), 8) / rank(decay_linear(delta(close*0.524434+vwap*(1-0.524434), 3), 16))"""
        neut_vwap = indneutralize(self.vwap, self.industry)
        numer = ts_rank(decay_linear(correlation(neut_vwap, self.adv(81), 17), 20), 8)
        inner = self.close * 0.524434 + self.vwap * (1 - 0.524434)
        denom = rank(decay_linear(delta(inner, 3), 16))
        return numer / denom

    def alpha097(self) -> pd.DataFrame:
        """Alpha #97: rank(decay_linear(delta(indneutralize((low*0.721001+vwap*(1-0.721001)),
                      IndClass.industry), 3), 20)) - ts_rank(decay_linear(ts_rank(correlation(ts_rank(low, 8),
                      ts_rank(adv60, 17), 5), 19), 16), 7)"""
        inner = self.low * 0.721001 + self.vwap * (1 - 0.721001)
        neut = indneutralize(inner, self.industry)
        part1 = rank(decay_linear(delta(neut, 3), 20))
        inner2 = correlation(ts_rank(self.low, 8), ts_rank(self.adv(60), 17), 5)
        part2 = ts_rank(decay_linear(ts_rank(inner2, 19), 16), 7)
        return part1 - part2

    def alpha100(self) -> pd.DataFrame:
        """Alpha #100: indneutralize(-1 * rank(stddev(returns, 5) - correlation(returns, volume, 5)) *
                       rank(correlation(returns, adv20, 5)), IndClass.subindustry)"""
        part1 = rank(stddev(self.returns_, 5) - correlation(self.returns_, self.volume, 5))
        part2 = rank(correlation(self.returns_, self.adv(20), 5))
        inner = -1 * part1 * part2
        return indneutralize(inner, self.industry)
