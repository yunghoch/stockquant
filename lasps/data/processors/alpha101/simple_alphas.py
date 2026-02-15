"""Alpha101 Simple Alphas - 82 alphas without industry neutralization.

Based on: https://arxiv.org/abs/1601.00991
Section 3: Alphas

Implements alphas that don't require IndClass (industry classification).
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

from .alpha_base import AlphaBase, MarketData
from .operators import (
    rank, scale, signedpower, log, sign, abs_, max_, min_,
    delta, delay, ts_min, ts_max, ts_argmin, ts_argmax,
    ts_rank, ts_sum, ts_product, stddev, correlation,
    covariance, decay_linear, sma, returns, where
)


class SimpleAlphas(AlphaBase):
    """82 Simple Alphas without industry neutralization."""

    # List of alphas that require industry data (excluded from this class)
    INDUSTRY_ALPHAS = [48, 56, 58, 59, 63, 67, 69, 70, 76, 79, 80, 82, 87, 89, 90, 91, 93, 97, 100]

    def get_implemented_alphas(self) -> list:
        """Return list of implemented alpha IDs (1-101 excluding industry alphas)."""
        return [i for i in range(1, 102) if i not in self.INDUSTRY_ALPHAS]

    def compute(self, alpha_id: int) -> pd.DataFrame:
        """Compute a specific alpha."""
        method_name = f'alpha{alpha_id:03d}'
        if hasattr(self, method_name):
            return getattr(self, method_name)()
        else:
            raise NotImplementedError(f"Alpha {alpha_id} not implemented")

    # =========================================================================
    # Alpha Implementations (1-101, excluding industry alphas)
    # =========================================================================

    def alpha001(self) -> pd.DataFrame:
        """Alpha #1: rank(ts_argmax(signedpower(((returns < 0) ? stddev(returns, 20) : close), 2), 5)) - 0.5"""
        cond = self.returns_ < 0
        inner = where(cond, stddev(self.returns_, 20), self.close)
        inner = signedpower(inner, 2)
        return rank(ts_argmax(inner, 5)) - 0.5

    def alpha002(self) -> pd.DataFrame:
        """Alpha #2: -1 * correlation(rank(delta(log(volume), 2)), rank((close - open) / open), 6)"""
        x = rank(delta(log(self.volume), 2))
        y = rank((self.close - self.open) / self.open)
        return -1 * correlation(x, y, 6)

    def alpha003(self) -> pd.DataFrame:
        """Alpha #3: -1 * correlation(rank(open), rank(volume), 10)"""
        return -1 * correlation(rank(self.open), rank(self.volume), 10)

    def alpha004(self) -> pd.DataFrame:
        """Alpha #4: -1 * ts_rank(rank(low), 9)"""
        return -1 * ts_rank(rank(self.low), 9)

    def alpha005(self) -> pd.DataFrame:
        """Alpha #5: rank(open - ts_sum(vwap, 10) / 10) * (-1 * abs(rank(close - vwap)))"""
        x = rank(self.open - ts_sum(self.vwap, 10) / 10)
        y = -1 * abs_(rank(self.close - self.vwap))
        return x * y

    def alpha006(self) -> pd.DataFrame:
        """Alpha #6: -1 * correlation(open, volume, 10)"""
        return -1 * correlation(self.open, self.volume, 10)

    def alpha007(self) -> pd.DataFrame:
        """Alpha #7: (adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : -1"""
        cond = self.adv(20) < self.volume
        x = -1 * ts_rank(abs_(delta(self.close, 7)), 60) * sign(delta(self.close, 7))
        return where(cond, x, pd.DataFrame(-1, index=self.close.index, columns=self.close.columns))

    def alpha008(self) -> pd.DataFrame:
        """Alpha #8: -1 * rank(ts_sum(open, 5) * ts_sum(returns, 5) - delay(ts_sum(open, 5) * ts_sum(returns, 5), 10))"""
        inner = ts_sum(self.open, 5) * ts_sum(self.returns_, 5)
        return -1 * rank(inner - delay(inner, 10))

    def alpha009(self) -> pd.DataFrame:
        """Alpha #9: (0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) :
                     ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : -1 * delta(close, 1))"""
        d = delta(self.close, 1)
        cond1 = ts_min(d, 5) > 0
        cond2 = ts_max(d, 5) < 0
        return where(cond1, d, where(cond2, d, -1 * d))

    def alpha010(self) -> pd.DataFrame:
        """Alpha #10: Same as alpha009 but with rank."""
        d = delta(self.close, 1)
        cond1 = ts_min(d, 4) > 0
        cond2 = ts_max(d, 4) < 0
        inner = where(cond1, d, where(cond2, d, -1 * d))
        return rank(inner)

    def alpha011(self) -> pd.DataFrame:
        """Alpha #11: (rank(ts_max(vwap - close, 3)) + rank(ts_min(vwap - close, 3))) * rank(delta(volume, 3))"""
        x = rank(ts_max(self.vwap - self.close, 3))
        y = rank(ts_min(self.vwap - self.close, 3))
        z = rank(delta(self.volume, 3))
        return (x + y) * z

    def alpha012(self) -> pd.DataFrame:
        """Alpha #12: sign(delta(volume, 1)) * (-1 * delta(close, 1))"""
        return sign(delta(self.volume, 1)) * (-1 * delta(self.close, 1))

    def alpha013(self) -> pd.DataFrame:
        """Alpha #13: -1 * rank(covariance(rank(close), rank(volume), 5))"""
        return -1 * rank(covariance(rank(self.close), rank(self.volume), 5))

    def alpha014(self) -> pd.DataFrame:
        """Alpha #14: -1 * rank(delta(returns, 3)) * correlation(open, volume, 10)"""
        return -1 * rank(delta(self.returns_, 3)) * correlation(self.open, self.volume, 10)

    def alpha015(self) -> pd.DataFrame:
        """Alpha #15: -1 * ts_sum(rank(correlation(rank(high), rank(volume), 3)), 3)"""
        inner = rank(correlation(rank(self.high), rank(self.volume), 3))
        return -1 * ts_sum(inner, 3)

    def alpha016(self) -> pd.DataFrame:
        """Alpha #16: -1 * rank(covariance(rank(high), rank(volume), 5))"""
        return -1 * rank(covariance(rank(self.high), rank(self.volume), 5))

    def alpha017(self) -> pd.DataFrame:
        """Alpha #17: -1 * rank(ts_rank(close, 10)) * rank(delta(delta(close, 1), 1)) * rank(ts_rank(volume / adv20, 5))"""
        x = rank(ts_rank(self.close, 10))
        y = rank(delta(delta(self.close, 1), 1))
        z = rank(ts_rank(self.volume / self.adv(20), 5))
        return -1 * x * y * z

    def alpha018(self) -> pd.DataFrame:
        """Alpha #18: -1 * rank(stddev(abs(close - open), 5) + (close - open) + correlation(close, open, 10))"""
        inner = stddev(abs_(self.close - self.open), 5) + (self.close - self.open) + correlation(self.close, self.open, 10)
        return -1 * rank(inner)

    def alpha019(self) -> pd.DataFrame:
        """Alpha #19: -1 * sign(close - delay(close, 7) + delta(close, 7)) * (1 + rank(1 + ts_sum(returns, 250)))"""
        x = sign(self.close - delay(self.close, 7) + delta(self.close, 7))
        y = 1 + rank(1 + ts_sum(self.returns_, 250))
        return -1 * x * y

    def alpha020(self) -> pd.DataFrame:
        """Alpha #20: -1 * rank(open - delay(high, 1)) * rank(open - delay(close, 1)) * rank(open - delay(low, 1))"""
        x = rank(self.open - delay(self.high, 1))
        y = rank(self.open - delay(self.close, 1))
        z = rank(self.open - delay(self.low, 1))
        return -1 * x * y * z

    def alpha021(self) -> pd.DataFrame:
        """Alpha #21: Complex conditional with sma and volume comparison."""
        cond1 = sma(self.close, 8) + stddev(self.close, 8) < sma(self.close, 2)
        cond2 = sma(self.volume, 20) / self.volume < 1
        inner = where(cond1, -1, where(cond2, 1, -1))
        return inner.astype(float)

    def alpha022(self) -> pd.DataFrame:
        """Alpha #22: -1 * delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))"""
        return -1 * delta(correlation(self.high, self.volume, 5), 5) * rank(stddev(self.close, 20))

    def alpha023(self) -> pd.DataFrame:
        """Alpha #23: (sma(high, 20) < high) ? -1 * delta(high, 2) : 0"""
        cond = sma(self.high, 20) < self.high
        return where(cond, -1 * delta(self.high, 2), pd.DataFrame(0, index=self.close.index, columns=self.close.columns))

    def alpha024(self) -> pd.DataFrame:
        """Alpha #24: (delta(sma(close, 100), 100) / delay(close, 100) <= 0.05) ?
                      -1 * (close - ts_min(close, 100)) : -1 * delta(close, 3)"""
        cond = delta(sma(self.close, 100), 100) / delay(self.close, 100) <= 0.05
        x = -1 * (self.close - ts_min(self.close, 100))
        y = -1 * delta(self.close, 3)
        return where(cond, x, y)

    def alpha025(self) -> pd.DataFrame:
        """Alpha #25: rank(-1 * returns * adv20 * vwap * (high - close))"""
        return rank(-1 * self.returns_ * self.adv(20) * self.vwap * (self.high - self.close))

    def alpha026(self) -> pd.DataFrame:
        """Alpha #26: -1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)"""
        inner = correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5)
        return -1 * ts_max(inner, 3)

    def alpha027(self) -> pd.DataFrame:
        """Alpha #27: (0.5 < rank(sma(correlation(rank(volume), rank(vwap), 6), 2))) ? -1 : 1"""
        inner = sma(correlation(rank(self.volume), rank(self.vwap), 6), 2)
        cond = rank(inner) > 0.5
        return where(cond, -1, pd.DataFrame(1, index=self.close.index, columns=self.close.columns)).astype(float)

    def alpha028(self) -> pd.DataFrame:
        """Alpha #28: scale(correlation(adv20, low, 5) + (high + low) / 2 - close)"""
        inner = correlation(self.adv(20), self.low, 5) + (self.high + self.low) / 2 - self.close
        return scale(inner)

    def alpha029(self) -> pd.DataFrame:
        """Alpha #29: min(product(rank(rank(scale(log(ts_sum(ts_min(rank(rank(-1*rank(delta(close-1,5)))),2),1))))),1),5)
                      + ts_rank(delay(-1*returns,6),5)"""
        inner1 = -1 * rank(delta(self.close - 1, 5))
        inner2 = ts_min(rank(rank(inner1)), 2)
        inner3 = log(ts_sum(inner2, 1))
        inner4 = rank(rank(scale(inner3)))
        inner5 = ts_product(inner4, 1)
        part1 = min_(inner5, 5)
        part2 = ts_rank(delay(-1 * self.returns_, 6), 5)
        return part1 + part2

    def alpha030(self) -> pd.DataFrame:
        """Alpha #30: (1 - rank(sign(close - delay(close, 1)) + sign(delay(close, 1) - delay(close, 2)) +
                      sign(delay(close, 2) - delay(close, 3)))) * ts_sum(volume, 5) / ts_sum(volume, 20)"""
        s1 = sign(self.close - delay(self.close, 1))
        s2 = sign(delay(self.close, 1) - delay(self.close, 2))
        s3 = sign(delay(self.close, 2) - delay(self.close, 3))
        part1 = 1 - rank(s1 + s2 + s3)
        part2 = ts_sum(self.volume, 5) / ts_sum(self.volume, 20)
        return part1 * part2

    def alpha031(self) -> pd.DataFrame:
        """Alpha #31: rank(rank(rank(decay_linear(-1*rank(rank(delta(close,10))),10))))
                      + rank(-1*delta(close,3)) + sign(scale(correlation(adv20,low,12)))"""
        part1 = rank(rank(rank(decay_linear(-1 * rank(rank(delta(self.close, 10))), 10))))
        part2 = rank(-1 * delta(self.close, 3))
        part3 = sign(scale(correlation(self.adv(20), self.low, 12)))
        return part1 + part2 + part3

    def alpha032(self) -> pd.DataFrame:
        """Alpha #32: scale(sma(close,7) - close) + 20*scale(correlation(vwap, delay(close,5), 230))"""
        part1 = scale(sma(self.close, 7) - self.close)
        part2 = 20 * scale(correlation(self.vwap, delay(self.close, 5), 230))
        return part1 + part2

    def alpha033(self) -> pd.DataFrame:
        """Alpha #33: rank(-1 * (1 - open / close))"""
        return rank(-1 * (1 - self.open / self.close))

    def alpha034(self) -> pd.DataFrame:
        """Alpha #34: rank(1 - rank(stddev(returns, 2) / stddev(returns, 5)) + 1 - rank(delta(close, 1)))"""
        part1 = rank(stddev(self.returns_, 2) / stddev(self.returns_, 5))
        part2 = rank(delta(self.close, 1))
        return rank(1 - part1 + 1 - part2)

    def alpha035(self) -> pd.DataFrame:
        """Alpha #35: ts_rank(volume, 32) * (1 - ts_rank(close + high - low, 16)) * (1 - ts_rank(returns, 32))"""
        x = ts_rank(self.volume, 32)
        y = 1 - ts_rank(self.close + self.high - self.low, 16)
        z = 1 - ts_rank(self.returns_, 32)
        return x * y * z

    def alpha036(self) -> pd.DataFrame:
        """Alpha #36: Complex multi-part formula."""
        part1 = 2.21 * rank(correlation(self.close - self.open, delay(self.volume, 1), 15))
        part2 = 0.7 * rank(self.open - self.close)
        part3 = 0.73 * rank(ts_rank(delay(-1 * self.returns_, 6), 5))
        part4 = rank(abs_(correlation(self.vwap, self.adv(20), 6)))
        part5 = 0.6 * rank(sma(self.close, 200) - self.open) * (self.close - self.open)
        return part1 + part2 + part3 + part4 + part5

    def alpha037(self) -> pd.DataFrame:
        """Alpha #37: rank(correlation(delay(open - close, 1), close, 200)) + rank(open - close)"""
        part1 = rank(correlation(delay(self.open - self.close, 1), self.close, 200))
        part2 = rank(self.open - self.close)
        return part1 + part2

    def alpha038(self) -> pd.DataFrame:
        """Alpha #38: -1 * rank(ts_rank(close, 10)) * rank(close / open)"""
        return -1 * rank(ts_rank(self.close, 10)) * rank(self.close / self.open)

    def alpha039(self) -> pd.DataFrame:
        """Alpha #39: -1 * rank(delta(close, 7) * (1 - rank(decay_linear(volume / adv20, 9)))) *
                      (1 + rank(ts_sum(returns, 250)))"""
        part1 = delta(self.close, 7) * (1 - rank(decay_linear(self.volume / self.adv(20), 9)))
        part2 = 1 + rank(ts_sum(self.returns_, 250))
        return -1 * rank(part1) * part2

    def alpha040(self) -> pd.DataFrame:
        """Alpha #40: -1 * rank(stddev(high, 10)) * correlation(high, volume, 10)"""
        return -1 * rank(stddev(self.high, 10)) * correlation(self.high, self.volume, 10)

    def alpha041(self) -> pd.DataFrame:
        """Alpha #41: power(high * low, 0.5) - vwap"""
        return np.power(self.high * self.low, 0.5) - self.vwap

    def alpha042(self) -> pd.DataFrame:
        """Alpha #42: rank(vwap - close) / rank(vwap + close)"""
        return rank(self.vwap - self.close) / rank(self.vwap + self.close)

    def alpha043(self) -> pd.DataFrame:
        """Alpha #43: ts_rank(volume / adv20, 20) * ts_rank(-1 * delta(close, 7), 8)"""
        return ts_rank(self.volume / self.adv(20), 20) * ts_rank(-1 * delta(self.close, 7), 8)

    def alpha044(self) -> pd.DataFrame:
        """Alpha #44: -1 * correlation(high, rank(volume), 5)"""
        return -1 * correlation(self.high, rank(self.volume), 5)

    def alpha045(self) -> pd.DataFrame:
        """Alpha #45: -1 * rank(sma(delay(close, 5), 20)) * correlation(close, volume, 2) *
                      rank(correlation(ts_sum(close, 5), ts_sum(close, 20), 2))"""
        part1 = rank(sma(delay(self.close, 5), 20))
        part2 = correlation(self.close, self.volume, 2)
        part3 = rank(correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2))
        return -1 * part1 * part2 * part3

    def alpha046(self) -> pd.DataFrame:
        """Alpha #46: Conditional with delta comparisons."""
        d20 = delta(self.close, 20)
        d10 = delta(self.close, 10)

        cond1 = 0.25 < d20 / delay(self.close, 20)
        cond2 = d20 / delay(self.close, 20) < 0

        inner = where(cond1, -1, where(cond2, 1, -1 * d10))
        return inner.astype(float)

    def alpha047(self) -> pd.DataFrame:
        """Alpha #47: rank(1/close) * volume / adv20 * high * rank(high - close) /
                      sma(high, 5) - rank(vwap - delay(vwap, 5))"""
        part1 = rank(1 / self.close) * self.volume / self.adv(20)
        part2 = self.high * rank(self.high - self.close) / sma(self.high, 5)
        part3 = rank(self.vwap - delay(self.vwap, 5))
        return part1 * part2 - part3

    # Alpha 48 requires industry - skipped

    def alpha049(self) -> pd.DataFrame:
        """Alpha #49: Conditional with close/delay comparison."""
        cond = delta(delay(self.close, 10), 10) / 10 - delta(self.close, 10) / 10 < -0.1 * self.close
        x = 1
        y = -1 * (self.close - ts_min(self.close, 12))
        return where(cond, x, y).astype(float)

    def alpha050(self) -> pd.DataFrame:
        """Alpha #50: -1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5)"""
        inner = correlation(rank(self.volume), rank(self.vwap), 5)
        return -1 * ts_max(rank(inner), 5)

    def alpha051(self) -> pd.DataFrame:
        """Alpha #51: Conditional similar to alpha049."""
        cond = delta(delay(self.close, 10), 10) / 10 - delta(self.close, 10) / 10 < -0.05 * self.close
        x = 1
        y = -1 * (self.close - ts_min(self.close, 12))
        return where(cond, x, y).astype(float)

    def alpha052(self) -> pd.DataFrame:
        """Alpha #52: -1 * delta(ts_min(low, 5), 5) * rank((ts_sum(returns, 240) - ts_sum(returns, 20)) / 220) *
                      ts_rank(volume, 5)"""
        part1 = delta(ts_min(self.low, 5), 5)
        part2 = rank((ts_sum(self.returns_, 240) - ts_sum(self.returns_, 20)) / 220)
        part3 = ts_rank(self.volume, 5)
        return -1 * part1 * part2 * part3

    def alpha053(self) -> pd.DataFrame:
        """Alpha #53: -1 * delta((high - low) / (delay(close, 1)), 9)"""
        return -1 * delta((self.high - self.low) / delay(self.close, 1), 9)

    def alpha054(self) -> pd.DataFrame:
        """Alpha #54: -1 * (low - close) * power(open, 5) / ((low - high) * power(close, 5))"""
        numer = (self.low - self.close) * np.power(self.open, 5)
        denom = (self.low - self.high) * np.power(self.close, 5)
        return -1 * numer / denom

    def alpha055(self) -> pd.DataFrame:
        """Alpha #55: -1 * correlation(rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))),
                      rank(volume), 6)"""
        numer = self.close - ts_min(self.low, 12)
        denom = ts_max(self.high, 12) - ts_min(self.low, 12)
        x = rank(numer / denom)
        y = rank(self.volume)
        return -1 * correlation(x, y, 6)

    # Alpha 56 requires industry - skipped

    def alpha057(self) -> pd.DataFrame:
        """Alpha #57: -1 * (close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2)"""
        numer = self.close - self.vwap
        denom = decay_linear(rank(ts_argmax(self.close, 30)), 2)
        return -1 * numer / denom

    # Alpha 58, 59 require industry - skipped

    def alpha060(self) -> pd.DataFrame:
        """Alpha #60: -1 * rank(2*scale(rank((close-low)/(high-low)-0.5))-scale(rank(ts_argmax(close,10))))"""
        inner1 = (self.close - self.low) / (self.high - self.low) - 0.5
        part1 = 2 * scale(rank(inner1))
        part2 = scale(rank(ts_argmax(self.close, 10)))
        return -1 * rank(part1 - part2)

    def alpha061(self) -> pd.DataFrame:
        """Alpha #61: rank(vwap - ts_min(vwap, 16)) < rank(correlation(vwap, adv180, 18))"""
        x = rank(self.vwap - ts_min(self.vwap, 16))
        y = rank(correlation(self.vwap, self.adv(180), 18))
        return (x < y).astype(float)

    def alpha062(self) -> pd.DataFrame:
        """Alpha #62: rank(correlation(vwap, ts_sum(adv20, 22), 10)) < rank((rank(open) + rank(open)) <
                      (rank(high - low) + rank(close)))"""
        x = rank(correlation(self.vwap, ts_sum(self.adv(20), 22), 10))
        left = rank(self.open) + rank(self.open)
        right = rank(self.high - self.low) + rank(self.close)
        y = rank((left < right).astype(float))
        return (x < y).astype(float)

    # Alpha 63 requires industry - skipped

    def alpha064(self) -> pd.DataFrame:
        """Alpha #64: rank(correlation(ts_sum((open*0.178404+(low*0.178404)),13),
                      ts_sum(adv120,13),17)) < rank(delta((high+low)/2*0.178404+vwap*0.178404,4)))"""
        part1 = ts_sum(self.open * 0.178404 + self.low * 0.178404, 13)
        part2 = ts_sum(self.adv(120), 13)
        x = rank(correlation(part1, part2, 17))
        inner = (self.high + self.low) / 2 * 0.178404 + self.vwap * 0.178404
        y = rank(delta(inner, 4))
        return (x < y).astype(float)

    def alpha065(self) -> pd.DataFrame:
        """Alpha #65: rank(correlation(open*0.00817205+vwap*0.00817205,
                      ts_sum(adv60,9),6)) < rank(open - ts_min(open,14))"""
        x = rank(correlation(self.open * 0.00817205 + self.vwap * 0.00817205, ts_sum(self.adv(60), 9), 6))
        y = rank(self.open - ts_min(self.open, 14))
        return (x < y).astype(float)

    def alpha066(self) -> pd.DataFrame:
        """Alpha #66: -1 * rank(decay_linear(delta(vwap, 4), 7)) + ts_rank(decay_linear((low*0.96633 +
                      low*0.96633 - vwap) / (open - (high+low)/2), 11), 7)"""
        part1 = rank(decay_linear(delta(self.vwap, 4), 7))
        inner = (self.low * 0.96633 + self.low * 0.96633 - self.vwap) / (self.open - (self.high + self.low) / 2)
        part2 = ts_rank(decay_linear(inner, 11), 7)
        return -1 * part1 + part2

    # Alpha 67 requires industry - skipped

    def alpha068(self) -> pd.DataFrame:
        """Alpha #68: ts_rank(correlation(rank(high), rank(adv15), 9), 14) < rank(delta(close*0.518371 +
                      low*0.518371, 1))"""
        x = ts_rank(correlation(rank(self.high), rank(self.adv(15)), 9), 14)
        y = rank(delta(self.close * 0.518371 + self.low * 0.518371, 1))
        return (x < y).astype(float)

    # Alpha 69, 70 require industry - skipped

    def alpha071(self) -> pd.DataFrame:
        """Alpha #71: max(ts_rank(decay_linear(correlation(ts_rank(close, 3), ts_rank(adv180, 12), 18), 4), 16),
                      ts_rank(decay_linear(rank(low + open - 2*vwap)^2, 16), 4))"""
        inner1 = correlation(ts_rank(self.close, 3), ts_rank(self.adv(180), 12), 18)
        part1 = ts_rank(decay_linear(inner1, 4), 16)
        inner2 = np.power(rank(self.low + self.open - 2 * self.vwap), 2)
        part2 = ts_rank(decay_linear(inner2, 16), 4)
        return max_(part1, part2)

    def alpha072(self) -> pd.DataFrame:
        """Alpha #72: rank(decay_linear(correlation(high-low/2, adv40, 9), 10)) /
                      rank(decay_linear(correlation(ts_rank(vwap, 4), ts_rank(volume, 19), 7), 3))"""
        numer = rank(decay_linear(correlation((self.high - self.low) / 2, self.adv(40), 9), 10))
        denom = rank(decay_linear(correlation(ts_rank(self.vwap, 4), ts_rank(self.volume, 19), 7), 3))
        return numer / denom

    def alpha073(self) -> pd.DataFrame:
        """Alpha #73: max(rank(decay_linear(delta(vwap, 5), 3)),
                      ts_rank(decay_linear((-1*delta(open*0.147155+low*0.147155, 2) /
                      (open*0.147155+low*0.147155)) * -1, 3), 17)) * -1"""
        part1 = rank(decay_linear(delta(self.vwap, 5), 3))
        inner = -1 * delta(self.open * 0.147155 + self.low * 0.147155, 2) / (self.open * 0.147155 + self.low * 0.147155)
        part2 = ts_rank(decay_linear(-1 * inner, 3), 17)
        return max_(part1, part2) * -1

    def alpha074(self) -> pd.DataFrame:
        """Alpha #74: rank(correlation(close, ts_sum(adv30, 37), 15)) <
                      rank(correlation(rank(high*0.0261661 + vwap*0.0261661), rank(volume), 11))"""
        x = rank(correlation(self.close, ts_sum(self.adv(30), 37), 15))
        inner = self.high * 0.0261661 + self.vwap * 0.0261661
        y = rank(correlation(rank(inner), rank(self.volume), 11))
        return (x < y).astype(float)

    def alpha075(self) -> pd.DataFrame:
        """Alpha #75: rank(correlation(vwap, volume, 4)) < rank(correlation(rank(low), rank(adv50), 12))"""
        x = rank(correlation(self.vwap, self.volume, 4))
        y = rank(correlation(rank(self.low), rank(self.adv(50)), 12))
        return (x < y).astype(float)

    # Alpha 76 requires industry - skipped

    def alpha077(self) -> pd.DataFrame:
        """Alpha #77: min(rank(decay_linear((high+low)/2 + high - close - open, 20)),
                      rank(decay_linear(correlation(high+low/2, adv40, 3), 6)))"""
        part1 = rank(decay_linear((self.high + self.low) / 2 + self.high - self.close - self.open, 20))
        part2 = rank(decay_linear(correlation((self.high + self.low) / 2, self.adv(40), 3), 6))
        return min_(part1, part2)

    def alpha078(self) -> pd.DataFrame:
        """Alpha #78: rank(correlation(ts_sum(low*0.352233 + vwap*0.352233, 20),
                      ts_sum(adv40, 20), 7)) ** rank(correlation(rank(vwap), rank(volume), 6))"""
        inner1 = ts_sum(self.low * 0.352233 + self.vwap * 0.352233, 20)
        inner2 = ts_sum(self.adv(40), 20)
        part1 = rank(correlation(inner1, inner2, 7))
        part2 = rank(correlation(rank(self.vwap), rank(self.volume), 6))
        return np.power(part1, part2)

    # Alpha 79, 80 require industry - skipped

    def alpha081(self) -> pd.DataFrame:
        """Alpha #81: rank(log(ts_product(rank(rank(correlation(vwap, ts_sum(adv10, 50), 8))^4), 15))) <
                      rank(correlation(rank(vwap), rank(volume), 5))"""
        inner = correlation(self.vwap, ts_sum(self.adv(10), 50), 8)
        part1 = rank(log(ts_product(np.power(rank(rank(inner)), 4), 15)))
        part2 = rank(correlation(rank(self.vwap), rank(self.volume), 5))
        return (part1 < part2).astype(float)

    # Alpha 82 requires industry - skipped

    def alpha083(self) -> pd.DataFrame:
        """Alpha #83: rank(delay((high - low) / sma(close, 14) * 100, 4)) / 100 * rank(delay(close, 4)) /
                      ts_sum(close, 5) * rank(close) * 5"""
        inner = (self.high - self.low) / sma(self.close, 14) * 100
        part1 = rank(delay(inner, 4)) / 100
        part2 = rank(delay(self.close, 4)) / ts_sum(self.close, 5)
        part3 = rank(self.close) * 5
        return part1 * part2 * part3

    def alpha084(self) -> pd.DataFrame:
        """Alpha #84: signedpower(ts_rank(vwap - ts_max(vwap, 15), 21), delta(close, 5))"""
        inner = ts_rank(self.vwap - ts_max(self.vwap, 15), 21)
        return signedpower(inner, delta(self.close, 5))

    def alpha085(self) -> pd.DataFrame:
        """Alpha #85: rank(correlation(high*0.876703 + close*0.876703 - low*0.876703, adv30, 10)) **
                      rank(correlation(ts_rank(high+low/2, 4), ts_rank(volume, 10), 7))"""
        inner1 = self.high * 0.876703 + self.close * 0.876703 - self.low * 0.876703
        part1 = rank(correlation(inner1, self.adv(30), 10))
        inner2 = ts_rank((self.high + self.low) / 2, 4)
        inner3 = ts_rank(self.volume, 10)
        part2 = rank(correlation(inner2, inner3, 7))
        return np.power(part1, part2)

    def alpha086(self) -> pd.DataFrame:
        """Alpha #86: ts_rank(correlation(close, ts_sum(adv20, 15), 6), 20) <
                      rank(close + open - 2*vwap)"""
        x = ts_rank(correlation(self.close, ts_sum(self.adv(20), 15), 6), 20)
        y = rank(self.close + self.open - 2 * self.vwap)
        return (x < y).astype(float)

    # Alpha 87 requires industry - skipped

    def alpha088(self) -> pd.DataFrame:
        """Alpha #88: min(rank(decay_linear(open+low-high-close, 8)),
                      ts_rank(decay_linear(correlation(ts_rank(close, 8), ts_rank(adv60, 21), 8), 7), 3))"""
        part1 = rank(decay_linear(self.open + self.low - self.high - self.close, 8))
        inner = correlation(ts_rank(self.close, 8), ts_rank(self.adv(60), 21), 8)
        part2 = ts_rank(decay_linear(inner, 7), 3)
        return min_(part1, part2)

    # Alpha 89, 90, 91 require industry - skipped

    def alpha092(self) -> pd.DataFrame:
        """Alpha #92: min(ts_rank(decay_linear(((high+low)/2+close)<(low+open), 15), 19),
                      ts_rank(decay_linear(correlation(rank(low), rank(adv30), 8), 7), 7))"""
        cond = ((self.high + self.low) / 2 + self.close) < (self.low + self.open)
        part1 = ts_rank(decay_linear(cond.astype(float), 15), 19)
        inner = correlation(rank(self.low), rank(self.adv(30)), 8)
        part2 = ts_rank(decay_linear(inner, 7), 7)
        return min_(part1, part2)

    # Alpha 93 requires industry - skipped

    def alpha094(self) -> pd.DataFrame:
        """Alpha #94: -1 * rank(vwap - ts_min(vwap, 12)) ** ts_rank(correlation(ts_rank(vwap, 20),
                      ts_rank(adv60, 4), 18), 3)"""
        part1 = rank(self.vwap - ts_min(self.vwap, 12))
        inner = correlation(ts_rank(self.vwap, 20), ts_rank(self.adv(60), 4), 18)
        part2 = ts_rank(inner, 3)
        return -1 * np.power(part1, part2)

    def alpha095(self) -> pd.DataFrame:
        """Alpha #95: rank(open - ts_min(open, 12)) < ts_rank(rank(correlation(ts_sum(high+low/2, 19),
                      ts_sum(adv40, 19), 13))^5, 12)"""
        x = rank(self.open - ts_min(self.open, 12))
        inner1 = ts_sum((self.high + self.low) / 2, 19)
        inner2 = ts_sum(self.adv(40), 19)
        inner3 = np.power(rank(correlation(inner1, inner2, 13)), 5)
        y = ts_rank(inner3, 12)
        return (x < y).astype(float)

    def alpha096(self) -> pd.DataFrame:
        """Alpha #96: max(ts_rank(decay_linear(correlation(rank(vwap), rank(volume), 4), 4), 8),
                      ts_rank(decay_linear(ts_argmax(correlation(ts_rank(close, 7), ts_rank(adv60, 4), 4), 13), 14), 13))"""
        inner1 = correlation(rank(self.vwap), rank(self.volume), 4)
        part1 = ts_rank(decay_linear(inner1, 4), 8)
        inner2 = ts_argmax(correlation(ts_rank(self.close, 7), ts_rank(self.adv(60), 4), 4), 13)
        part2 = ts_rank(decay_linear(inner2, 14), 13)
        return max_(part1, part2)

    # Alpha 97 requires industry - skipped

    def alpha098(self) -> pd.DataFrame:
        """Alpha #98: rank(decay_linear(correlation(vwap, ts_sum(adv5, 26), 5), 7)) -
                      rank(decay_linear(ts_rank(ts_argmin(correlation(rank(open), rank(adv15), 21), 9), 7), 8))"""
        part1 = rank(decay_linear(correlation(self.vwap, ts_sum(self.adv(5), 26), 5), 7))
        inner = ts_argmin(correlation(rank(self.open), rank(self.adv(15)), 21), 9)
        part2 = rank(decay_linear(ts_rank(inner, 7), 8))
        return part1 - part2

    def alpha099(self) -> pd.DataFrame:
        """Alpha #99: rank(correlation(ts_sum(high+low/2, 20), ts_sum(adv60, 20), 9)) <
                      rank(correlation(low, volume, 6))"""
        inner1 = ts_sum((self.high + self.low) / 2, 20)
        inner2 = ts_sum(self.adv(60), 20)
        x = rank(correlation(inner1, inner2, 9))
        y = rank(correlation(self.low, self.volume, 6))
        return (x < y).astype(float)

    # Alpha 100 requires industry - skipped

    def alpha101(self) -> pd.DataFrame:
        """Alpha #101: (close - open) / ((high - low) + 0.001)"""
        return (self.close - self.open) / ((self.high - self.low) + 0.001)
