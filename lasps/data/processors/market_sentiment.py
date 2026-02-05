import numpy as np
import pandas as pd
from typing import Optional


class MarketSentimentCalculator:
    """시장 기반 감성 5차원 계산기 (PRD 기준)

    5D Sentiment Features:
        1. volume_ratio: clip(volume / MA20, 0, 3) / 3 -> 0~1
        2. volatility_ratio: clip(TR / ATR20, 0, 3) / 3 -> 0~1
        3. gap_direction: clip(gap%, -0.1, 0.1) * 10 -> -1~+1
        4. rsi_norm: RSI(14) / 100 -> 0~1
        5. foreign_inst_flow: sign * min(1, log10(|flow|+1)/8) -> -1~+1
    """

    LOOKBACK = 20

    DEFAULT_VALUES = {
        "volume_ratio": 0.33,
        "volatility_ratio": 0.33,
        "gap_direction": 0.0,
        "rsi_norm": 0.5,
        "foreign_inst_flow": 0.0,
    }

    def calculate(
        self,
        ohlcv_df: pd.DataFrame,
        investor_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """OHLCV와 투자자 데이터로 5D 감성 지표를 계산한다.

        Args:
            ohlcv_df: date, open, high, low, close, volume 컬럼 포함 DataFrame
            investor_df: date, foreign_net, inst_net 컬럼 포함 DataFrame (Optional)

        Returns:
            date + 5개 감성 피처 컬럼을 포함하는 DataFrame
        """
        df = ohlcv_df.copy()

        # 1. volume_ratio
        df["volume_ma20"] = df["volume"].rolling(self.LOOKBACK).mean()
        df["volume_ratio"] = (df["volume"] / df["volume_ma20"]).clip(0, 3) / 3

        # 2. volatility_ratio
        df["true_range"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1)),
            ),
        )
        df["atr_20"] = df["true_range"].rolling(self.LOOKBACK).mean()
        df["volatility_ratio"] = (df["true_range"] / df["atr_20"]).clip(0, 3) / 3

        # 3. gap_direction
        df["prev_close"] = df["close"].shift(1)
        df["gap_pct"] = (df["open"] - df["prev_close"]) / df["prev_close"]
        df["gap_direction"] = df["gap_pct"].clip(-0.1, 0.1) * 10

        # 4. rsi_norm
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi_norm"] = df["rsi"] / 100

        # 5. foreign_inst_flow
        if investor_df is not None and len(investor_df) > 0:
            merged = df.merge(investor_df, on="date", how="left")
            merged["total_flow"] = (
                merged["foreign_net"].fillna(0) + merged["inst_net"].fillna(0)
            )
            merged["flow_sign"] = np.sign(merged["total_flow"])
            merged["flow_magnitude"] = np.minimum(
                1, np.log10(abs(merged["total_flow"]) + 1) / 8
            )
            merged["foreign_inst_flow"] = (
                merged["flow_sign"] * merged["flow_magnitude"]
            )
            df = merged
        else:
            df["foreign_inst_flow"] = 0.0

        # 결과 추출 + NaN 기본값 채우기
        result = df[
            [
                "date",
                "volume_ratio",
                "volatility_ratio",
                "gap_direction",
                "rsi_norm",
                "foreign_inst_flow",
            ]
        ].copy()
        for col, default in self.DEFAULT_VALUES.items():
            result[col] = result[col].fillna(default)
        return result

    def get_feature_names(self) -> list:
        """감성 피처 이름 리스트를 반환한다."""
        return [
            "volume_ratio",
            "volatility_ratio",
            "gap_direction",
            "rsi_norm",
            "foreign_inst_flow",
        ]
