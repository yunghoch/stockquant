import pandas as pd
import numpy as np


class TechnicalIndicatorCalculator:
    """15개 기술지표 계산기: 추세(4) + 모멘텀(4) + 변동성(5) + 거래량(2)"""

    def calculate(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """OHLCV DataFrame에 15개 기술지표를 추가하여 반환한다.

        Args:
            ohlcv_df: date, open, high, low, close, volume 컬럼을 포함하는 DataFrame

        Returns:
            원본 컬럼 + 15개 기술지표 컬럼이 추가된 DataFrame
        """
        df = ohlcv_df.copy()

        # 추세 (4개)
        df["ma5"] = df["close"].rolling(5).mean()
        df["ma20"] = df["close"].rolling(20).mean()
        df["ma60"] = df["close"].rolling(60).mean()
        df["ma120"] = df["close"].rolling(120).mean()

        # 모멘텀: RSI(14)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # 모멘텀: MACD(12,26,9)
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # 변동성: Bollinger Bands(20, 2)
        df["bb_middle"] = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_middle"] + 2 * bb_std
        df["bb_lower"] = df["bb_middle"] - 2 * bb_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

        # 변동성: ATR(14)
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close = abs(df["low"] - df["close"].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = true_range.rolling(14).mean()

        # 거래량: OBV
        obv = [0.0]
        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i - 1]:
                obv.append(obv[-1] + df["volume"].iloc[i])
            elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
                obv.append(obv[-1] - df["volume"].iloc[i])
            else:
                obv.append(obv[-1])
        df["obv"] = obv

        # 거래량: Volume MA20
        df["volume_ma20"] = df["volume"].rolling(20).mean()

        return df
