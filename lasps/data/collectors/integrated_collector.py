import numpy as np
import pandas as pd
from typing import List, Dict
from loguru import logger
from lasps.data.collectors.kiwoom_base import KiwoomAPIBase
from lasps.data.collectors.kiwoom_collector import KiwoomCollector
from lasps.data.processors.technical_indicators import TechnicalIndicatorCalculator
from lasps.data.processors.market_sentiment import MarketSentimentCalculator
from lasps.data.processors.chart_generator import ChartGenerator
from lasps.utils.constants import (
    OHLCV_FEATURES, INDICATOR_FEATURES, SENTIMENT_FEATURES,
    TIME_SERIES_LENGTH,
)


class IntegratedCollector:
    """수집 -> 가공 -> 피처 벡터 + 차트 전체 파이프라인"""

    def __init__(self, kiwoom_api: KiwoomAPIBase):
        self.kiwoom = KiwoomCollector(kiwoom_api)
        self.indicator_calc = TechnicalIndicatorCalculator()
        self.sentiment_calc = MarketSentimentCalculator()
        self.chart_gen = ChartGenerator()

    def collect_stock_data(self, stock_code: str) -> Dict:
        info = self.kiwoom.get_stock_info(stock_code)
        ohlcv = self.kiwoom.get_daily_ohlcv(stock_code, days=180)
        investor = self.kiwoom.get_investor_data(stock_code, days=180)

        with_indicators = self.indicator_calc.calculate(ohlcv)
        sentiment = self.sentiment_calc.calculate(ohlcv, investor)

        merged = with_indicators.merge(sentiment, on="date", how="left")
        all_feat = OHLCV_FEATURES + INDICATOR_FEATURES + SENTIMENT_FEATURES
        feature_cols = [c for c in all_feat if c in merged.columns]

        valid = merged.dropna(subset=feature_cols)
        recent = valid.tail(TIME_SERIES_LENGTH)
        time_series_25d = recent[feature_cols].values.astype(np.float32)

        chart_ohlcv = ohlcv.tail(TIME_SERIES_LENGTH).copy()
        chart_ohlcv.index = pd.to_datetime(chart_ohlcv["date"])
        chart_df = chart_ohlcv.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })
        chart_tensor = self.chart_gen.generate_tensor(
            chart_df[["Open", "High", "Low", "Close", "Volume"]]
        )

        return {
            "info": info,
            "time_series_25d": time_series_25d,
            "chart_tensor": chart_tensor,
            "sector_id": info["sector_id"],
        }

    def collect_batch(self, stock_codes: List[str]) -> List[Dict]:
        results = []
        for code in stock_codes:
            try:
                data = self.collect_stock_data(code)
                results.append(data)
                logger.info(f"Collected {code}: {data['info']['name']}")
            except Exception as e:
                logger.error(f"Failed to collect {code}: {e}")
        return results
