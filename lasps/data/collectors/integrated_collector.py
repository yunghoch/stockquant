import numpy as np
import pandas as pd
from typing import List, Dict
from loguru import logger
from lasps.data.collectors.kiwoom_base import KiwoomAPIBase
from lasps.data.collectors.kiwoom_collector import KiwoomCollector
from lasps.data.processors.technical_indicators import TechnicalIndicatorCalculator
from lasps.data.processors.market_sentiment import MarketSentimentCalculator
from lasps.data.processors.chart_generator import ChartGenerator
from lasps.config.sector_config import NUM_SECTORS
from lasps.utils.constants import (
    OHLCV_FEATURES, INDICATOR_FEATURES, SENTIMENT_FEATURES,
    TOTAL_FEATURE_DIM, TIME_SERIES_LENGTH,
)
from lasps.utils.helpers import normalize_time_series


class IntegratedCollector:
    """수집 -> 가공 -> 피처 벡터 + 차트 전체 파이프라인"""

    def __init__(self, kiwoom_api: KiwoomAPIBase, rate_limit: bool = True):
        self.kiwoom = KiwoomCollector(kiwoom_api, rate_limit=rate_limit)
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

        if len(feature_cols) != TOTAL_FEATURE_DIM:
            missing = [f for f in all_feat if f not in merged.columns]
            raise ValueError(
                f"Feature count mismatch: expected {TOTAL_FEATURE_DIM}, "
                f"got {len(feature_cols)}. Missing: {missing}"
            )

        sector_id = info["sector_id"]
        if not (0 <= sector_id < NUM_SECTORS):
            raise ValueError(
                f"Invalid sector_id {sector_id} for {stock_code}. "
                f"Expected range [0, {NUM_SECTORS})"
            )

        valid = merged.dropna(subset=feature_cols)
        if len(valid) < TIME_SERIES_LENGTH:
            raise ValueError(
                f"Insufficient data for {stock_code}: "
                f"got {len(valid)} rows, need {TIME_SERIES_LENGTH}"
            )

        recent = valid.tail(TIME_SERIES_LENGTH)
        time_series_25d = recent[feature_cols].values.astype(np.float32)
        time_series_25d = normalize_time_series(time_series_25d)

        # Use same date range as time series for chart (temporal alignment)
        chart_dates = recent["date"].values
        chart_ohlcv = ohlcv[ohlcv["date"].isin(chart_dates)].copy()
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

    def collect_batch(self, stock_codes: List[str]) -> Dict[str, List]:
        """배치 수집. 성공/실패 결과를 분리하여 반환.

        Args:
            stock_codes: 종목코드 리스트.

        Returns:
            Dict with 'results' (성공 데이터 리스트) and 'failures' (실패 정보 리스트).
        """
        results: List[Dict] = []
        failures: List[Dict] = []
        for code in stock_codes:
            try:
                data = self.collect_stock_data(code)
                results.append(data)
                logger.info(f"Collected {code}: {data['info']['name']}")
            except Exception as e:
                failures.append({"code": code, "error": str(e)})
                logger.error(f"Failed to collect {code}: {e}")
        if failures:
            logger.warning(
                f"Batch complete: {len(results)} success, {len(failures)} failed"
            )
        return {"results": results, "failures": failures}
