"""Phase 3 Milestone: 수집 -> 25차원 피처 + 차트 이미지 통합 테스트"""

import pytest
import pandas as pd
import numpy as np
from lasps.data.collectors.kiwoom_mock import KiwoomMockAPI
from lasps.data.collectors.kiwoom_collector import KiwoomCollector
from lasps.data.processors.technical_indicators import TechnicalIndicatorCalculator
from lasps.data.processors.market_sentiment import MarketSentimentCalculator
from lasps.data.processors.chart_generator import ChartGenerator
from lasps.utils.constants import (
    INDICATOR_FEATURES,
    SENTIMENT_FEATURES,
    TOTAL_FEATURE_DIM,
    ALL_FEATURES,
)


@pytest.fixture
def collector():
    return KiwoomCollector(KiwoomMockAPI(seed=42), rate_limit=False)


def test_full_pipeline_single_stock(collector):
    """수집 -> 기술지표 -> 감성 -> 25차원 피처 매트릭스 통합 테스트"""
    code = "005930"
    ohlcv = collector.get_daily_ohlcv(code, days=200)
    investor = collector.get_investor_data(code, days=200)

    # 기술지표 계산
    ind_calc = TechnicalIndicatorCalculator()
    with_indicators = ind_calc.calculate(ohlcv)
    for feat in INDICATOR_FEATURES:
        assert feat in with_indicators.columns

    # 감성 계산
    sent_calc = MarketSentimentCalculator()
    sentiment = sent_calc.calculate(ohlcv, investor)
    for feat in SENTIMENT_FEATURES:
        assert feat in sentiment.columns

    # 병합
    merged = with_indicators.merge(sentiment, on="date", how="left")
    feature_cols = [c for c in ALL_FEATURES if c in merged.columns]
    valid = merged.dropna(subset=feature_cols)
    assert len(valid) >= 60, f"Only {len(valid)} valid rows"

    # 최근 60일 피처 매트릭스 추출
    recent = valid.tail(60)
    feature_matrix = recent[feature_cols].values
    assert feature_matrix.shape == (60, TOTAL_FEATURE_DIM)


def test_chart_from_collected_data(collector):
    """수집 데이터 -> 캔들차트 텐서 변환 통합 테스트"""
    ohlcv = collector.get_daily_ohlcv("005930", days=60)
    chart_df = ohlcv.copy()
    chart_df.index = pd.to_datetime(chart_df["date"])
    chart_df = chart_df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })
    gen = ChartGenerator()
    tensor = gen.generate_tensor(chart_df[["Open", "High", "Low", "Close", "Volume"]])
    assert tensor.shape == (3, 224, 224)
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0
