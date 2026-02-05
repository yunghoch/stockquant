"""Phase 2 Milestone: 키움 데이터 수집 모듈 통합 테스트"""

import pytest
import pandas as pd
from lasps.data.collectors.kiwoom_mock import KiwoomMockAPI
from lasps.data.collectors.kiwoom_collector import KiwoomCollector


@pytest.fixture
def collector():
    return KiwoomCollector(KiwoomMockAPI(seed=42))


class TestEndToEndCollection:
    def test_full_stock_pipeline(self, collector):
        code = "005930"
        info = collector.get_stock_info(code)
        assert info["sector_id"] >= 0
        assert info["sector_id"] < 20

        ohlcv = collector.get_daily_ohlcv(code, days=60)
        assert len(ohlcv) == 60
        assert ohlcv["close"].notna().all()

        investor = collector.get_investor_data(code, days=60)
        assert len(investor) == 60

        short = collector.get_short_selling(code)
        assert len(short) > 0

    def test_multiple_stocks(self, collector):
        codes = ["005930", "000660", "005380", "035420", "105560"]
        results = []
        for code in codes:
            info = collector.get_stock_info(code)
            ohlcv = collector.get_daily_ohlcv(code, days=60)
            results.append({"info": info, "ohlcv_len": len(ohlcv)})
        assert len(results) == 5
        sector_ids = [r["info"]["sector_id"] for r in results]
        assert all(0 <= sid < 20 for sid in sector_ids)

    def test_ohlcv_data_quality(self, collector):
        ohlcv = collector.get_daily_ohlcv("005930", days=100)
        assert (ohlcv["high"] >= ohlcv["low"]).all()
        assert (ohlcv["volume"] > 0).all()
        assert (ohlcv["close"] > 0).all()
        date_diffs = ohlcv["date"].diff().dropna()
        assert date_diffs.max() <= pd.Timedelta(days=5)

    def test_investor_data_balance(self, collector):
        investor = collector.get_investor_data("005930", days=60)
        assert "foreign_net" in investor.columns
        assert "inst_net" in investor.columns
