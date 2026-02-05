import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from lasps.data.collectors.kiwoom_base import KiwoomAPIBase
from lasps.data.collectors.kiwoom_mock import KiwoomMockAPI
from lasps.data.collectors.kiwoom_collector import KiwoomCollector
from lasps.data.collectors.dart_collector import DartCollector


# ── Task 2.1: Mock API Tests ──


def test_mock_api_implements_interface():
    mock = KiwoomMockAPI()
    assert isinstance(mock, KiwoomAPIBase)


def test_mock_get_stock_info():
    mock = KiwoomMockAPI()
    info = mock.request("OPT10001", 종목코드="005930")
    assert "종목코드" in info
    assert "업종코드" in info
    assert "종목명" in info


def test_mock_get_daily_ohlcv():
    mock = KiwoomMockAPI()
    data = mock.request("OPT10081", 종목코드="005930")
    assert isinstance(data, list)
    assert len(data) >= 60
    row = data[0]
    assert "일자" in row
    assert "시가" in row
    assert "현재가" in row
    assert "거래량" in row


def test_mock_get_investor_data():
    mock = KiwoomMockAPI()
    data = mock.request("OPT10059", 종목코드="005930")
    assert isinstance(data, list)
    row = data[0]
    assert "외국인순매수" in row
    assert "기관계순매수" in row


# ── Task 2.2: KiwoomCollector Tests ──


@pytest.fixture
def collector():
    return KiwoomCollector(KiwoomMockAPI(seed=42))


def test_collector_get_stock_info(collector):
    info = collector.get_stock_info("005930")
    assert info["code"] == "005930"
    assert info["name"] == "삼성전자"
    assert info["sector_code"] == "001"
    assert info["sector_id"] == 0


def test_collector_get_daily_ohlcv(collector):
    df = collector.get_daily_ohlcv("005930", days=60)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 60
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert df["date"].is_monotonic_increasing


def test_collector_get_investor_data(collector):
    df = collector.get_investor_data("005930", days=60)
    assert isinstance(df, pd.DataFrame)
    assert "foreign_net" in df.columns
    assert "inst_net" in df.columns
    assert len(df) == 60


def test_collector_ohlcv_price_sanity(collector):
    df = collector.get_daily_ohlcv("005930", days=60)
    assert (df["high"] >= df["low"]).all()


# ── Task 2.3: DART Collector Tests ──


def test_dart_collector_returns_ratio():
    mock_response_data = {
        "status": "000",
        "list": [{"account_nm": "부채비율", "thstrm_dt": "45.2"}],
    }
    with patch("lasps.data.collectors.dart_collector.requests") as mock_req:
        mock_req.get.return_value.status_code = 200
        mock_req.get.return_value.json.return_value = mock_response_data
        collector = DartCollector(api_key="test_key")
        ratio = collector.get_debt_ratio("00126380")
        assert isinstance(ratio, float)


def test_dart_collector_handles_error():
    with patch("lasps.data.collectors.dart_collector.requests") as mock_req:
        mock_req.get.side_effect = Exception("Network error")
        collector = DartCollector(api_key="test_key")
        ratio = collector.get_debt_ratio("00126380")
        assert ratio is None
