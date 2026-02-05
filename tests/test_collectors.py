import pytest
import pandas as pd
from lasps.data.collectors.kiwoom_base import KiwoomAPIBase
from lasps.data.collectors.kiwoom_mock import KiwoomMockAPI


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
