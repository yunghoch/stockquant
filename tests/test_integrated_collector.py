import pytest
from lasps.data.collectors.kiwoom_mock import KiwoomMockAPI
from lasps.data.collectors.integrated_collector import IntegratedCollector


@pytest.fixture
def int_collector():
    return IntegratedCollector(KiwoomMockAPI(seed=42))


def test_collect_single_stock(int_collector):
    result = int_collector.collect_stock_data("005930")
    assert "info" in result
    assert "time_series_25d" in result
    assert "chart_tensor" in result
    assert result["time_series_25d"].shape[1] == 25
    assert result["chart_tensor"].shape == (3, 224, 224)


def test_collect_batch(int_collector):
    codes = ["005930", "000660", "005380"]
    results = int_collector.collect_batch(codes)
    assert len(results) == 3
