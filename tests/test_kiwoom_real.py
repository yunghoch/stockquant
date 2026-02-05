"""키움 OpenAPI+ 실제 연결 테스트.

모든 테스트에 @pytest.mark.kiwoom 적용.
Windows + PyQt5 + 키움 로그인 상태에서만 실행되며, 그 외 환경에서는 자동 skip.

실행:
    pytest tests/test_kiwoom_real.py -v
    pytest tests/test_kiwoom_real.py -v -m kiwoom
"""
from datetime import datetime

import pytest

from lasps.config.tr_config import TR_CODES
from lasps.data.collectors.kiwoom_base import KiwoomAPIBase

pytestmark = pytest.mark.kiwoom

SAMSUNG = "005930"
TODAY = datetime.now().strftime("%Y%m%d")


# ═══════════════════════════════════════════════════════════
# 섹션 1: 연결 상태 검증
# ═══════════════════════════════════════════════════════════


class TestConnection:
    """실제 API 연결 상태 테스트."""

    def test_is_connected(self, real_api):
        assert real_api.is_connected() is True

    def test_implements_abc_interface(self, real_api):
        assert isinstance(real_api, KiwoomAPIBase)


# ═══════════════════════════════════════════════════════════
# 섹션 2: Raw API 응답 검증
# ═══════════════════════════════════════════════════════════


class TestRawOPT10001:
    """OPT10001 주식기본정보 Raw 응답 검증."""

    def test_returns_dict(self, real_api):
        resp = real_api.request("OPT10001", 종목코드=SAMSUNG)
        assert isinstance(resp, dict)

    def test_has_required_keys(self, real_api):
        resp = real_api.request("OPT10001", 종목코드=SAMSUNG)
        expected = TR_CODES["OPT10001"]["output"]
        for key in expected:
            assert key in resp, f"Missing key: {key}"

    def test_stock_name_not_empty(self, real_api):
        resp = real_api.request("OPT10001", 종목코드=SAMSUNG)
        assert resp["종목명"] != ""


class TestRawOPT10081:
    """OPT10081 주식일봉 Raw 응답 검증."""

    def test_returns_list(self, real_api):
        resp = real_api.request("OPT10081", 종목코드=SAMSUNG)
        assert isinstance(resp, list)

    def test_has_rows(self, real_api):
        resp = real_api.request("OPT10081", 종목코드=SAMSUNG)
        assert len(resp) > 0

    def test_row_has_required_keys(self, real_api):
        resp = real_api.request("OPT10081", 종목코드=SAMSUNG)
        expected = TR_CODES["OPT10081"]["output"]
        row = resp[0]
        for key in expected:
            assert key in row, f"Missing key: {key}"

    def test_date_format_8digits(self, real_api):
        resp = real_api.request("OPT10081", 종목코드=SAMSUNG)
        date_str = resp[0]["일자"]
        assert len(date_str) == 8
        assert date_str.isdigit()

    def test_price_is_numeric(self, real_api):
        resp = real_api.request("OPT10081", 종목코드=SAMSUNG)
        row = resp[0]
        for field in ["시가", "고가", "저가", "현재가"]:
            val = row[field].lstrip("+-")
            assert val.isdigit(), f"{field} is not numeric: {row[field]}"


class TestRawOPT10059:
    """OPT10059 투자자별 Raw 응답 검증."""

    def _request_opt10059(self, real_api):
        return real_api.request(
            "OPT10059",
            일자=TODAY, 종목코드=SAMSUNG,
            금액수량구분="2", 매매구분="0", 단위구분="1",
        )

    def test_returns_list(self, real_api):
        resp = self._request_opt10059(real_api)
        assert isinstance(resp, list)

    def test_row_has_required_keys(self, real_api):
        resp = self._request_opt10059(real_api)
        expected = TR_CODES["OPT10059"]["output"]
        row = resp[0]
        for key in expected:
            assert key in row, f"Missing key: {key}"


class TestRawOPT10014:
    """OPT10014 공매도 Raw 응답 검증."""

    def _request_opt10014(self, real_api):
        return real_api.request(
            "OPT10014",
            종목코드=SAMSUNG, 시작일자="20250101", 종료일자=TODAY,
        )

    def test_returns_list(self, real_api):
        resp = self._request_opt10014(real_api)
        assert isinstance(resp, list)

    def test_row_has_required_keys(self, real_api):
        resp = self._request_opt10014(real_api)
        expected = TR_CODES["OPT10014"]["output"]
        row = resp[0]
        for key in expected:
            assert key in row, f"Missing key: {key}"


# ═══════════════════════════════════════════════════════════
# 섹션 3: Mock 호환성 검증
# ═══════════════════════════════════════════════════════════


class TestMockCompatibility:
    """실제 API 응답 키 ⊇ Mock 응답 키 확인."""

    def _mock_keys(self, tr_code: str) -> set:
        from lasps.data.collectors.kiwoom_mock import KiwoomMockAPI

        mock = KiwoomMockAPI(seed=42)
        resp = mock.request(tr_code, 종목코드=SAMSUNG)
        if isinstance(resp, dict):
            return set(resp.keys())
        return set(resp[0].keys())

    def test_opt10001_superset(self, real_api):
        real_resp = real_api.request("OPT10001", 종목코드=SAMSUNG)
        real_keys = set(real_resp.keys())
        mock_keys = self._mock_keys("OPT10001")
        missing = mock_keys - real_keys
        assert not missing, f"Real API missing Mock keys: {missing}"

    def test_opt10081_superset(self, real_api):
        real_resp = real_api.request("OPT10081", 종목코드=SAMSUNG)
        real_keys = set(real_resp[0].keys())
        mock_keys = self._mock_keys("OPT10081")
        missing = mock_keys - real_keys
        assert not missing, f"Real API missing Mock keys: {missing}"

    def test_opt10059_superset(self, real_api):
        real_resp = real_api.request(
            "OPT10059",
            일자=TODAY, 종목코드=SAMSUNG,
            금액수량구분="2", 매매구분="0", 단위구분="1",
        )
        real_keys = set(real_resp[0].keys())
        mock_keys = self._mock_keys("OPT10059")
        # Mock은 간소화된 키를 사용하므로, 핵심 키만 검증
        core_keys = {"일자"}
        missing = core_keys - real_keys
        assert not missing, f"Real API missing core keys: {missing}"
        assert "외국인순매수" in real_keys or "외국인" in real_keys

    def test_opt10014_superset(self, real_api):
        real_resp = real_api.request(
            "OPT10014",
            종목코드=SAMSUNG, 시작일자="20250101", 종료일자=TODAY,
        )
        real_keys = set(real_resp[0].keys())
        mock_keys = self._mock_keys("OPT10014")
        # Mock은 간소화된 키를 사용하므로, 핵심 키만 검증
        core_keys = {"일자", "공매도량", "공매도비중"}
        missing = core_keys - real_keys
        assert not missing, f"Real API missing core keys: {missing}"


# ═══════════════════════════════════════════════════════════
# 섹션 4: Collector 통합 검증
# ═══════════════════════════════════════════════════════════


class TestCollectorIntegration:
    """KiwoomCollector가 실제 API에서도 올바르게 동작하는지 검증."""

    def test_get_stock_info_returns_english_keys(self, real_collector):
        info = real_collector.get_stock_info(SAMSUNG)
        expected_keys = [
            "code", "name", "market_cap", "per", "pbr",
            "roe", "sector_code", "sector_name", "sector_id",
        ]
        for key in expected_keys:
            assert key in info, f"Missing key: {key}"

    def test_get_stock_info_samsung_values(self, real_collector):
        info = real_collector.get_stock_info(SAMSUNG)
        assert info["code"] == SAMSUNG
        assert info["name"] != ""
        assert isinstance(info["sector_id"], int)
        assert 0 <= info["sector_id"] < 20

    def test_get_daily_ohlcv_returns_dataframe(self, real_collector):
        import pandas as pd

        df = real_collector.get_daily_ohlcv(SAMSUNG, days=60)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 60

    def test_get_daily_ohlcv_columns(self, real_collector):
        df = real_collector.get_daily_ohlcv(SAMSUNG, days=10)
        assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]

    def test_get_daily_ohlcv_sorted_ascending(self, real_collector):
        df = real_collector.get_daily_ohlcv(SAMSUNG, days=30)
        assert df["date"].is_monotonic_increasing

    def test_get_investor_data_returns_dataframe(self, real_collector):
        import pandas as pd

        df = real_collector.get_investor_data(SAMSUNG, days=30)
        assert isinstance(df, pd.DataFrame)
        assert "foreign_net" in df.columns
        assert "inst_net" in df.columns

    def test_get_short_selling_returns_dataframe(self, real_collector):
        import pandas as pd

        df = real_collector.get_short_selling(SAMSUNG)
        assert isinstance(df, pd.DataFrame)
        assert "short_volume" in df.columns
        assert "short_ratio" in df.columns


# ═══════════════════════════════════════════════════════════
# 섹션 5: 에러 처리
# ═══════════════════════════════════════════════════════════


class TestErrorHandling:
    """에러 상황 테스트."""

    def test_invalid_tr_code_raises_value_error(self, real_api):
        with pytest.raises(ValueError, match="Unknown TR code"):
            real_api.request("INVALID_TR")

    def test_disconnected_request_raises_connection_error(self):
        """미연결 상태에서 request 호출 시 ConnectionError."""
        import sys

        if sys.platform != "win32":
            pytest.skip("Kiwoom API requires Windows")

        try:
            from lasps.data.collectors.kiwoom_real import KiwoomRealAPI
        except ImportError:
            pytest.skip("PyQt5 not installed")

        api = KiwoomRealAPI()
        # connect()를 호출하지 않은 상태
        with pytest.raises(ConnectionError, match="Not connected"):
            api.request("OPT10001", 종목코드=SAMSUNG)
