import time
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Union
from loguru import logger
from lasps.data.collectors.kiwoom_base import KiwoomAPIBase
from lasps.config.sector_config import get_sector_id
from lasps.config.tr_config import TR_CODES


class KiwoomCollector:
    """키움 OpenAPI 데이터 수집기"""

    def __init__(self, api: KiwoomAPIBase, rate_limit: bool = True):
        self.api = api
        self.rate_limit = rate_limit

    def _request(self, tr_code: str, **kwargs: Any) -> Union[Dict, List[Dict]]:
        """API 요청 (rate limit 적용).

        Args:
            tr_code: TR 코드.
            **kwargs: TR 요청 파라미터.

        Returns:
            API 응답 (dict 또는 list of dict).

        Raises:
            ConnectionError: API 요청 실패 시.
        """
        if self.rate_limit:
            interval = TR_CODES.get(tr_code, {}).get("interval", 0.2)
            time.sleep(interval)
        try:
            return self.api.request(tr_code, **kwargs)
        except Exception as e:
            raise ConnectionError(
                f"API request failed for {tr_code} ({kwargs}): {e}"
            ) from e

    def get_stock_info(self, stock_code: str) -> dict:
        """종목 기본정보 조회.

        Args:
            stock_code: 종목코드 (e.g., '005930').

        Returns:
            종목 정보 dict.

        Raises:
            ConnectionError: API 요청 실패 시.
            KeyError: 응답 데이터에 필수 필드 누락 시.
        """
        resp = self._request("OPT10001", 종목코드=stock_code)
        sector_code = resp.get("업종코드", "")
        return {
            "code": stock_code,
            "name": resp.get("종목명", ""),
            "market_cap": resp.get("시가총액", ""),
            "per": resp.get("PER", ""),
            "pbr": resp.get("PBR", ""),
            "roe": resp.get("ROE", ""),
            "sector_code": sector_code,
            "sector_name": resp.get("업종명", ""),
            "sector_id": get_sector_id(sector_code),
        }

    def get_daily_ohlcv(self, stock_code: str, days: int = 60) -> pd.DataFrame:
        """일봉 OHLCV 데이터 조회.

        Args:
            stock_code: 종목코드.
            days: 조회 일수.

        Returns:
            OHLCV DataFrame (date, open, high, low, close, volume).

        Raises:
            ConnectionError: API 요청 실패 시.
            ValueError: 응답 데이터가 비어있을 때.
        """
        resp = self._request("OPT10081", 종목코드=stock_code)
        if not isinstance(resp, list):
            resp = [resp]
        if not resp:
            raise ValueError(f"Empty OHLCV response for {stock_code}")
        rows = []
        for r in resp[:days]:
            rows.append({
                "date": pd.to_datetime(r["일자"]),
                "open": abs(int(r["시가"])),
                "high": abs(int(r["고가"])),
                "low": abs(int(r["저가"])),
                "close": abs(int(r["현재가"])),
                "volume": abs(int(r["거래량"])),
            })
        df = pd.DataFrame(rows)
        return df.sort_values("date").reset_index(drop=True)

    def get_investor_data(self, stock_code: str, days: int = 60) -> pd.DataFrame:
        """투자자별 매매동향 조회.

        Args:
            stock_code: 종목코드.
            days: 조회 일수.

        Returns:
            투자자 데이터 DataFrame (date, foreign_net, inst_net).

        Raises:
            ConnectionError: API 요청 실패 시.
        """
        today = pd.Timestamp.today().strftime("%Y%m%d")
        resp = self._request(
            "OPT10059",
            일자=today, 종목코드=stock_code,
            금액수량구분="2", 매매구분="0", 단위구분="1",
        )
        if not isinstance(resp, list):
            resp = [resp]
        rows = []
        for r in resp[:days]:
            # 실제 API: "외국인순매수", Mock: "외국인순매수"
            foreign = r.get("외국인순매수", r.get("외국인", 0))
            # 실제 API: "기관계", Mock: "기관계순매수"
            inst = r.get("기관계순매수", r.get("기관계", 0))
            rows.append({
                "date": pd.to_datetime(r["일자"]),
                "foreign_net": int(foreign),
                "inst_net": int(inst),
            })
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    def get_short_selling(self, stock_code: str) -> pd.DataFrame:
        """공매도 데이터 조회.

        Args:
            stock_code: 종목코드.

        Returns:
            공매도 데이터 DataFrame (date, short_volume, short_ratio).

        Raises:
            ConnectionError: API 요청 실패 시.
        """
        today = pd.Timestamp.today().strftime("%Y%m%d")
        start = (pd.Timestamp.today() - pd.DateOffset(years=1)).strftime("%Y%m%d")
        resp = self._request(
            "OPT10014",
            종목코드=stock_code, 시작일자=start, 종료일자=today,
        )
        if not isinstance(resp, list):
            resp = [resp]
        rows = []
        for r in resp:
            rows.append({
                "date": pd.to_datetime(r["일자"]),
                "short_volume": int(r.get("공매도량", 0)),
                "short_ratio": float(r.get("공매도비중", 0)),
            })
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
