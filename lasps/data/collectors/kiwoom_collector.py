from typing import Any, Dict, List, Union

import pandas as pd
from loguru import logger

from lasps.config.sector_config import get_sector_id
from lasps.config.tr_config import TR_CODES
from lasps.data.collectors.kiwoom_base import KiwoomAPIBase


class KiwoomCollector:
    """키움 OpenAPI 데이터 수집기.

    Mock/Real API 모두 동일한 인터페이스로 사용 가능하다.
    10년치 대량 데이터 수집 시 페이지네이션을 자동으로 처리한다.
    """

    def __init__(self, api: KiwoomAPIBase, rate_limit: bool = True):
        self.api = api
        self.rate_limit = rate_limit

    def _request(self, tr_code: str, **kwargs: Any) -> Union[Dict, List[Dict]]:
        """단일 API 요청 (rate limit 적용).

        Args:
            tr_code: TR 코드.
            **kwargs: TR 요청 파라미터.

        Returns:
            API 응답 (dict 또는 list of dict).

        Raises:
            ConnectionError: API 요청 실패 시.
        """
        try:
            return self.api.request(tr_code, **kwargs)
        except Exception as e:
            raise ConnectionError(
                f"API request failed for {tr_code} ({kwargs}): {e}"
            ) from e

    def _request_all(self, tr_code: str, max_pages: int = 10, **kwargs: Any) -> List[Dict]:
        """페이지네이션 API 요청.

        Args:
            tr_code: TR 코드.
            max_pages: 최대 페이지 수.
            **kwargs: TR 요청 파라미터.

        Returns:
            전체 결과 List[Dict].

        Raises:
            ConnectionError: API 요청 실패 시.
        """
        try:
            return self.api.request_all(tr_code, max_pages=max_pages, **kwargs)
        except Exception as e:
            raise ConnectionError(
                f"API request_all failed for {tr_code} ({kwargs}): {e}"
            ) from e

    def get_all_stock_codes(self) -> List[str]:
        """KOSPI + KOSDAQ 전체 종목코드를 조회한다.

        Returns:
            종목코드 리스트 (중복 제거).
        """
        kospi = self.api.get_code_list_by_market("0")
        kosdaq = self.api.get_code_list_by_market("10")
        all_codes = list(dict.fromkeys(kospi + kosdaq))  # 순서 유지 중복 제거
        logger.info(f"Stock codes: KOSPI={len(kospi)}, KOSDAQ={len(kosdaq)}, total={len(all_codes)}")
        return all_codes

    def get_stock_info(self, stock_code: str) -> dict:
        """종목 기본정보 조회 (OPT10001).

        Args:
            stock_code: 종목코드 (e.g., '005930').

        Returns:
            종목 정보 dict.
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

    def get_daily_ohlcv(self, stock_code: str, days: int = 2600) -> pd.DataFrame:
        """일봉 OHLCV 데이터 조회 (OPT10081).

        10년치(~2,500일) 등 대량 데이터는 자동 페이지네이션으로 수집한다.

        Args:
            stock_code: 종목코드.
            days: 필요한 거래일 수 (기본 2600 = 약 10년).

        Returns:
            OHLCV DataFrame (date, open, high, low, close, volume).
        """
        today = pd.Timestamp.today().strftime("%Y%m%d")
        max_pages = max(1, (days // 500) + 1)

        resp = self._request_all(
            "OPT10081",
            max_pages=max_pages,
            종목코드=stock_code,
            기준일자=today,
            수정주가구분="1",
            조회일수=days,
        )

        if not resp:
            raise ValueError(f"Empty OHLCV response for {stock_code}")

        rows = []
        for r in resp:
            try:
                date_str = r.get("일자", "")
                if not date_str:
                    continue
                rows.append({
                    "date": pd.to_datetime(date_str),
                    "open": abs(int(r.get("시가", 0))),
                    "high": abs(int(r.get("고가", 0))),
                    "low": abs(int(r.get("저가", 0))),
                    "close": abs(int(r.get("현재가", 0))),
                    "volume": abs(int(r.get("거래량", 0))),
                })
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid OHLCV row for {stock_code}: {e}")
                continue

        if not rows:
            raise ValueError(f"No valid OHLCV rows for {stock_code}")

        df = pd.DataFrame(rows)
        df = df.sort_values("date").reset_index(drop=True)

        # 요청 일수 이내로 제한
        if len(df) > days:
            df = df.tail(days).reset_index(drop=True)

        return df

    def get_investor_data(self, stock_code: str, days: int = 2600) -> pd.DataFrame:
        """투자자별 매매동향 조회 (OPT10059).

        Args:
            stock_code: 종목코드.
            days: 필요한 거래일 수.

        Returns:
            투자자 데이터 DataFrame (date, foreign_net, inst_net).
        """
        today = pd.Timestamp.today().strftime("%Y%m%d")
        max_pages = max(1, (days // 500) + 1)

        resp = self._request_all(
            "OPT10059",
            max_pages=max_pages,
            일자=today,
            종목코드=stock_code,
            금액수량구분="2",
            매매구분="0",
            단위구분="1",
            조회일수=days,
        )

        if not isinstance(resp, list):
            resp = [resp]

        rows = []
        for r in resp:
            try:
                date_str = r.get("일자", "")
                if not date_str:
                    continue
                foreign = r.get("외국인순매수", r.get("외국인", 0))
                inst = r.get("기관계순매수", r.get("기관계", 0))
                rows.append({
                    "date": pd.to_datetime(date_str),
                    "foreign_net": int(foreign),
                    "inst_net": int(inst),
                })
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid investor row for {stock_code}: {e}")
                continue

        df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["date", "foreign_net", "inst_net"]
        )
        if not df.empty:
            df = df.sort_values("date").reset_index(drop=True)
            if len(df) > days:
                df = df.tail(days).reset_index(drop=True)
        return df

    def get_short_selling(self, stock_code: str, days: int = 2600) -> pd.DataFrame:
        """공매도 데이터 조회 (OPT10014).

        Args:
            stock_code: 종목코드.
            days: 필요한 거래일 수.

        Returns:
            공매도 데이터 DataFrame (date, short_volume, short_ratio).
        """
        today = pd.Timestamp.today()
        # 공매도는 시작/종료일자 기반이므로 10년 전부터 조회
        start = (today - pd.DateOffset(years=10)).strftime("%Y%m%d")
        end = today.strftime("%Y%m%d")
        max_pages = max(1, (days // 500) + 1)

        resp = self._request_all(
            "OPT10014",
            max_pages=max_pages,
            종목코드=stock_code,
            시작일자=start,
            종료일자=end,
            조회일수=days,
        )

        if not isinstance(resp, list):
            resp = [resp]

        rows = []
        for r in resp:
            try:
                date_str = r.get("일자", "")
                if not date_str:
                    continue
                rows.append({
                    "date": pd.to_datetime(date_str),
                    "short_volume": int(r.get("공매도량", 0)),
                    "short_ratio": float(r.get("공매도비중", 0)),
                })
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid short selling row for {stock_code}: {e}")
                continue

        df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["date", "short_volume", "short_ratio"]
        )
        if not df.empty:
            df = df.sort_values("date").reset_index(drop=True)
        return df
