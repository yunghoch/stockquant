import time
import pandas as pd
import numpy as np
from typing import Optional
from loguru import logger
from lasps.data.collectors.kiwoom_base import KiwoomAPIBase
from lasps.config.sector_config import get_sector_id
from lasps.config.tr_config import TR_CODES


class KiwoomCollector:
    """키움 OpenAPI 데이터 수집기"""

    def __init__(self, api: KiwoomAPIBase):
        self.api = api

    def _request(self, tr_code: str, **kwargs):
        interval = TR_CODES.get(tr_code, {}).get("interval", 0.2)
        time.sleep(interval)
        return self.api.request(tr_code, **kwargs)

    def get_stock_info(self, stock_code: str) -> dict:
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
        resp = self._request("OPT10081", 종목코드=stock_code)
        if not isinstance(resp, list):
            resp = [resp]
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
        resp = self._request("OPT10059", 종목코드=stock_code)
        if not isinstance(resp, list):
            resp = [resp]
        rows = []
        for r in resp[:days]:
            rows.append({
                "date": pd.to_datetime(r["일자"]),
                "foreign_net": int(r.get("외국인순매수", 0)),
                "inst_net": int(r.get("기관계순매수", 0)),
            })
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    def get_short_selling(self, stock_code: str) -> pd.DataFrame:
        resp = self._request("OPT10014", 종목코드=stock_code)
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
