import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union
from lasps.data.collectors.kiwoom_base import KiwoomAPIBase
from lasps.config.sector_config import SECTOR_CODES

MOCK_STOCKS = {
    "005930": {"종목명": "삼성전자", "업종코드": "001", "업종명": "전기전자",
               "시가총액": "400000000000000", "PER": "12.5", "PBR": "1.3", "ROE": "10.2"},
    "000660": {"종목명": "SK하이닉스", "업종코드": "001", "업종명": "전기전자",
               "시가총액": "100000000000000", "PER": "8.0", "PBR": "1.5", "ROE": "15.0"},
    "005380": {"종목명": "현대차", "업종코드": "015", "업종명": "운수장비",
               "시가총액": "50000000000000", "PER": "6.0", "PBR": "0.6", "ROE": "12.0"},
    "035420": {"종목명": "NAVER", "업종코드": "003", "업종명": "서비스업",
               "시가총액": "30000000000000", "PER": "25.0", "PBR": "1.8", "ROE": "8.0"},
    "105560": {"종목명": "KB금융", "업종코드": "002", "업종명": "금융업",
               "시가총액": "25000000000000", "PER": "5.0", "PBR": "0.5", "ROE": "9.0"},
}

# Mock 시장 분류 (KOSPI / KOSDAQ)
_MOCK_KOSPI = ["005930", "000660", "005380", "105560"]
_MOCK_KOSDAQ = ["035420"]


class KiwoomMockAPI(KiwoomAPIBase):
    """키움 OpenAPI Mock (GBM 기반 현실적 가격 생성)"""

    def __init__(self, seed: int = 42):
        self._rng = np.random.RandomState(seed)

    def get_code_list_by_market(self, market: str) -> List[str]:
        """Mock 시장별 종목코드 조회."""
        if market == "0":
            return list(_MOCK_KOSPI)
        elif market == "10":
            return list(_MOCK_KOSDAQ)
        return []

    def request(self, tr_code: str, **kwargs) -> Union[Dict, List[Dict]]:
        stock_code = kwargs.get("종목코드", "005930")
        days = kwargs.get("조회일수", 200)
        if tr_code == "OPT10001":
            return self._stock_info(stock_code)
        elif tr_code == "OPT10081":
            return self._daily_ohlcv(stock_code, days=days)
        elif tr_code == "OPT10059":
            return self._investor_data(stock_code, days=days)
        elif tr_code == "OPT10014":
            return self._short_selling(stock_code, days=days)
        else:
            raise ValueError(f"Unknown TR code: {tr_code}")

    def is_connected(self) -> bool:
        return True

    def _stock_info(self, stock_code: str) -> Dict:
        if stock_code in MOCK_STOCKS:
            return {"종목코드": stock_code, **MOCK_STOCKS[stock_code]}
        sector_codes = list(SECTOR_CODES.keys())
        sc = self._rng.choice(sector_codes)
        _, name, _ = SECTOR_CODES[sc]
        return {
            "종목코드": stock_code, "종목명": f"종목_{stock_code}",
            "업종코드": sc, "업종명": name,
            "시가총액": str(self._rng.randint(1e11, 1e14)),
            "PER": f"{self._rng.uniform(3, 30):.1f}",
            "PBR": f"{self._rng.uniform(0.3, 3.0):.1f}",
            "ROE": f"{self._rng.uniform(2, 20):.1f}",
        }

    def _daily_ohlcv(self, stock_code: str, days: int = 200) -> List[Dict]:
        dates = pd.bdate_range(start="2015-01-02", periods=days)
        base_price = self._rng.uniform(10000, 200000)
        returns = self._rng.normal(0.0005, 0.02, days)
        prices = base_price * np.cumprod(1 + returns)
        rows = []
        for i, date in enumerate(dates):
            close = int(prices[i])
            daily_vol = abs(self._rng.normal(0.015, 0.005))
            high = int(close * (1 + daily_vol))
            low = int(close * (1 - daily_vol))
            open_ = int(close * (1 + self._rng.normal(0, 0.005)))
            volume = int(self._rng.uniform(100000, 5000000))
            rows.append({
                "일자": date.strftime("%Y%m%d"),
                "시가": str(open_), "고가": str(high),
                "저가": str(low), "현재가": str(close),
                "거래량": str(volume),
            })
        return rows

    def _investor_data(self, stock_code: str, days: int = 200) -> List[Dict]:
        dates = pd.bdate_range(start="2015-01-02", periods=days)
        rows = []
        for date in dates:
            foreign = int(self._rng.normal(0, 500000))
            inst = int(self._rng.normal(0, 300000))
            rows.append({
                "일자": date.strftime("%Y%m%d"),
                "외국인순매수": str(foreign), "기관계순매수": str(inst),
                "개인순매수": str(-(foreign + inst)),
            })
        return rows

    def _short_selling(self, stock_code: str, days: int = 200) -> List[Dict]:
        dates = pd.bdate_range(start="2015-01-02", periods=days)
        rows = []
        for date in dates:
            rows.append({
                "일자": date.strftime("%Y%m%d"),
                "공매도량": str(int(self._rng.uniform(1000, 50000))),
                "공매도비중": f"{self._rng.uniform(0.5, 5.0):.2f}",
            })
        return rows
