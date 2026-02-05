import requests
from typing import Optional
from loguru import logger


class DartCollector:
    """DART API 부채비율 수집기"""

    BASE_URL = "https://opendart.fss.or.kr/api"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_debt_ratio(self, corp_code: str, year: str = "2024") -> Optional[float]:
        try:
            url = f"{self.BASE_URL}/fnlttSinglAcnt.json"
            params = {
                "crtfc_key": self.api_key,
                "corp_code": corp_code,
                "bsns_year": year,
                "reprt_code": "11011",
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("list", [])
                for item in items:
                    if "부채비율" in item.get("account_nm", ""):
                        return float(item.get("thstrm_dt", "0").replace(",", ""))
        except Exception as e:
            logger.warning(f"DART API error for {corp_code}: {e}")
        return None
