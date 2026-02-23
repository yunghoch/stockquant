from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union


class KiwoomAPIBase(ABC):
    """키움 OpenAPI 추상 인터페이스"""

    @abstractmethod
    def request(self, tr_code: str, **kwargs: Any) -> Union[Dict, List[Dict]]:
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        pass

    def request_all(
        self, tr_code: str, max_pages: int = 10, **kwargs: Any
    ) -> List[Dict]:
        """페이지네이션으로 모든 데이터를 조회한다.

        기본 구현은 request()를 한 번 호출한다.
        KiwoomRealAPI에서 실제 페이지네이션으로 오버라이드된다.

        Args:
            tr_code: TR 코드.
            max_pages: 최대 페이지 수.
            **kwargs: TR 요청 파라미터.

        Returns:
            전체 결과 List[Dict].
        """
        result = self.request(tr_code, **kwargs)
        if isinstance(result, dict):
            return [result]
        return result

    def get_code_list_by_market(self, market: str) -> List[str]:
        """시장별 종목코드를 조회한다.

        Args:
            market: 시장 구분 ("0"=KOSPI, "10"=KOSDAQ).

        Returns:
            종목코드 리스트.
        """
        return []
