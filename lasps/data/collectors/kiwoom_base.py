from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union


class KiwoomAPIBase(ABC):
    """키움 OpenAPI 추상 인터페이스"""

    @abstractmethod
    def request(self, tr_code: str, **kwargs) -> Union[Dict, List[Dict]]:
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        pass
