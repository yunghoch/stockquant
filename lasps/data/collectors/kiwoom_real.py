"""키움 OpenAPI+ 실제 연결 어댑터.

PyQt5 QAxWidget을 사용하여 키움 OpenAPI COM 객체에 접속한다.
processEvents() 폴링으로 비동기 이벤트를 동기 request() 호출로 래핑한다.

Note:
    - Windows 32-bit Python + 키움 OpenAPI+ 설치 필요
    - 키움 로그인 다이얼로그가 뜨므로 GUI 환경 필수
"""
import sys
import time
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from lasps.config.tr_config import TR_CODES
from lasps.data.collectors.kiwoom_base import KiwoomAPIBase

try:
    from PyQt5.QAxContainer import QAxWidget
    from PyQt5.QtCore import QCoreApplication
    from PyQt5.QtWidgets import QApplication

    HAS_PYQT5 = True
except ImportError:
    HAS_PYQT5 = False


_REQUEST_TIMEOUT_S = 30.0  # 30초 기본 타임아웃 (10년치 대량 데이터 수집용)
_LOGIN_TIMEOUT_S = 120.0  # 로그인 대기 2분
_POLL_INTERVAL_S = 0.05  # 50ms 폴링 간격

# 키움 API 조회 제한 정책
_MAX_REQUESTS_PER_HOUR = 1000  # 시간당 최대 요청 수
_SAFE_REQUESTS_PER_HOUR = 900  # 안전 마진 적용 (90%)
_HOURLY_PAUSE_SECONDS = 60  # 시간당 제한 도달 시 대기 시간


def _ensure_qapp() -> "QApplication":
    """QApplication 인스턴스가 없으면 생성한다."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


class KiwoomRealAPI(KiwoomAPIBase):
    """키움 OpenAPI+ 실제 연결 어댑터.

    Args:
        timeout_s: TR 요청 타임아웃 (초).

    Raises:
        ImportError: PyQt5가 설치되지 않은 환경.
        RuntimeError: Windows가 아닌 환경.
    """

    def __init__(self, timeout_s: float = _REQUEST_TIMEOUT_S) -> None:
        if not HAS_PYQT5:
            raise ImportError(
                "PyQt5 is required for KiwoomRealAPI. "
                "Install with: pip install PyQt5>=5.15.0"
            )
        if sys.platform != "win32":
            raise RuntimeError("KiwoomRealAPI requires Windows (win32)")

        self._timeout_s = timeout_s
        self._app = _ensure_qapp()
        self._ocx: Optional[QAxWidget] = None
        self._connected = False

        # 콜백 상태 플래그
        self._login_done = False
        self._login_result: int = -1
        self._tr_done = False
        self._response_data: List[Dict[str, str]] = []
        self._prev_next: int = 0

        # Rate limiting
        self._last_request_time: float = 0.0

        # 시간당 요청 제한 추적
        self._hourly_request_times: List[float] = []

        # 스크린번호 순환 (0101~0199)
        self._screen_counter: int = 0

    # ── 연결 관리 ──────────────────────────────────────────

    def connect(self) -> bool:
        """키움 OpenAPI 로그인.

        로그인 다이얼로그가 표시되며, 사용자가 로그인을 완료하면 반환된다.

        Returns:
            로그인 성공 여부.
        """
        if self._connected:
            logger.info("Already connected to Kiwoom API")
            return True

        self._ocx = QAxWidget("KHOPENAPI.KHOpenAPICtrl.1")

        if self._ocx.isNull():
            logger.error(
                "Kiwoom OpenAPI+ COM object not found. "
                "Install Kiwoom OpenAPI+ from https://www.kiwoom.com"
            )
            return False

        self._ocx.OnEventConnect.connect(self._on_event_connect)
        self._ocx.OnReceiveTrData.connect(self._on_receive_tr_data)

        self._login_done = False
        self._login_result = -1
        self._ocx.dynamicCall("CommConnect()")

        # processEvents 폴링으로 로그인 완료 대기
        deadline = time.time() + _LOGIN_TIMEOUT_S
        while not self._login_done and time.time() < deadline:
            QCoreApplication.processEvents()
            time.sleep(_POLL_INTERVAL_S)

        state = self._ocx.dynamicCall("GetConnectState()")
        if state == 1:
            self._connected = True
            logger.info("Kiwoom API login successful")
        else:
            logger.error(
                f"Kiwoom API login failed "
                f"(login_result={self._login_result}, state={state})"
            )

        return self._connected

    def get_code_list_by_market(self, market: str) -> List[str]:
        """시장별 종목코드를 조회한다.

        Args:
            market: 시장 구분 ("0"=KOSPI, "10"=KOSDAQ).

        Returns:
            종목코드 리스트.

        Raises:
            ConnectionError: 미연결 상태.
        """
        if not self._connected:
            raise ConnectionError("Not connected to Kiwoom API")

        result = self._ocx.dynamicCall(
            "GetCodeListByMarket(QString)", market
        )
        if not result:
            return []
        return [code for code in result.split(";") if code.strip()]

    def disconnect(self) -> None:
        """연결 종료."""
        if self._ocx is not None:
            self._ocx.dynamicCall("CommTerminate()")
            self._connected = False
            logger.info("Kiwoom API disconnected")

    def is_connected(self) -> bool:
        """연결 상태 확인."""
        if self._ocx is None:
            return False
        status = self._ocx.dynamicCall("GetConnectState()")
        self._connected = status == 1
        return self._connected

    def _next_screen_no(self) -> str:
        """스크린번호를 0101~0199 범위에서 순환 반환한다."""
        num = 101 + (self._screen_counter % 99)
        self._screen_counter += 1
        return f"{num:04d}"

    # ── 데이터 요청 ────────────────────────────────────────

    def request(self, tr_code: str, **kwargs: Any) -> Union[Dict, List[Dict]]:
        """단일 TR 요청을 수행한다.

        Args:
            tr_code: TR 코드 (e.g., 'OPT10001').
            **kwargs: 입력 파라미터 (한글 키=값).

        Returns:
            단일 응답이면 Dict, 복수 응답이면 List[Dict].

        Raises:
            ConnectionError: 미연결 상태에서 요청.
            ValueError: 알 수 없는 TR 코드.
            TimeoutError: 응답 타임아웃.
        """
        if not self._connected:
            raise ConnectionError("Not connected to Kiwoom API")

        if tr_code not in TR_CODES:
            raise ValueError(f"Unknown TR code: {tr_code}")

        tr_info = TR_CODES[tr_code]
        return self._send_request(tr_code, tr_info, prev_next=0, **kwargs)

    def request_all(
        self, tr_code: str, max_pages: int = 5, **kwargs: Any
    ) -> List[Dict]:
        """페이지네이션을 통해 모든 데이터를 조회한다.

        Args:
            tr_code: TR 코드.
            max_pages: 최대 페이지 수.
            **kwargs: 입력 파라미터.

        Returns:
            전체 결과 List[Dict].

        Raises:
            ConnectionError: 미연결 상태.
            ValueError: 알 수 없는 TR 코드.
            TimeoutError: 응답 타임아웃.
        """
        if not self._connected:
            raise ConnectionError("Not connected to Kiwoom API")

        if tr_code not in TR_CODES:
            raise ValueError(f"Unknown TR code: {tr_code}")

        tr_info = TR_CODES[tr_code]
        is_data_ex = tr_info.get("data_ex", False)
        all_data: List[Dict] = []

        prev_next = 0
        for page in range(max_pages):
            try:
                result = self._send_request(
                    tr_code, tr_info, prev_next=prev_next, **kwargs
                )
            except TimeoutError:
                if page == 0:
                    raise  # 첫 페이지 타임아웃은 진짜 오류
                logger.debug(
                    f"{tr_code} page {page + 1} timeout, stopping pagination"
                )
                break

            if isinstance(result, dict):
                all_data.append(result)
                break

            page_rows = len(result)
            if page_rows == 0:
                break
            all_data.extend(result)

            # data_ex TR은 prev_next 콜백이 신뢰할 수 없으므로
            # 페이지 행 수가 580 이상이면 다음 페이지가 있다고 판단
            if is_data_ex:
                if page_rows < 580:
                    break
                prev_next = 2
            else:
                if self._prev_next == 0:
                    break
                prev_next = 2

            logger.debug(
                f"{tr_code} page {page + 1} fetched ({page_rows} rows)"
            )

        logger.info(f"{tr_code} request_all: {len(all_data)} total rows")
        return all_data

    # ── 내부 메서드 ────────────────────────────────────────

    def _check_hourly_limit(self) -> None:
        """시간당 요청 제한을 확인하고 필요시 대기한다."""
        now = time.time()
        one_hour_ago = now - 3600

        # 1시간 이내의 요청만 유지
        self._hourly_request_times = [
            t for t in self._hourly_request_times if t > one_hour_ago
        ]

        hourly_count = len(self._hourly_request_times)

        # 안전 한도 도달 시 경고 및 대기
        if hourly_count >= _SAFE_REQUESTS_PER_HOUR:
            # 가장 오래된 요청이 만료될 때까지 대기
            if self._hourly_request_times:
                oldest = self._hourly_request_times[0]
                wait_time = oldest + 3600 - now + 1  # +1초 여유
                if wait_time > 0:
                    logger.warning(
                        f"시간당 요청 제한 도달 ({hourly_count}/{_MAX_REQUESTS_PER_HOUR}), "
                        f"{wait_time:.0f}초 대기 중..."
                    )
                    time.sleep(min(wait_time, _HOURLY_PAUSE_SECONDS))
                    # 대기 후 다시 정리
                    now = time.time()
                    one_hour_ago = now - 3600
                    self._hourly_request_times = [
                        t for t in self._hourly_request_times if t > one_hour_ago
                    ]

    def get_hourly_request_count(self) -> int:
        """현재 시간 내 요청 수를 반환한다."""
        now = time.time()
        one_hour_ago = now - 3600
        return len([t for t in self._hourly_request_times if t > one_hour_ago])

    def _send_request(
        self,
        tr_code: str,
        tr_info: Dict,
        prev_next: int = 0,
        **kwargs: Any,
    ) -> Union[Dict, List[Dict]]:
        """SetInputValue → CommRqData → OnReceiveTrData 대기."""
        # 시간당 요청 제한 확인
        self._check_hourly_limit()

        # Rate limiting: TR 간격 준수
        interval = TR_CODES.get(tr_code, {}).get("interval", 0.2)
        elapsed = time.time() - self._last_request_time
        if elapsed < interval:
            time.sleep(interval - elapsed)

        # 입력값 설정
        for key, value in kwargs.items():
            self._ocx.dynamicCall(
                "SetInputValue(QString, QString)", key, str(value)
            )

        # 응답 버퍼 초기화
        self._response_data = []
        self._prev_next = 0
        self._tr_done = False

        # 요청 전송
        screen_no = self._next_screen_no()
        rq_name = tr_info["name"]
        ret = self._ocx.dynamicCall(
            "CommRqData(QString, QString, int, QString)",
            rq_name,
            tr_code,
            prev_next,
            screen_no,
        )

        if ret != 0:
            raise ConnectionError(
                f"CommRqData failed for {tr_code} (return code={ret})"
            )

        # processEvents 폴링으로 응답 대기
        deadline = time.time() + self._timeout_s
        while not self._tr_done and time.time() < deadline:
            QCoreApplication.processEvents()
            time.sleep(_POLL_INTERVAL_S)

        self._last_request_time = time.time()
        self._hourly_request_times.append(self._last_request_time)

        if not self._response_data:
            raise TimeoutError(
                f"No response for {tr_code} within {self._timeout_s}s"
            )

        # single=True이면 단일 Dict, 아니면 List[Dict]
        is_single = tr_info.get("single", False)
        if is_single and len(self._response_data) == 1:
            return self._response_data[0]
        return self._response_data

    # ── 이벤트 핸들러 ─────────────────────────────────────

    def _on_event_connect(self, err_code: int) -> None:
        """OnEventConnect 이벤트 핸들러."""
        self._login_result = err_code
        self._login_done = True

    def _on_receive_tr_data(
        self,
        screen_no: str,
        rq_name: str,
        tr_code: str,
        record_name: str,
        prev_next: str,
    ) -> None:
        """OnReceiveTrData 이벤트 핸들러."""
        self._prev_next = int(prev_next) if prev_next else 0

        tr_info = TR_CODES.get(tr_code, {})
        output_fields = tr_info.get("output", [])
        is_single = tr_info.get("single", False)

        if is_single:
            row = {}
            for field in output_fields:
                val = self._ocx.dynamicCall(
                    "GetCommData(QString, QString, int, QString)",
                    tr_code,
                    rq_name,
                    0,
                    field,
                )
                row[field] = val.strip()
            self._response_data = [row]
        elif tr_info.get("data_ex", False):
            # GetCommDataEx: 인덱스 기반으로 전체 데이터를 한번에 가져옴
            # GetCommData 필드명이 통하지 않는 TR에 사용
            raw = self._ocx.dynamicCall(
                "GetCommDataEx(QString, QString)", tr_code, rq_name
            )
            if raw:
                for raw_row in raw:
                    row = {}
                    for idx, field in enumerate(output_fields):
                        if idx < len(raw_row):
                            row[field] = str(raw_row[idx]).strip()
                        else:
                            row[field] = ""
                    self._response_data.append(row)
        else:
            # GetCommData: 필드명 기반으로 개별 조회
            count = self._ocx.dynamicCall(
                "GetRepeatCnt(QString, QString)", tr_code, rq_name
            )
            for i in range(count):
                row = {}
                for field in output_fields:
                    val = self._ocx.dynamicCall(
                        "GetCommData(QString, QString, int, QString)",
                        tr_code,
                        rq_name,
                        i,
                        field,
                    )
                    row[field] = val.strip()
                self._response_data.append(row)

        self._tr_done = True
