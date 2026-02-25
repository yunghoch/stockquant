"""키움 OpenAPI 매수/매도 주문 스크립트.

지정가·시장가 매수/매도를 실행하고, 미체결 주문을 장 마감까지 관리한다.
KiwoomRealAPI의 COM 연결/이벤트 패턴을 재사용한다.

Usage:
    # 지정가 매수
    python scripts/kiwoom_order.py buy 005380 10 250000

    # 시장가 매수
    python scripts/kiwoom_order.py buy 005380 10 --market

    # 지정가 매도
    python scripts/kiwoom_order.py sell 005380 10 260000

    # 시장가 매도
    python scripts/kiwoom_order.py sell 005380 10 --market

    # 미체결 조회
    python scripts/kiwoom_order.py status

    # 미체결 전체 취소
    python scripts/kiwoom_order.py cancel-all
"""
import argparse
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger

# ── 프로젝트 경로 설정 ─────────────────────────────────────
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from lasps.config.settings import settings
from lasps.data.collectors.kiwoom_real import KiwoomRealAPI

try:
    from PyQt5.QtCore import QCoreApplication
except ImportError:
    QCoreApplication = None  # type: ignore[assignment,misc]


# ── 상수 ───────────────────────────────────────────────────
_ORDER_SCREEN = "0201"
_CANCEL_SCREEN = "0202"
_POLL_INTERVAL_S = 0.05
_CHECK_INTERVAL_S = 30.0
_MARKET_CLOSE_HOUR = 15
_MARKET_CLOSE_MINUTE = 20  # 15:20에 미체결 자동 취소 (15:30 장 마감 전 여유)
_MARKET_OPEN_HOUR = 9
_MARKET_OPEN_MINUTE = 0
_MARKET_END_HOUR = 15
_MARKET_END_MINUTE = 30
_DEFAULT_MAX_AMOUNT = 100_000_000  # 1억원
_CONFIRM_WAIT_S = 3


# ── 주문유형 / 거래구분 ────────────────────────────────────
ORDER_TYPE_BUY = 1
ORDER_TYPE_SELL = 2
ORDER_TYPE_CANCEL_BUY = 3
ORDER_TYPE_CANCEL_SELL = 4

HOGA_LIMIT = "00"   # 지정가
HOGA_MARKET = "03"  # 시장가


class KiwoomTrader:
    """키움 OpenAPI 주문 실행기.

    KiwoomRealAPI를 통해 키움에 접속하고, SendOrder로 매수/매도/취소를 수행한다.
    OnReceiveChejanData 이벤트로 체결·잔고 변경을 실시간 추적한다.

    Attributes:
        api: KiwoomRealAPI 인스턴스.
        account: 키움 계좌번호.
        orders: 주문번호별 상태 추적 딕셔너리.
    """

    def __init__(self) -> None:
        self.api = KiwoomRealAPI()
        self.account: str = ""
        self.orders: Dict[str, Dict] = {}

        if not self.api.connect():
            raise ConnectionError("키움 API 로그인 실패")

        # OnReceiveChejanData 이벤트 연결
        self.api._ocx.OnReceiveChejanData.connect(self._on_chejan_data)

        # 계좌번호 조회
        self.account = self.get_account()
        logger.info(f"계좌번호: {self.account}")

    def get_account(self) -> str:
        """GetLoginInfo로 첫 번째 계좌번호를 조회한다.

        Returns:
            계좌번호 문자열.

        Raises:
            RuntimeError: 계좌번호 조회 실패.
        """
        # settings에 계좌번호가 있으면 우선 사용
        if settings.KIWOOM_ACCOUNT:
            return settings.KIWOOM_ACCOUNT

        raw = self.api._ocx.dynamicCall(
            'GetLoginInfo(QString)', "ACCNO"
        )
        if not raw:
            raise RuntimeError("계좌번호 조회 실패")
        accounts = [a.strip() for a in raw.split(";") if a.strip()]
        if not accounts:
            raise RuntimeError("계좌번호가 없습니다")
        return accounts[0]

    def _on_chejan_data(self, s_gubun: str, n_item_cnt: int, s_fid_list: str) -> None:
        """OnReceiveChejanData 이벤트 핸들러.

        Args:
            s_gubun: "0"=주문체결, "1"=잔고변경.
            n_item_cnt: 아이템 개수.
            s_fid_list: FID 리스트.
        """
        if s_gubun == "0":
            order_no = self._get_chejan("9203").strip()
            code = self._get_chejan("9001").strip().replace("A", "")
            name = self._get_chejan("302").strip()
            order_qty = self._get_chejan("900").strip()
            order_price = self._get_chejan("901").strip()
            unfilled_qty = self._get_chejan("902").strip()
            status = self._get_chejan("913").strip()

            logger.info(
                f"[체결] 주문번호={order_no} 종목={name}({code}) "
                f"주문수량={order_qty} 주문가격={order_price} "
                f"미체결={unfilled_qty} 상태={status}"
            )

            if order_no:
                self.orders[order_no] = {
                    "code": code,
                    "name": name,
                    "order_qty": int(order_qty) if order_qty else 0,
                    "order_price": int(order_price) if order_price else 0,
                    "unfilled_qty": int(unfilled_qty) if unfilled_qty else 0,
                    "status": status,
                }

        elif s_gubun == "1":
            code = self._get_chejan("9001").strip().replace("A", "")
            name = self._get_chejan("302").strip()
            holding_qty = self._get_chejan("930").strip()
            avg_price = self._get_chejan("931").strip()

            logger.info(
                f"[잔고] 종목={name}({code}) "
                f"보유수량={holding_qty} 평균가={avg_price}"
            )

    def _get_chejan(self, fid: str) -> str:
        """GetChejanData로 체결 데이터를 조회한다.

        Args:
            fid: FID 번호.

        Returns:
            FID에 해당하는 값.
        """
        return self.api._ocx.dynamicCall(
            "GetChejanData(int)", int(fid)
        )

    def _send_order(
        self,
        order_name: str,
        order_type: int,
        code: str,
        qty: int,
        price: int,
        hoga_type: str,
        original_order_no: str = "",
    ) -> str:
        """SendOrder를 호출하여 주문을 전송한다.

        Args:
            order_name: 주문명 (로그용).
            order_type: 1=신규매수, 2=신규매도, 3=매수취소, 4=매도취소.
            code: 종목코드 (6자리).
            qty: 주문수량.
            price: 주문가격 (시장가는 0).
            hoga_type: "00"=지정가, "03"=시장가.
            original_order_no: 취소/정정 시 원주문번호.

        Returns:
            SendOrder 리턴값 (0=성공).

        Raises:
            RuntimeError: 주문 전송 실패.
        """
        screen = _CANCEL_SCREEN if order_type in (3, 4) else _ORDER_SCREEN

        ret = self.api._ocx.dynamicCall(
            "SendOrder(QString, QString, QString, int, QString, int, int, QString, QString)",
            order_name,
            screen,
            self.account,
            order_type,
            code,
            qty,
            price,
            hoga_type,
            original_order_no,
        )

        if ret != 0:
            raise RuntimeError(
                f"SendOrder 실패: ret={ret}, order_type={order_type}, "
                f"code={code}, qty={qty}, price={price}"
            )

        # 이벤트 처리를 위해 잠시 대기
        deadline = time.time() + 5.0
        while time.time() < deadline:
            QCoreApplication.processEvents()
            time.sleep(_POLL_INTERVAL_S)
            # 주문번호가 수신되었으면 종료
            if self.orders:
                break

        logger.info(
            f"주문 전송 성공: {order_name} "
            f"(type={order_type}, code={code}, qty={qty}, price={price})"
        )
        return str(ret)

    def buy_limit(self, code: str, qty: int, price: int) -> str:
        """지정가 매수.

        Args:
            code: 종목코드.
            qty: 매수 수량.
            price: 지정가.

        Returns:
            주문번호.
        """
        logger.info(f"지정가 매수: {code} {qty}주 @ {price:,}원")
        self._send_order("지정가매수", ORDER_TYPE_BUY, code, qty, price, HOGA_LIMIT)
        return self._last_order_no()

    def sell_limit(self, code: str, qty: int, price: int) -> str:
        """지정가 매도.

        Args:
            code: 종목코드.
            qty: 매도 수량.
            price: 지정가.

        Returns:
            주문번호.
        """
        logger.info(f"지정가 매도: {code} {qty}주 @ {price:,}원")
        self._send_order("지정가매도", ORDER_TYPE_SELL, code, qty, price, HOGA_LIMIT)
        return self._last_order_no()

    def buy_market(self, code: str, qty: int) -> str:
        """시장가 매수.

        Args:
            code: 종목코드.
            qty: 매수 수량.

        Returns:
            주문번호.
        """
        logger.info(f"시장가 매수: {code} {qty}주")
        self._send_order("시장가매수", ORDER_TYPE_BUY, code, qty, 0, HOGA_MARKET)
        return self._last_order_no()

    def sell_market(self, code: str, qty: int) -> str:
        """시장가 매도.

        Args:
            code: 종목코드.
            qty: 매도 수량.

        Returns:
            주문번호.
        """
        logger.info(f"시장가 매도: {code} {qty}주")
        self._send_order("시장가매도", ORDER_TYPE_SELL, code, qty, 0, HOGA_MARKET)
        return self._last_order_no()

    def _last_order_no(self) -> str:
        """가장 최근에 수신된 주문번호를 반환한다.

        Returns:
            주문번호 문자열. 없으면 빈 문자열.
        """
        if self.orders:
            return list(self.orders.keys())[-1]
        return ""

    def cancel_order(self, original_order_no: str, code: str, qty: int) -> None:
        """주문 취소.

        원주문의 매수/매도 구분을 orders에서 조회하여 적절한 취소 유형을 사용한다.

        Args:
            original_order_no: 원주문번호.
            code: 종목코드.
            qty: 취소 수량.
        """
        order_info = self.orders.get(original_order_no, {})
        order_gubun = order_info.get("status", "")

        # 주문구분에 "매수"가 포함되면 매수취소, 아니면 매도취소
        if "매도" in order_gubun:
            cancel_type = ORDER_TYPE_CANCEL_SELL
        else:
            cancel_type = ORDER_TYPE_CANCEL_BUY

        logger.info(
            f"주문 취소: 원주문={original_order_no} 종목={code} 수량={qty}"
        )
        self._send_order(
            "주문취소", cancel_type, code, qty, 0, HOGA_LIMIT, original_order_no
        )

    def get_unfilled_orders(self) -> List[Dict]:
        """미체결 주문 목록을 조회한다.

        OPT10075 TR을 사용하여 현재 미체결 주문을 조회한다.

        Returns:
            미체결 주문 리스트. 각 항목은 종목코드, 주문번호, 종목명 등을 포함.
        """
        try:
            result = self.api.request(
                "OPT10075",
                계좌번호=self.account,
                전체종목구분="0",
                매매구분="0",
                종목코드="",
                체결구분="1",  # 미체결
            )
        except TimeoutError:
            logger.warning("미체결 조회 타임아웃")
            return []

        if isinstance(result, dict):
            result = [result]

        # 미체결수량이 0인 항목 제외
        unfilled = []
        for row in result:
            unfilled_qty = row.get("미체결수량", "0").strip()
            if unfilled_qty and int(unfilled_qty) > 0:
                unfilled.append(row)

        return unfilled

    def cancel_all_unfilled(self) -> int:
        """모든 미체결 주문을 취소한다.

        Returns:
            취소 요청한 건수.
        """
        unfilled = self.get_unfilled_orders()
        if not unfilled:
            logger.info("미체결 주문 없음")
            return 0

        cancel_count = 0
        for order in unfilled:
            order_no = order.get("주문번호", "").strip()
            code = order.get("종목코드", "").strip().replace("A", "")
            qty_str = order.get("미체결수량", "0").strip()
            qty = int(qty_str) if qty_str else 0
            gubun = order.get("주문구분", "").strip()

            if not order_no or qty <= 0:
                continue

            if "매도" in gubun:
                cancel_type = ORDER_TYPE_CANCEL_SELL
            else:
                cancel_type = ORDER_TYPE_CANCEL_BUY

            try:
                self._send_order(
                    "미체결취소", cancel_type, code, qty, 0, HOGA_LIMIT, order_no
                )
                cancel_count += 1
                logger.info(
                    f"미체결 취소: 주문번호={order_no} "
                    f"종목={code} 수량={qty} 구분={gubun}"
                )
            except RuntimeError as e:
                logger.error(f"취소 실패: {order_no} - {e}")

        logger.info(f"미체결 취소 완료: {cancel_count}건")
        return cancel_count

    def wait_until_filled_or_close(
        self, order_no: str, code: str, qty: int
    ) -> Dict:
        """장 마감(15:20)까지 체결을 대기한다.

        30초 간격으로 미체결 수량을 확인하고, 15:20 이후 미체결분을 자동 취소한다.

        Args:
            order_no: 주문번호.
            code: 종목코드.
            qty: 주문 수량.

        Returns:
            {"filled_qty": int, "unfilled_qty": int, "cancelled": bool}
        """
        logger.info(
            f"체결 대기 시작: 주문번호={order_no} 종목={code} "
            f"수량={qty} (15:20까지 대기)"
        )

        cancelled = False

        while True:
            now = datetime.now()

            # 15:20 이후면 미체결 취소
            if (now.hour > _MARKET_CLOSE_HOUR or
                    (now.hour == _MARKET_CLOSE_HOUR and
                     now.minute >= _MARKET_CLOSE_MINUTE)):
                logger.info("장 마감 임박 (15:20) - 미체결 주문 취소 시작")
                order_info = self.orders.get(order_no, {})
                unfilled = order_info.get("unfilled_qty", qty)

                if unfilled > 0:
                    try:
                        self.cancel_order(order_no, code, unfilled)
                        cancelled = True
                    except RuntimeError as e:
                        logger.error(f"자동 취소 실패: {e}")
                break

            # 체결 상태 확인
            order_info = self.orders.get(order_no, {})
            unfilled = order_info.get("unfilled_qty", qty)

            if unfilled <= 0:
                logger.info(f"전량 체결 완료: 주문번호={order_no}")
                break

            logger.debug(
                f"미체결 대기 중: 주문번호={order_no} 미체결={unfilled}주 "
                f"현재시각={now.strftime('%H:%M:%S')}"
            )

            # 30초 대기 (processEvents 계속 호출)
            wait_deadline = time.time() + _CHECK_INTERVAL_S
            while time.time() < wait_deadline:
                QCoreApplication.processEvents()
                time.sleep(_POLL_INTERVAL_S)

        order_info = self.orders.get(order_no, {})
        unfilled_qty = order_info.get("unfilled_qty", 0)
        filled_qty = qty - unfilled_qty

        result = {
            "filled_qty": filled_qty,
            "unfilled_qty": unfilled_qty,
            "cancelled": cancelled,
        }
        logger.info(
            f"주문 결과: 체결={filled_qty}주, 미체결={unfilled_qty}주, "
            f"취소={cancelled}"
        )
        return result


def _is_market_hours() -> bool:
    """현재 시각이 장 시간(09:00~15:30) 내인지 확인한다."""
    now = datetime.now()
    market_open = now.replace(
        hour=_MARKET_OPEN_HOUR, minute=_MARKET_OPEN_MINUTE, second=0
    )
    market_close = now.replace(
        hour=_MARKET_END_HOUR, minute=_MARKET_END_MINUTE, second=0
    )
    return market_open <= now <= market_close


def _get_stock_name(api: KiwoomRealAPI, code: str) -> str:
    """종목코드로 종목명을 조회한다."""
    try:
        result = api.request("OPT10001", 종목코드=code)
        if isinstance(result, dict):
            return result.get("종목명", code).strip()
    except Exception:
        pass
    return code


def _confirm_order(
    action: str,
    code: str,
    name: str,
    qty: int,
    price: int,
    is_market: bool,
) -> bool:
    """주문 전 확인을 출력하고 3초 대기한다.

    Returns:
        True면 주문 진행, False면 취소.
    """
    price_str = "시장가" if is_market else f"{price:,}원"
    total = "시장가" if is_market else f"{price * qty:,}원"

    print("\n" + "=" * 50)
    print(f"  주문 확인")
    print("=" * 50)
    print(f"  유형: {action}")
    print(f"  종목: {name} ({code})")
    print(f"  수량: {qty}주")
    print(f"  가격: {price_str}")
    print(f"  총액: {total}")
    print("=" * 50)
    print(f"  {_CONFIRM_WAIT_S}초 후 주문이 실행됩니다... (Ctrl+C로 취소)")
    print()

    try:
        time.sleep(_CONFIRM_WAIT_S)
        return True
    except KeyboardInterrupt:
        print("\n주문이 취소되었습니다.")
        return False


def main() -> None:
    """CLI 엔트리포인트."""
    parser = argparse.ArgumentParser(
        description="키움 OpenAPI 매수/매도 주문",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s buy 005380 10 250000      지정가 매수 (현대차 10주 @ 250,000원)
  %(prog)s buy 005380 10 --market    시장가 매수
  %(prog)s sell 005380 10 260000     지정가 매도
  %(prog)s sell 005380 10 --market   시장가 매도
  %(prog)s status                    미체결 조회
  %(prog)s cancel-all                미체결 전체 취소
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="주문 명령")

    # buy 명령
    buy_parser = subparsers.add_parser("buy", help="매수 주문")
    buy_parser.add_argument("code", help="종목코드 (예: 005380)")
    buy_parser.add_argument("qty", type=int, help="주문 수량")
    buy_parser.add_argument("price", type=int, nargs="?", default=0, help="지정가 (시장가면 생략)")
    buy_parser.add_argument("--market", action="store_true", help="시장가 주문")
    buy_parser.add_argument("--no-confirm", action="store_true", help="확인 없이 즉시 주문")
    buy_parser.add_argument("--no-wait", action="store_true", help="체결 대기 없이 종료")
    buy_parser.add_argument("--force", action="store_true", help="장 시간 외 주문 허용")
    buy_parser.add_argument("--max-amount", type=int, default=_DEFAULT_MAX_AMOUNT, help="최대 주문 금액 (기본: 1억)")

    # sell 명령
    sell_parser = subparsers.add_parser("sell", help="매도 주문")
    sell_parser.add_argument("code", help="종목코드 (예: 005380)")
    sell_parser.add_argument("qty", type=int, help="주문 수량")
    sell_parser.add_argument("price", type=int, nargs="?", default=0, help="지정가 (시장가면 생략)")
    sell_parser.add_argument("--market", action="store_true", help="시장가 주문")
    sell_parser.add_argument("--no-confirm", action="store_true", help="확인 없이 즉시 주문")
    sell_parser.add_argument("--no-wait", action="store_true", help="체결 대기 없이 종료")
    sell_parser.add_argument("--force", action="store_true", help="장 시간 외 주문 허용")
    sell_parser.add_argument("--max-amount", type=int, default=_DEFAULT_MAX_AMOUNT, help="최대 주문 금액 (기본: 1억)")

    # status 명령
    subparsers.add_parser("status", help="미체결 주문 조회")

    # cancel-all 명령
    subparsers.add_parser("cancel-all", help="미체결 전체 취소")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # ── status / cancel-all ──────────────────────────────
    if args.command == "status":
        trader = KiwoomTrader()
        unfilled = trader.get_unfilled_orders()
        if not unfilled:
            print("미체결 주문이 없습니다.")
            return

        print(f"\n미체결 주문: {len(unfilled)}건")
        print("-" * 70)
        for order in unfilled:
            print(
                f"  주문번호={order.get('주문번호', '').strip():>10s}  "
                f"종목={order.get('종목명', '').strip():<10s}  "
                f"수량={order.get('주문수량', '').strip():>6s}  "
                f"가격={order.get('주문가격', '').strip():>10s}  "
                f"미체결={order.get('미체결수량', '').strip():>6s}  "
                f"구분={order.get('주문구분', '').strip()}"
            )
        print("-" * 70)
        return

    if args.command == "cancel-all":
        trader = KiwoomTrader()
        count = trader.cancel_all_unfilled()
        print(f"미체결 {count}건 취소 요청 완료")
        return

    # ── buy / sell ───────────────────────────────────────
    is_market = args.market
    price = 0 if is_market else args.price

    if not is_market and price <= 0:
        print("오류: 지정가 주문은 가격을 지정해야 합니다. (시장가: --market)")
        sys.exit(1)

    # 장 시간 체크
    if not args.force and not _is_market_hours():
        print(
            f"경고: 현재 장 시간(09:00~15:30)이 아닙니다. "
            f"(현재: {datetime.now().strftime('%H:%M')})"
        )
        print("장 시간 외 주문을 하려면 --force 옵션을 사용하세요.")
        sys.exit(1)

    # 최대 금액 체크 (시장가가 아닌 경우)
    if not is_market:
        total_amount = price * args.qty
        if total_amount > args.max_amount:
            print(
                f"오류: 주문 총액({total_amount:,}원)이 "
                f"최대 허용 금액({args.max_amount:,}원)을 초과합니다."
            )
            print("--max-amount 옵션으로 한도를 조정할 수 있습니다.")
            sys.exit(1)

    trader = KiwoomTrader()

    # 종목명 조회
    stock_name = _get_stock_name(trader.api, args.code)

    # 주문 확인
    if not args.no_confirm:
        action = "매수" if args.command == "buy" else "매도"
        if not _confirm_order(action, args.code, stock_name, args.qty, price, is_market):
            sys.exit(0)

    # 주문 실행
    if args.command == "buy":
        if is_market:
            order_no = trader.buy_market(args.code, args.qty)
        else:
            order_no = trader.buy_limit(args.code, args.qty, price)
    else:
        if is_market:
            order_no = trader.sell_market(args.code, args.qty)
        else:
            order_no = trader.sell_limit(args.code, args.qty, price)

    if not order_no:
        print("주문이 전송되었으나 주문번호를 수신하지 못했습니다.")
        print("키움 HTS에서 주문 상태를 확인하세요.")
        return

    print(f"\n주문 전송 완료: 주문번호={order_no}")

    # 시장가 주문이거나 --no-wait이면 대기 없이 종료
    if is_market or args.no_wait:
        print("주문 대기 없이 종료합니다.")
        return

    # 지정가 주문: 장 마감까지 체결 대기
    print("체결 대기 중... (15:20까지 대기, Ctrl+C로 중단)")
    try:
        result = trader.wait_until_filled_or_close(order_no, args.code, args.qty)
    except KeyboardInterrupt:
        print("\n대기 중단. 미체결 주문은 키움 HTS에서 확인하세요.")
        return

    print(f"\n{'=' * 40}")
    print(f"  주문 결과")
    print(f"{'=' * 40}")
    print(f"  체결 수량: {result['filled_qty']}주")
    print(f"  미체결 수량: {result['unfilled_qty']}주")
    print(f"  자동 취소: {'예' if result['cancelled'] else '아니오'}")
    print(f"{'=' * 40}")


if __name__ == "__main__":
    main()
