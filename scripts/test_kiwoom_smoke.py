"""키움 OpenAPI+ 연결 스모크 테스트.

pytest 없이 단독 실행 가능. 키움 로그인 → 삼성전자 기본정보 + 일봉 60일 조회.

Usage:
    python scripts/test_kiwoom_smoke.py
"""
import sys
import os

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    print("=" * 60)
    print("Kiwoom OpenAPI+ Smoke Test")
    print("=" * 60)

    # 환경 확인
    if sys.platform != "win32":
        print("[FAIL] Windows required")
        sys.exit(1)

    try:
        from lasps.data.collectors.kiwoom_real import KiwoomRealAPI
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        sys.exit(1)

    # 1. 로그인
    print("\n[1/4] Connecting to Kiwoom API...")
    api = KiwoomRealAPI(timeout_ms=15_000)
    if not api.connect():
        print("[FAIL] Login failed")
        sys.exit(1)
    print("[PASS] Connected")

    # 2. 연결 상태 확인
    print("\n[2/4] Checking connection status...")
    assert api.is_connected(), "is_connected() returned False"
    print("[PASS] is_connected() = True")

    # 3. 삼성전자 기본정보
    print("\n[3/4] Requesting OPT10001 (삼성전자 005930)...")
    info = api.request("OPT10001", 종목코드="005930")
    print(f"  종목명: {info.get('종목명', 'N/A')}")
    print(f"  업종코드: {info.get('업종코드', 'N/A')}")
    print(f"  업종명: {info.get('업종명', 'N/A')}")
    print(f"  시가총액: {info.get('시가총액', 'N/A')}")
    print(f"  PER: {info.get('PER', 'N/A')}")
    assert "종목명" in info, "Missing 종목명"
    assert "업종코드" in info, "Missing 업종코드"
    print("[PASS] Stock info retrieved")

    # 4. 일봉 데이터 (60일)
    print("\n[4/4] Requesting OPT10081 (일봉 차트)...")
    ohlcv = api.request("OPT10081", 종목코드="005930")
    row_count = len(ohlcv) if isinstance(ohlcv, list) else 1
    print(f"  Rows received: {row_count}")
    if isinstance(ohlcv, list) and len(ohlcv) > 0:
        latest = ohlcv[0]
        print(f"  Latest: 일자={latest.get('일자')}, "
              f"현재가={latest.get('현재가')}, "
              f"거래량={latest.get('거래량')}")
    assert row_count >= 1, "Empty OHLCV response"
    print("[PASS] OHLCV data retrieved")

    # 정리
    api.disconnect()

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
