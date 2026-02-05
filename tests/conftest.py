"""pytest 공통 설정 및 픽스처.

kiwoom 마커: 실제 키움 OpenAPI+ 연결이 필요한 테스트에 사용.
Windows + PyQt5 + 키움 로그인 상태에서만 실행되며, 그 외 환경에서는 자동 skip.
"""
import sys

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """커스텀 마커 등록."""
    config.addinivalue_line(
        "markers", "kiwoom: live Kiwoom OpenAPI tests (require Windows + login)"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list
) -> None:
    """kiwoom 마커가 붙은 테스트를 비-Windows 환경에서 자동 skip."""
    if sys.platform == "win32":
        return

    skip_kiwoom = pytest.mark.skip(reason="Kiwoom API requires Windows")
    for item in items:
        if "kiwoom" in item.keywords:
            item.add_marker(skip_kiwoom)


@pytest.fixture(scope="session")
def real_api():
    """Session-scoped 실제 키움 API 픽스처.

    키움 로그인을 1회 수행하고 전체 테스트 세션에서 공유한다.
    PyQt5가 없거나 로그인 실패 시 자동 skip.
    """
    if sys.platform != "win32":
        pytest.skip("Kiwoom API requires Windows")

    try:
        from lasps.data.collectors.kiwoom_real import KiwoomRealAPI
    except ImportError:
        pytest.skip("PyQt5 not installed")

    api = KiwoomRealAPI(timeout_s=15.0)
    if not api.connect():
        pytest.skip("Kiwoom login failed or timed out")

    yield api

    api.disconnect()


@pytest.fixture(scope="session")
def real_collector(real_api):
    """Session-scoped 실제 API 기반 KiwoomCollector.

    rate_limit=True로 TR 간격을 준수한다.
    """
    from lasps.data.collectors.kiwoom_collector import KiwoomCollector

    return KiwoomCollector(real_api, rate_limit=True)
