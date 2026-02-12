from lasps.utils.constants import (
    TOTAL_FEATURE_DIM, NUM_SECTORS, NUM_CLASSES,
    ALL_FEATURES, TIME_SERIES_LENGTH,
)


def test_feature_dim_consistency():
    assert TOTAL_FEATURE_DIM == 28  # v2: OHLCV(5) + indicators(15) + sentiment(5) + temporal(3)
    assert len(ALL_FEATURES) == TOTAL_FEATURE_DIM


def test_constants():
    assert NUM_SECTORS == 13  # v3: 20개 → 13개 병합
    assert NUM_CLASSES == 3
    assert TIME_SERIES_LENGTH == 60


from lasps.config.sector_config import (
    SECTOR_NAMES, NUM_SECTORS as CONFIG_NUM_SECTORS,
    DEFAULT_SECTOR_ID, get_sector_name,
)


def test_sector_names_count():
    assert len(SECTOR_NAMES) == 13


def test_sector_ids_valid():
    """섹터 ID가 0~12 범위인지 확인."""
    assert set(SECTOR_NAMES.keys()) == set(range(13))


def test_default_sector_id():
    """기본 섹터 ID가 '기타'(12)인지 확인."""
    assert DEFAULT_SECTOR_ID == 12
    assert 0 <= DEFAULT_SECTOR_ID < 13


def test_get_sector_name():
    assert get_sector_name(0) == "전기전자"
    assert get_sector_name(11) == "제조업"
    assert get_sector_name(12) == "기타"
    assert get_sector_name(99) == "Unknown"


def test_num_sectors_consistency():
    """constants와 sector_config의 NUM_SECTORS 일치 확인."""
    assert NUM_SECTORS == CONFIG_NUM_SECTORS
