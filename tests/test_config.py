from lasps.utils.constants import (
    TOTAL_FEATURE_DIM, NUM_SECTORS, NUM_CLASSES,
    ALL_FEATURES, TIME_SERIES_LENGTH,
)


def test_feature_dim_consistency():
    assert TOTAL_FEATURE_DIM == 25
    assert len(ALL_FEATURES) == TOTAL_FEATURE_DIM


def test_constants():
    assert NUM_SECTORS == 20
    assert NUM_CLASSES == 3
    assert TIME_SERIES_LENGTH == 60


from lasps.config.sector_config import (
    SECTOR_CODES, get_sector_id, get_sector_name,
)


def test_sector_codes_count():
    assert len(SECTOR_CODES) == 20


def test_sector_ids_unique():
    ids = [v[0] for v in SECTOR_CODES.values()]
    assert len(set(ids)) == 20
    assert set(ids) == set(range(20))


def test_get_sector_id():
    assert get_sector_id("001") == 0
    assert get_sector_id("020") == 19
    assert get_sector_id("999") == -1


def test_get_sector_name():
    assert get_sector_name(0) == "전기전자"
    assert get_sector_name(19) == "광업"
    assert get_sector_name(99) == "Unknown"
