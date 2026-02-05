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
