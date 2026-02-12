"""historical_data.py 3-Phase 파이프라인 통합 테스트.

in-memory SQLite + Mock API로 전체 파이프라인을 검증한다.
"""

import datetime

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from lasps.data.collectors.kiwoom_collector import KiwoomCollector
from lasps.data.collectors.kiwoom_mock import MOCK_STOCKS, KiwoomMockAPI
from lasps.db.base import Base
from lasps.db.models import Sector
from lasps.db.repositories import (
    IndicatorRepository,
    InvestorRepository,
    PriceRepository,
    SentimentRepository,
    ShortSellingRepository,
    StockRepository,
    TrainingLabelRepository,
)
from scripts.historical_data import (
    SPLIT_CONFIG,
    collect_raw_data,
    compute_derived_data,
    export_npy,
    get_split,
    seed_sectors,
)

# 300일: 2015-01-02 ~ 약 2016-03 (train split만 포함)
SMALL_DAYS = 300
# 2600일: 2015-01-02 ~ 약 2025-03 (전체 split 포함)
FULL_DAYS = 2600

MOCK_STOCK_CODES = list(MOCK_STOCKS.keys())


def _make_collector() -> KiwoomCollector:
    """테스트용 Mock collector를 생성한다."""
    api = KiwoomMockAPI(seed=42)
    return KiwoomCollector(api, rate_limit=False)


@pytest.fixture()
def session():
    """In-memory SQLite session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    _session = sessionmaker(bind=engine)()
    yield _session
    _session.close()


@pytest.fixture()
def small_pipeline(session):
    """Phase 1+2를 300일로 실행한 session."""
    collector = _make_collector()
    collect_raw_data(session, collector, MOCK_STOCK_CODES, days=SMALL_DAYS, resume=False)
    compute_derived_data(session)
    return session


class TestGetSplit:
    def test_train_date(self) -> None:
        assert get_split(datetime.date(2020, 6, 15)) == "train"

    def test_val_date(self) -> None:
        assert get_split(datetime.date(2023, 6, 15)) == "val"

    def test_test_date(self) -> None:
        assert get_split(datetime.date(2024, 6, 15)) == "test"

    def test_out_of_range(self) -> None:
        assert get_split(datetime.date(2025, 6, 15)) == ""


class TestSeedSectors:
    def test_seeds_13_sectors(self, session) -> None:
        seed_sectors(session)
        count = session.query(Sector).count()
        assert count == 13

    def test_idempotent(self, session) -> None:
        seed_sectors(session)
        seed_sectors(session)
        count = session.query(Sector).count()
        assert count == 13


class TestPhase1CollectRawData:
    def test_stocks_created(self, session) -> None:
        collector = _make_collector()
        collect_raw_data(session, collector, MOCK_STOCK_CODES, days=SMALL_DAYS, resume=False)
        repo = StockRepository(session)
        stocks = repo.get_active()
        assert len(stocks) == 5

    def test_prices_count(self, session) -> None:
        collector = _make_collector()
        collect_raw_data(session, collector, MOCK_STOCK_CODES, days=SMALL_DAYS, resume=False)
        repo = PriceRepository(session)
        prices = repo.get_range(
            "005930", datetime.date(2015, 1, 1), datetime.date(2030, 1, 1),
        )
        assert len(prices) == SMALL_DAYS

    def test_investor_data_stored(self, session) -> None:
        collector = _make_collector()
        collect_raw_data(session, collector, MOCK_STOCK_CODES, days=SMALL_DAYS, resume=False)
        repo = InvestorRepository(session)
        records = repo.get_range(
            "005930", datetime.date(2015, 1, 1), datetime.date(2030, 1, 1),
        )
        assert len(records) == SMALL_DAYS

    def test_short_selling_stored(self, session) -> None:
        collector = _make_collector()
        collect_raw_data(session, collector, MOCK_STOCK_CODES, days=SMALL_DAYS, resume=False)
        repo = ShortSellingRepository(session)
        records = repo.get_range(
            "005930", datetime.date(2015, 1, 1), datetime.date(2030, 1, 1),
        )
        assert len(records) == SMALL_DAYS

    def test_sectors_seeded(self, session) -> None:
        collector = _make_collector()
        collect_raw_data(session, collector, MOCK_STOCK_CODES, days=SMALL_DAYS, resume=False)
        count = session.query(Sector).count()
        assert count == 13

    def test_resume_skips_collected(self, session) -> None:
        """resume=True 시 이미 수집된 종목을 스킵하는지 확인."""
        collector = _make_collector()
        # 첫 번째 수집
        collect_raw_data(session, collector, MOCK_STOCK_CODES, days=SMALL_DAYS, resume=False)

        price_repo = PriceRepository(session)
        first_count = len(price_repo.get_range(
            "005930", datetime.date(2015, 1, 1), datetime.date(2030, 1, 1),
        ))

        # resume 수집 (동일 데이터이므로 스킵되어야 함)
        collector2 = _make_collector()
        collect_raw_data(session, collector2, MOCK_STOCK_CODES, days=SMALL_DAYS, resume=True)

        second_count = len(price_repo.get_range(
            "005930", datetime.date(2015, 1, 1), datetime.date(2030, 1, 1),
        ))
        assert second_count == first_count


class TestPhase2ComputeDerived:
    def test_indicators_computed(self, small_pipeline) -> None:
        repo = IndicatorRepository(small_pipeline)
        records = repo.get_range(
            "005930", datetime.date(2015, 1, 1), datetime.date(2030, 1, 1),
        )
        assert len(records) > 0
        # MA5는 5일 이후부터 유효
        non_null_ma5 = [r for r in records if r.ma5 is not None]
        assert len(non_null_ma5) > SMALL_DAYS - 10

    def test_sentiment_computed(self, small_pipeline) -> None:
        repo = SentimentRepository(small_pipeline)
        records = repo.get_range(
            "005930", datetime.date(2015, 1, 1), datetime.date(2030, 1, 1),
        )
        assert len(records) > 0

    def test_labels_computed(self, small_pipeline) -> None:
        repo = TrainingLabelRepository(small_pipeline)
        labels = repo.get_by_split("train")
        assert len(labels) > 0

    def test_label_values_valid(self, small_pipeline) -> None:
        repo = TrainingLabelRepository(small_pipeline)
        labels = repo.get_by_split("train")
        label_values = {lbl.label for lbl in labels}
        # 최소 2종류 이상의 라벨 (300일 mock 데이터에서)
        assert len(label_values) >= 2
        assert label_values.issubset({0, 1, 2})

    def test_labels_have_sector_id(self, small_pipeline) -> None:
        repo = TrainingLabelRepository(small_pipeline)
        labels = repo.get_by_split("train")
        with_sector = [lbl for lbl in labels if lbl.sector_id is not None]
        assert len(with_sector) == len(labels)


class TestPhase3ExportNpy:
    def test_train_npy_created(self, small_pipeline, tmp_path) -> None:
        export_npy(small_pipeline, tmp_path, stride=20, no_charts=True)

        assert (tmp_path / "train" / "time_series.npy").exists()
        assert (tmp_path / "train" / "sector_ids.npy").exists()
        assert (tmp_path / "train" / "labels.npy").exists()

    def test_time_series_shape(self, small_pipeline, tmp_path) -> None:
        export_npy(small_pipeline, tmp_path, stride=20, no_charts=True)

        ts = np.load(str(tmp_path / "train" / "time_series.npy"))
        assert ts.ndim == 3
        assert ts.shape[1] == 60   # TIME_SERIES_LENGTH
        assert ts.shape[2] == 28   # TOTAL_FEATURE_DIM
        assert ts.dtype == np.float32

    def test_labels_shape_matches(self, small_pipeline, tmp_path) -> None:
        export_npy(small_pipeline, tmp_path, stride=20, no_charts=True)

        ts = np.load(str(tmp_path / "train" / "time_series.npy"))
        labels = np.load(str(tmp_path / "train" / "labels.npy"))
        sectors = np.load(str(tmp_path / "train" / "sector_ids.npy"))

        assert labels.shape[0] == ts.shape[0]
        assert sectors.shape[0] == ts.shape[0]

    def test_label_values(self, small_pipeline, tmp_path) -> None:
        export_npy(small_pipeline, tmp_path, stride=20, no_charts=True)

        labels = np.load(str(tmp_path / "train" / "labels.npy"))
        assert labels.dtype == np.int64
        assert set(np.unique(labels)).issubset({0, 1, 2})

    def test_no_charts_skips_images(self, small_pipeline, tmp_path) -> None:
        export_npy(small_pipeline, tmp_path, stride=20, no_charts=True)
        assert not (tmp_path / "train" / "chart_images.npy").exists()

    def test_normalized_values(self, small_pipeline, tmp_path) -> None:
        export_npy(small_pipeline, tmp_path, stride=20, no_charts=True)

        ts = np.load(str(tmp_path / "train" / "time_series.npy"))
        # OHLCV + indicators (cols 0-19) should be normalized to [0, 1]
        ohlcv_ind = ts[:, :, :20]
        assert ohlcv_ind.min() >= -0.01  # small tolerance
        assert ohlcv_ind.max() <= 1.01


class TestFullPipeline:
    """2600일 전체 파이프라인 테스트 (all splits)."""

    @pytest.fixture(scope="class")
    def full_session(self):
        """Class-scoped: Phase 1+2를 2600일로 한 번만 실행."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        _session = sessionmaker(bind=engine)()
        collector = _make_collector()
        collect_raw_data(_session, collector, MOCK_STOCK_CODES, days=FULL_DAYS, resume=False)
        compute_derived_data(_session)
        yield _session
        _session.close()

    def test_all_splits_have_labels(self, full_session) -> None:
        repo = TrainingLabelRepository(full_session)
        for split in ["train", "val", "test"]:
            labels = repo.get_by_split(split)
            assert len(labels) > 0, f"No labels for {split}"

    def test_split_time_ordering(self, full_session) -> None:
        repo = TrainingLabelRepository(full_session)
        train_dates = [lbl.date for lbl in repo.get_by_split("train")]
        val_dates = [lbl.date for lbl in repo.get_by_split("val")]
        test_dates = [lbl.date for lbl in repo.get_by_split("test")]

        assert max(train_dates) < min(val_dates)
        assert max(val_dates) < min(test_dates)

    def test_all_splits_exported(self, full_session, tmp_path) -> None:
        export_npy(full_session, tmp_path, stride=20, no_charts=True)

        for split in ["train", "val", "test"]:
            assert (tmp_path / split / "time_series.npy").exists(), \
                f"Missing {split}/time_series.npy"
            assert (tmp_path / split / "labels.npy").exists(), \
                f"Missing {split}/labels.npy"

            ts = np.load(str(tmp_path / split / "time_series.npy"))
            labels = np.load(str(tmp_path / split / "labels.npy"))
            assert ts.shape[0] == labels.shape[0]
            assert ts.shape[1:] == (60, 28)

    def test_label_distribution_all_classes(self, full_session) -> None:
        repo = TrainingLabelRepository(full_session)
        # train split은 8년치 데이터로 3가지 라벨 모두 포함
        labels = repo.get_by_split("train")
        label_values = {lbl.label for lbl in labels}
        assert label_values == {0, 1, 2}, f"Missing labels: {label_values}"
