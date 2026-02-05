"""Repository CRUD + 도메인 쿼리 테스트 (SQLite in-memory)."""
import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from lasps.db.base import Base
import pandas as pd

from lasps.db.models import (
    BatchLog,
    DailyPrice,
    ModelCheckpoint,
    Prediction,
    Sector,
    Stock,
    TrainingLabel,
)
from lasps.db.repositories.base_repository import BaseRepository
from lasps.db.repositories.batch_log_repo import BatchLogRepository
from lasps.db.repositories.checkpoint_repo import CheckpointRepository
from lasps.db.repositories.indicator_repo import IndicatorRepository
from lasps.db.repositories.investor_repo import InvestorRepository
from lasps.db.repositories.prediction_repo import PredictionRepository
from lasps.db.repositories.price_repo import PriceRepository
from lasps.db.repositories.sentiment_repo import SentimentRepository
from lasps.db.repositories.short_selling_repo import ShortSellingRepository
from lasps.db.repositories.stock_repo import StockRepository
from lasps.db.repositories.training_label_repo import TrainingLabelRepository


@pytest.fixture(scope="module")
def engine():
    e = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(e)
    return e


@pytest.fixture()
def session(engine):
    _session = sessionmaker(bind=engine)()
    yield _session
    _session.rollback()
    _session.close()


@pytest.fixture()
def seed_data(session: Session):
    """공통 seed: sector + stock 2개."""
    s0 = Sector(id=0, code="001", name="전기전자")
    s1 = Sector(id=1, code="002", name="금융업")
    session.add_all([s0, s1])
    session.flush()

    st1 = Stock(code="005930", name="삼성전자", sector_id=0, is_active=True)
    st2 = Stock(code="000660", name="SK하이닉스", sector_id=0, is_active=True)
    st3 = Stock(code="105560", name="KB금융", sector_id=1, is_active=False)
    session.add_all([st1, st2, st3])
    session.flush()
    return {"sectors": [s0, s1], "stocks": [st1, st2, st3]}


class TestBaseRepository:
    def test_get_all(self, session: Session, seed_data) -> None:
        repo = BaseRepository(session, Sector)
        result = repo.get_all()
        assert len(result) == 2

    def test_get_by_id(self, session: Session, seed_data) -> None:
        repo = BaseRepository(session, Sector)
        s = repo.get_by_id(0)
        assert s is not None
        assert s.name == "전기전자"

    def test_create_and_delete(self, session: Session) -> None:
        repo = BaseRepository(session, Sector)
        s = Sector(id=99, code="099", name="테스트")
        repo.create(s)
        assert repo.get_by_id(99) is not None
        repo.delete(s)
        assert repo.get_by_id(99) is None


class TestStockRepository:
    def test_get_by_code(self, session: Session, seed_data) -> None:
        repo = StockRepository(session)
        stock = repo.get_by_code("005930")
        assert stock is not None
        assert stock.name == "삼성전자"

    def test_get_by_sector(self, session: Session, seed_data) -> None:
        repo = StockRepository(session)
        stocks = repo.get_by_sector(0)
        codes = [s.code for s in stocks]
        assert "005930" in codes
        assert "000660" in codes

    def test_get_active_excludes_inactive(self, session: Session, seed_data) -> None:
        repo = StockRepository(session)
        active = repo.get_active()
        codes = [s.code for s in active]
        assert "105560" not in codes

    def test_upsert_new(self, session: Session, seed_data) -> None:
        repo = StockRepository(session)
        stock = repo.upsert("035720", name="카카오", sector_id=0)
        assert stock.code == "035720"
        assert stock.name == "카카오"

    def test_upsert_existing(self, session: Session, seed_data) -> None:
        repo = StockRepository(session)
        repo.upsert("005930", name="삼성전자(수정)", market_cap=500_000_000_000_000)
        updated = repo.get_by_code("005930")
        assert updated is not None
        assert updated.name == "삼성전자(수정)"
        assert updated.market_cap == 500_000_000_000_000


class TestBatchLogRepository:
    def test_start_complete_flow(self, session: Session) -> None:
        repo = BatchLogRepository(session)
        log = repo.start(datetime.date(2024, 2, 1), model_version="v7a")
        assert log.status == "running"
        assert log.id is not None

        repo.complete(log, stocks_predicted=50)
        assert log.status == "completed"
        assert log.stocks_predicted == 50

    def test_fail_flow(self, session: Session) -> None:
        repo = BatchLogRepository(session)
        log = repo.start(datetime.date(2024, 2, 2))
        repo.fail(log, "timeout error")
        assert log.status == "failed"
        assert log.error_message == "timeout error"

    def test_get_by_date(self, session: Session) -> None:
        repo = BatchLogRepository(session)
        repo.start(datetime.date(2024, 3, 1))
        found = repo.get_by_date(datetime.date(2024, 3, 1))
        assert found is not None
        assert found.status == "running"


class TestCheckpointRepository:
    def test_activate(self, session: Session) -> None:
        repo = CheckpointRepository(session)
        mc1 = ModelCheckpoint(
            version="v1", phase=1, file_path="v1.pt", is_active=True
        )
        mc2 = ModelCheckpoint(
            version="v2", phase=3, file_path="v2.pt", is_active=False
        )
        session.add_all([mc1, mc2])
        session.flush()

        repo.activate("v2")
        assert mc1.is_active is False
        assert mc2.is_active is True

    def test_get_active(self, session: Session) -> None:
        repo = CheckpointRepository(session)
        mc = ModelCheckpoint(
            version="v3", phase=3, file_path="v3.pt", is_active=True
        )
        session.add(mc)
        session.flush()
        active = repo.get_active()
        assert active is not None
        assert active.version == "v3"


class TestPredictionRepository:
    def test_get_buy_signals(self, session: Session, seed_data) -> None:
        repo = PredictionRepository(session)

        bl = BatchLog(
            date=datetime.date(2024, 4, 1),
            status="completed",
            started_at=datetime.datetime(2024, 4, 1, 9, 0),
        )
        session.add(bl)
        session.flush()

        p1 = Prediction(
            stock_code="005930", date=datetime.date(2024, 4, 1),
            prediction=2, label="BUY", confidence=0.9,
            model_version="v7a", batch_id=bl.id,
        )
        p2 = Prediction(
            stock_code="000660", date=datetime.date(2024, 4, 1),
            prediction=1, label="HOLD", confidence=0.5,
            model_version="v7a", batch_id=bl.id,
        )
        session.add_all([p1, p2])
        session.flush()

        buys = repo.get_buy_signals(datetime.date(2024, 4, 1), model_version="v7a")
        assert len(buys) == 1
        assert buys[0].stock_code == "005930"


class TestTrainingLabelRepository:
    def test_get_by_split(self, session: Session, seed_data) -> None:
        repo = TrainingLabelRepository(session)

        tl1 = TrainingLabel(
            stock_code="005930", date=datetime.date(2022, 1, 1),
            label=1, split="train", sector_id=0,
        )
        tl2 = TrainingLabel(
            stock_code="005930", date=datetime.date(2023, 1, 1),
            label=2, split="val", sector_id=0,
        )
        session.add_all([tl1, tl2])
        session.flush()

        train = repo.get_by_split("train")
        assert len(train) >= 1
        assert all(t.split == "train" for t in train)

    def test_get_by_sector_and_split(self, session: Session, seed_data) -> None:
        repo = TrainingLabelRepository(session)
        results = repo.get_by_sector_and_split(0, "train")
        assert all(t.sector_id == 0 for t in results)


class TestPriceRepositoryUpsert:
    def test_upsert_from_dataframe(self, session: Session, seed_data) -> None:
        repo = PriceRepository(session)
        df = pd.DataFrame([
            {"date": datetime.date(2024, 1, 2), "open": 70000, "high": 71000,
             "low": 69000, "close": 70500, "volume": 10_000_000},
        ])
        count = repo.upsert_from_dataframe("005930", df)
        assert count == 1

        # upsert 동일 날짜 → 업데이트
        df2 = pd.DataFrame([
            {"date": datetime.date(2024, 1, 2), "open": 70000, "high": 72000,
             "low": 69000, "close": 71000, "volume": 12_000_000},
        ])
        repo.upsert_from_dataframe("005930", df2)
        prices = repo.get_latest_n_days("005930", 5)
        assert len(prices) == 1
        assert prices[0].close == 71000

    def test_get_latest_date(self, session: Session, seed_data) -> None:
        repo = PriceRepository(session)
        assert repo.get_latest_date("000660") is None

    def test_to_dataframe_empty(self, session: Session) -> None:
        repo = PriceRepository(session)
        df = repo.to_dataframe([])
        assert len(df) == 0


class TestInvestorRepositoryUpsert:
    def test_upsert_insert_and_update(self, session: Session, seed_data) -> None:
        repo = InvestorRepository(session)
        repo.upsert("005930", datetime.date(2024, 1, 2), 500_000, -200_000)
        results = repo.get_range(
            "005930", datetime.date(2024, 1, 1), datetime.date(2024, 1, 3)
        )
        assert len(results) == 1
        assert results[0].foreign_net == 500_000

        # upsert 업데이트
        repo.upsert("005930", datetime.date(2024, 1, 2), 600_000, -300_000)
        results = repo.get_range(
            "005930", datetime.date(2024, 1, 1), datetime.date(2024, 1, 3)
        )
        assert len(results) == 1
        assert results[0].foreign_net == 600_000


class TestShortSellingRepositoryUpsert:
    def test_upsert(self, session: Session, seed_data) -> None:
        repo = ShortSellingRepository(session)
        repo.upsert("005930", datetime.date(2024, 1, 2), 100_000, 2.5)
        results = repo.get_range(
            "005930", datetime.date(2024, 1, 1), datetime.date(2024, 1, 3)
        )
        assert len(results) == 1
        assert results[0].short_volume == 100_000


class TestIndicatorRepositoryUpsert:
    def test_upsert(self, session: Session, seed_data) -> None:
        repo = IndicatorRepository(session)
        repo.upsert("005930", datetime.date(2024, 1, 2), {"ma5": 70200.0, "rsi": 55.3})
        results = repo.get_range(
            "005930", datetime.date(2024, 1, 1), datetime.date(2024, 1, 3)
        )
        assert len(results) == 1
        assert results[0].rsi == pytest.approx(55.3)

        # upsert 업데이트
        repo.upsert("005930", datetime.date(2024, 1, 2), {"ma5": 70500.0, "rsi": 60.0})
        results = repo.get_range(
            "005930", datetime.date(2024, 1, 1), datetime.date(2024, 1, 3)
        )
        assert len(results) == 1
        assert results[0].ma5 == pytest.approx(70500.0)


class TestSentimentRepositoryUpsert:
    def test_upsert(self, session: Session, seed_data) -> None:
        repo = SentimentRepository(session)
        repo.upsert("005930", datetime.date(2024, 1, 2), {
            "volume_ratio": 0.65, "rsi_norm": 0.55,
        })
        results = repo.get_range(
            "005930", datetime.date(2024, 1, 1), datetime.date(2024, 1, 3)
        )
        assert len(results) == 1
        assert results[0].volume_ratio == pytest.approx(0.65, abs=0.01)
