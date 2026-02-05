"""ORM 모델 생성/조회 테스트 (SQLite in-memory)."""
import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from lasps.db.base import Base
from lasps.db.models import (
    BatchLog,
    DailyPrice,
    InvestorTrading,
    LlmAnalysis,
    MarketSentiment,
    ModelCheckpoint,
    Prediction,
    QvmScore,
    Sector,
    ShortSelling,
    Stock,
    TechnicalIndicator,
    TrainingLabel,
)


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
def seed_sector(session: Session) -> Sector:
    s = Sector(id=0, code="001", name="전기전자")
    session.add(s)
    session.flush()
    return s


@pytest.fixture()
def seed_stock(session: Session, seed_sector: Sector) -> Stock:
    st = Stock(code="005930", name="삼성전자", sector_id=0, sector_code="001", is_active=True)
    session.add(st)
    session.flush()
    return st


class TestSector:
    def test_create_and_query(self, session: Session) -> None:
        s = Sector(id=1, code="002", name="금융업")
        session.add(s)
        session.flush()
        result = session.get(Sector, 1)
        assert result is not None
        assert result.code == "002"
        assert result.name == "금융업"

    def test_repr(self, seed_sector: Sector) -> None:
        assert "001" in repr(seed_sector)


class TestStock:
    def test_create_with_sector(self, session: Session, seed_stock: Stock) -> None:
        result = session.get(Stock, "005930")
        assert result is not None
        assert result.name == "삼성전자"
        assert result.sector_id == 0
        assert result.is_active is True

    def test_sector_relationship(self, session: Session, seed_stock: Stock) -> None:
        assert seed_stock.sector is not None
        assert seed_stock.sector.name == "전기전자"


class TestDailyPrice:
    def test_create(self, session: Session, seed_stock: Stock) -> None:
        p = DailyPrice(
            stock_code="005930",
            date=datetime.date(2024, 1, 2),
            open=70000,
            high=71000,
            low=69000,
            close=70500,
            volume=10_000_000,
        )
        session.add(p)
        session.flush()
        assert p.id is not None
        assert p.close == 70500


class TestInvestorTrading:
    def test_create(self, session: Session, seed_stock: Stock) -> None:
        it = InvestorTrading(
            stock_code="005930",
            date=datetime.date(2024, 1, 2),
            foreign_net=500_000,
            inst_net=-200_000,
        )
        session.add(it)
        session.flush()
        assert it.id is not None
        assert it.individual_net is None


class TestShortSelling:
    def test_create(self, session: Session, seed_stock: Stock) -> None:
        ss = ShortSelling(
            stock_code="005930",
            date=datetime.date(2024, 1, 2),
            short_volume=100_000,
            short_ratio=2.5,
        )
        session.add(ss)
        session.flush()
        assert ss.id is not None


class TestTechnicalIndicator:
    def test_create_partial(self, session: Session, seed_stock: Stock) -> None:
        ti = TechnicalIndicator(
            stock_code="005930",
            date=datetime.date(2024, 1, 2),
            ma5=70200.0,
            rsi=55.3,
        )
        session.add(ti)
        session.flush()
        assert ti.ma20 is None
        assert ti.rsi == pytest.approx(55.3)


class TestMarketSentiment:
    def test_create(self, session: Session, seed_stock: Stock) -> None:
        ms = MarketSentiment(
            stock_code="005930",
            date=datetime.date(2024, 1, 2),
            volume_ratio=0.65,
            volatility_ratio=0.4,
            gap_direction=0.1,
            rsi_norm=0.55,
            foreign_inst_flow=0.3,
        )
        session.add(ms)
        session.flush()
        assert ms.volume_ratio == pytest.approx(0.65, abs=0.01)


class TestQvmScore:
    def test_create(self, session: Session, seed_stock: Stock) -> None:
        q = QvmScore(
            stock_code="005930",
            date=datetime.date(2024, 1, 2),
            q_score=0.8,
            v_score=0.6,
            m_score=0.7,
            qvm_score=0.7,
            rank=1,
            selected=True,
        )
        session.add(q)
        session.flush()
        assert q.selected is True


class TestBatchLog:
    def test_create(self, session: Session) -> None:
        bl = BatchLog(
            date=datetime.date(2024, 1, 2),
            status="running",
            started_at=datetime.datetime(2024, 1, 2, 9, 0),
        )
        session.add(bl)
        session.flush()
        assert bl.id is not None
        assert bl.completed_at is None


class TestPrediction:
    def test_create(self, session: Session, seed_stock: Stock) -> None:
        bl = BatchLog(
            date=datetime.date(2024, 1, 3),
            status="completed",
            started_at=datetime.datetime(2024, 1, 3, 9, 0),
        )
        session.add(bl)
        session.flush()

        pred = Prediction(
            stock_code="005930",
            date=datetime.date(2024, 1, 3),
            prediction=2,
            label="BUY",
            confidence=0.85,
            prob_sell=0.05,
            prob_hold=0.10,
            prob_buy=0.85,
            model_version="v7a-phase3",
            sector_id=0,
            batch_id=bl.id,
        )
        session.add(pred)
        session.flush()
        assert pred.label == "BUY"
        assert pred.batch_log is not None


class TestLlmAnalysis:
    def test_create(self, session: Session, seed_stock: Stock) -> None:
        la = LlmAnalysis(
            stock_code="005930",
            date=datetime.date(2024, 1, 2),
            analysis_text="삼성전자는 반도체 업황 개선으로 매수 의견.",
            model_name="claude-3-opus",
        )
        session.add(la)
        session.flush()
        assert la.id is not None


class TestModelCheckpoint:
    def test_create(self, session: Session) -> None:
        mc = ModelCheckpoint(
            version="v7a-phase3-20240102",
            phase=3,
            file_path="checkpoints/v7a_phase3.pt",
            val_loss=0.45,
            accuracy=0.62,
            is_active=True,
        )
        session.add(mc)
        session.flush()
        assert mc.is_active is True


class TestTrainingLabel:
    def test_create(self, session: Session, seed_stock: Stock) -> None:
        tl = TrainingLabel(
            stock_code="005930",
            date=datetime.date(2022, 6, 15),
            label=1,
            split="train",
            sector_id=0,
        )
        session.add(tl)
        session.flush()
        assert tl.split == "train"
