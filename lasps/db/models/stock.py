from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    ForeignKey,
    Numeric,
    SmallInteger,
    String,
)
from sqlalchemy.orm import relationship

from lasps.db.base import Base, TimestampMixin


class Stock(TimestampMixin, Base):
    """종목 마스터."""

    __tablename__ = "stocks"

    code = Column(String(6), primary_key=True)
    name = Column(String(50), nullable=False)
    sector_id = Column(SmallInteger, ForeignKey("sectors.id"), nullable=True)
    sector_code = Column(String(3), nullable=True)
    market_cap = Column(BigInteger, nullable=True)
    per = Column(Numeric(8, 2), nullable=True)
    pbr = Column(Numeric(8, 3), nullable=True)
    roe = Column(Numeric(8, 2), nullable=True)
    debt_ratio = Column(Numeric(8, 2), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    # pykrx 원본 섹터 정보 (매핑 검증용)
    pykrx_sector_idx = Column(String(10), nullable=True)
    pykrx_sector_name = Column(String(30), nullable=True)

    sector = relationship("Sector", back_populates="stocks")
    daily_prices = relationship("DailyPrice", back_populates="stock")
    investor_tradings = relationship("InvestorTrading", back_populates="stock")
    short_sellings = relationship("ShortSelling", back_populates="stock")
    technical_indicators = relationship("TechnicalIndicator", back_populates="stock")
    market_sentiments = relationship("MarketSentiment", back_populates="stock")
    qvm_scores = relationship("QvmScore", back_populates="stock")
    predictions = relationship("Prediction", back_populates="stock")
    llm_analyses = relationship("LlmAnalysis", back_populates="stock")
    training_labels = relationship("TrainingLabel", back_populates="stock")

    def __repr__(self) -> str:
        return f"<Stock(code='{self.code}', name='{self.name}')>"
