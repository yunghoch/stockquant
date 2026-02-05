from sqlalchemy import Column, Date, ForeignKey, String, UniqueConstraint
from sqlalchemy.types import REAL
from sqlalchemy.orm import relationship

from lasps.db.base import Base, BigIntPK


class MarketSentiment(Base):
    """시장감성 5차원."""

    __tablename__ = "market_sentiment"
    __table_args__ = (
        UniqueConstraint("stock_code", "date", name="uq_market_sentiment_stock_date"),
    )

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    stock_code = Column(String(6), ForeignKey("stocks.code"), nullable=False)
    date = Column(Date, nullable=False)
    volume_ratio = Column(REAL, nullable=True)
    volatility_ratio = Column(REAL, nullable=True)
    gap_direction = Column(REAL, nullable=True)
    rsi_norm = Column(REAL, nullable=True)
    foreign_inst_flow = Column(REAL, nullable=True)

    stock = relationship("Stock", back_populates="market_sentiments")

    def __repr__(self) -> str:
        return f"<MarketSentiment({self.stock_code} {self.date})>"
