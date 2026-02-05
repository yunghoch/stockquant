from sqlalchemy import Column, Date, Float, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import relationship

from lasps.db.base import Base, BigIntPK


class TechnicalIndicator(Base):
    """기술지표 15개."""

    __tablename__ = "technical_indicators"
    __table_args__ = (
        UniqueConstraint("stock_code", "date", name="uq_technical_indicators_stock_date"),
    )

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    stock_code = Column(String(6), ForeignKey("stocks.code"), nullable=False)
    date = Column(Date, nullable=False)

    # MA (4)
    ma5 = Column(Float, nullable=True)
    ma20 = Column(Float, nullable=True)
    ma60 = Column(Float, nullable=True)
    ma120 = Column(Float, nullable=True)

    # Momentum (4)
    rsi = Column(Float, nullable=True)
    macd = Column(Float, nullable=True)
    macd_signal = Column(Float, nullable=True)
    macd_hist = Column(Float, nullable=True)

    # Bollinger (4) + ATR (1)
    bb_upper = Column(Float, nullable=True)
    bb_middle = Column(Float, nullable=True)
    bb_lower = Column(Float, nullable=True)
    bb_width = Column(Float, nullable=True)
    atr = Column(Float, nullable=True)

    # Volume (2)
    obv = Column(Float, nullable=True)
    volume_ma20 = Column(Float, nullable=True)

    stock = relationship("Stock", back_populates="technical_indicators")

    def __repr__(self) -> str:
        return f"<TechnicalIndicator({self.stock_code} {self.date})>"
