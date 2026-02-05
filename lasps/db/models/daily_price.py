from sqlalchemy import BigInteger, Column, Date, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import relationship

from lasps.db.base import Base, BigIntPK


class DailyPrice(Base):
    """일봉 OHLCV."""

    __tablename__ = "daily_prices"
    __table_args__ = (
        UniqueConstraint("stock_code", "date", name="uq_daily_prices_stock_date"),
    )

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    stock_code = Column(String(6), ForeignKey("stocks.code"), nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Integer, nullable=False)
    high = Column(Integer, nullable=False)
    low = Column(Integer, nullable=False)
    close = Column(Integer, nullable=False)
    volume = Column(BigInteger, nullable=False)

    stock = relationship("Stock", back_populates="daily_prices")

    def __repr__(self) -> str:
        return f"<DailyPrice({self.stock_code} {self.date} C={self.close})>"
