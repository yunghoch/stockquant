from sqlalchemy import BigInteger, Column, Date, ForeignKey, Numeric, String, UniqueConstraint
from sqlalchemy.orm import relationship

from lasps.db.base import Base, BigIntPK


class ShortSelling(Base):
    """공매도 (OPT10014)."""

    __tablename__ = "short_selling"
    __table_args__ = (
        UniqueConstraint("stock_code", "date", name="uq_short_selling_stock_date"),
    )

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    stock_code = Column(String(6), ForeignKey("stocks.code"), nullable=False)
    date = Column(Date, nullable=False)
    short_volume = Column(BigInteger, nullable=False)
    short_ratio = Column(Numeric(6, 3), nullable=True)

    stock = relationship("Stock", back_populates="short_sellings")

    def __repr__(self) -> str:
        return f"<ShortSelling({self.stock_code} {self.date})>"
