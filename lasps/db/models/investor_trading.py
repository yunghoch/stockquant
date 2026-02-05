from sqlalchemy import BigInteger, Column, Date, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import relationship

from lasps.db.base import Base, BigIntPK


class InvestorTrading(Base):
    """투자자별 매매 (OPT10059)."""

    __tablename__ = "investor_trading"
    __table_args__ = (
        UniqueConstraint("stock_code", "date", name="uq_investor_trading_stock_date"),
    )

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    stock_code = Column(String(6), ForeignKey("stocks.code"), nullable=False)
    date = Column(Date, nullable=False)
    foreign_net = Column(BigInteger, nullable=False)
    inst_net = Column(BigInteger, nullable=False)
    individual_net = Column(BigInteger, nullable=True)

    stock = relationship("Stock", back_populates="investor_tradings")

    def __repr__(self) -> str:
        return f"<InvestorTrading({self.stock_code} {self.date})>"
