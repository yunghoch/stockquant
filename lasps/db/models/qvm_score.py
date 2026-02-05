from sqlalchemy import Boolean, Column, Date, ForeignKey, SmallInteger, String, UniqueConstraint
from sqlalchemy.types import REAL
from sqlalchemy.orm import relationship

from lasps.db.base import Base, BigIntPK


class QvmScore(Base):
    """QVM 스크리닝 결과."""

    __tablename__ = "qvm_scores"
    __table_args__ = (
        UniqueConstraint("stock_code", "date", name="uq_qvm_scores_stock_date"),
    )

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    stock_code = Column(String(6), ForeignKey("stocks.code"), nullable=False)
    date = Column(Date, nullable=False)
    q_score = Column(REAL, nullable=True)
    v_score = Column(REAL, nullable=True)
    m_score = Column(REAL, nullable=True)
    qvm_score = Column(REAL, nullable=True)
    rank = Column(SmallInteger, nullable=True)
    selected = Column(Boolean, nullable=True)

    stock = relationship("Stock", back_populates="qvm_scores")

    def __repr__(self) -> str:
        return f"<QvmScore({self.stock_code} {self.date} qvm={self.qvm_score})>"
