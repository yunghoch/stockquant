from sqlalchemy import BigInteger, Column, Date, ForeignKey, SmallInteger, String, UniqueConstraint
from sqlalchemy.types import REAL
from sqlalchemy.orm import relationship

from lasps.db.base import Base, BigIntPK, TimestampMixin


class Prediction(TimestampMixin, Base):
    """예측 결과."""

    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint(
            "stock_code", "date", "model_version",
            name="uq_predictions_stock_date_version",
        ),
    )

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    stock_code = Column(String(6), ForeignKey("stocks.code"), nullable=False)
    date = Column(Date, nullable=False)
    prediction = Column(SmallInteger, nullable=False)
    label = Column(String(4), nullable=False)
    confidence = Column(REAL, nullable=True)
    prob_sell = Column(REAL, nullable=True)
    prob_hold = Column(REAL, nullable=True)
    prob_buy = Column(REAL, nullable=True)
    model_version = Column(String(30), nullable=False)
    sector_id = Column(SmallInteger, ForeignKey("sectors.id"), nullable=True)
    batch_id = Column(BigInteger, ForeignKey("batch_logs.id"), nullable=True)

    stock = relationship("Stock", back_populates="predictions")
    sector = relationship("Sector")
    batch_log = relationship("BatchLog", back_populates="predictions")
    llm_analysis = relationship("LlmAnalysis", back_populates="prediction", uselist=False)

    def __repr__(self) -> str:
        return f"<Prediction({self.stock_code} {self.date} {self.label})>"
