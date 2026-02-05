from sqlalchemy import BigInteger, Column, Date, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from lasps.db.base import Base, BigIntPK, TimestampMixin


class LlmAnalysis(TimestampMixin, Base):
    """LLM 분석 결과."""

    __tablename__ = "llm_analyses"
    __table_args__ = (
        UniqueConstraint("stock_code", "date", name="uq_llm_analyses_stock_date"),
    )

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    stock_code = Column(String(6), ForeignKey("stocks.code"), nullable=False)
    date = Column(Date, nullable=False)
    analysis_text = Column(Text, nullable=False)
    model_name = Column(String(50), nullable=True)
    prediction_id = Column(BigInteger, ForeignKey("predictions.id"), nullable=True)
    batch_id = Column(BigInteger, ForeignKey("batch_logs.id"), nullable=True)

    stock = relationship("Stock", back_populates="llm_analyses")
    prediction = relationship("Prediction", back_populates="llm_analysis")
    batch_log = relationship("BatchLog", back_populates="llm_analyses")

    def __repr__(self) -> str:
        return f"<LlmAnalysis({self.stock_code} {self.date})>"
