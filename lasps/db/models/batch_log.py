from sqlalchemy import Column, Date, DateTime, Integer, String, Text
from sqlalchemy.orm import relationship

from lasps.db.base import Base, BigIntPK


class BatchLog(Base):
    """배치 실행 로그."""

    __tablename__ = "batch_logs"

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    date = Column(Date, unique=True, nullable=False)
    status = Column(String(20), nullable=False, default="running")
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    stocks_predicted = Column(Integer, nullable=True)
    model_version = Column(String(30), nullable=True)
    error_message = Column(Text, nullable=True)

    predictions = relationship("Prediction", back_populates="batch_log")
    llm_analyses = relationship("LlmAnalysis", back_populates="batch_log")

    def __repr__(self) -> str:
        return f"<BatchLog({self.date} {self.status})>"
