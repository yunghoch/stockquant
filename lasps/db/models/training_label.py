from sqlalchemy import Column, Date, ForeignKey, SmallInteger, String, UniqueConstraint
from sqlalchemy.orm import relationship

from lasps.db.base import Base, BigIntPK


class TrainingLabel(Base):
    """학습 데이터 매핑."""

    __tablename__ = "training_labels"
    __table_args__ = (
        UniqueConstraint("stock_code", "date", name="uq_training_labels_stock_date"),
    )

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    stock_code = Column(String(6), ForeignKey("stocks.code"), nullable=False)
    date = Column(Date, nullable=False)
    label = Column(SmallInteger, nullable=False)
    split = Column(String(5), nullable=False)
    sector_id = Column(SmallInteger, ForeignKey("sectors.id"), nullable=True)

    stock = relationship("Stock", back_populates="training_labels")
    sector = relationship("Sector")

    def __repr__(self) -> str:
        return f"<TrainingLabel({self.stock_code} {self.date} label={self.label})>"
