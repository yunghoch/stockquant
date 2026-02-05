from datetime import datetime

from sqlalchemy import Column, DateTime, SmallInteger, String
from sqlalchemy.orm import relationship

from lasps.db.base import Base


class Sector(Base):
    """업종 마스터 (20행 고정)."""

    __tablename__ = "sectors"

    id = Column(SmallInteger, primary_key=True, autoincrement=False)
    code = Column(String(3), unique=True, nullable=False)
    name = Column(String(30), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    stocks = relationship("Stock", back_populates="sector")

    def __repr__(self) -> str:
        return f"<Sector(id={self.id}, code='{self.code}', name='{self.name}')>"
