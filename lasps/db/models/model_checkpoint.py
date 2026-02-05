from sqlalchemy import Boolean, Column, SmallInteger, String
from sqlalchemy.types import REAL

from lasps.db.base import Base, BigIntPK, TimestampMixin


class ModelCheckpoint(TimestampMixin, Base):
    """모델 메타데이터."""

    __tablename__ = "model_checkpoints"

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    version = Column(String(30), unique=True, nullable=False)
    phase = Column(SmallInteger, nullable=False)
    file_path = Column(String(255), nullable=False)
    val_loss = Column(REAL, nullable=True)
    accuracy = Column(REAL, nullable=True)
    is_active = Column(Boolean, default=False, nullable=False)

    def __repr__(self) -> str:
        return f"<ModelCheckpoint({self.version} phase={self.phase} active={self.is_active})>"
