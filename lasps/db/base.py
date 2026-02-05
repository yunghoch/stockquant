from datetime import datetime

from sqlalchemy import BigInteger, Column, DateTime, Integer
from sqlalchemy.orm import DeclarativeBase

# SQLite INTEGER 자동증가 (BigInteger는 PostgreSQL 전환 시 BIGSERIAL로 매핑)
BigIntPK = BigInteger().with_variant(Integer, "sqlite")


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    """created_at / updated_at 자동 관리 Mixin."""

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
