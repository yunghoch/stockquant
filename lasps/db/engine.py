from typing import Generator, Optional

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from lasps.config.settings import settings

_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def get_engine() -> Engine:
    """SQLite 엔진을 지연 생성."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.DATABASE_URL,
            connect_args={"check_same_thread": False},
        )
    return _engine


def get_session_factory() -> sessionmaker:
    """SessionLocal 팩토리를 지연 생성."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(), autocommit=False, autoflush=False
        )
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """FastAPI Depends용 세션 제공자."""
    factory = get_session_factory()
    db = factory()
    try:
        yield db
    finally:
        db.close()
