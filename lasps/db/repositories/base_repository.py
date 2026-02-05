from typing import Generic, List, Optional, Type, TypeVar

from sqlalchemy.orm import Session

from lasps.db.base import Base

T = TypeVar("T", bound=Base)


class BaseRepository(Generic[T]):
    """공통 CRUD 연산."""

    def __init__(self, session: Session, model: Type[T]) -> None:
        self.session = session
        self.model = model

    def get_by_id(self, id_value: object) -> Optional[T]:
        return self.session.get(self.model, id_value)

    def get_all(self) -> List[T]:
        return self.session.query(self.model).all()

    def create(self, entity: T) -> T:
        self.session.add(entity)
        self.session.flush()
        return entity

    def bulk_create(self, entities: List[T]) -> List[T]:
        self.session.add_all(entities)
        self.session.flush()
        return entities

    def delete(self, entity: T) -> None:
        self.session.delete(entity)
        self.session.flush()

    def commit(self) -> None:
        self.session.commit()
