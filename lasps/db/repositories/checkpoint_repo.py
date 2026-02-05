from typing import Optional

from sqlalchemy.orm import Session

from lasps.db.models.model_checkpoint import ModelCheckpoint
from lasps.db.repositories.base_repository import BaseRepository


class CheckpointRepository(BaseRepository[ModelCheckpoint]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, ModelCheckpoint)

    def get_active(self) -> Optional[ModelCheckpoint]:
        return (
            self.session.query(ModelCheckpoint)
            .filter(ModelCheckpoint.is_active.is_(True))
            .first()
        )

    def get_by_version(self, version: str) -> Optional[ModelCheckpoint]:
        return (
            self.session.query(ModelCheckpoint)
            .filter(ModelCheckpoint.version == version)
            .first()
        )

    def activate(self, version: str) -> Optional[ModelCheckpoint]:
        """지정 버전을 활성화하고 나머지를 비활성화."""
        self.session.query(ModelCheckpoint).filter(
            ModelCheckpoint.is_active.is_(True)
        ).update({"is_active": False})
        cp = self.get_by_version(version)
        if cp:
            cp.is_active = True
            self.session.flush()
        return cp
