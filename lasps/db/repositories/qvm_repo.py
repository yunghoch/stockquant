import datetime
from typing import List

from sqlalchemy import and_
from sqlalchemy.orm import Session

from lasps.db.models.qvm_score import QvmScore
from lasps.db.repositories.base_repository import BaseRepository


class QvmRepository(BaseRepository[QvmScore]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, QvmScore)

    def get_by_date(self, date: datetime.date) -> List[QvmScore]:
        return (
            self.session.query(QvmScore)
            .filter(QvmScore.date == date)
            .order_by(QvmScore.rank)
            .all()
        )

    def get_selected(self, date: datetime.date) -> List[QvmScore]:
        return (
            self.session.query(QvmScore)
            .filter(and_(QvmScore.date == date, QvmScore.selected.is_(True)))
            .order_by(QvmScore.rank)
            .all()
        )
