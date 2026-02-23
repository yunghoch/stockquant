import datetime
from typing import List, Optional

from sqlalchemy import and_
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from lasps.db.models.training_label import TrainingLabel
from lasps.db.repositories.base_repository import BaseRepository


class TrainingLabelRepository(BaseRepository[TrainingLabel]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, TrainingLabel)

    def get_by_split(self, split: str) -> List[TrainingLabel]:
        return (
            self.session.query(TrainingLabel)
            .filter(TrainingLabel.split == split)
            .order_by(TrainingLabel.date)
            .all()
        )

    def get_by_sector_and_split(self, sector_id: int, split: str) -> List[TrainingLabel]:
        return (
            self.session.query(TrainingLabel)
            .filter(
                and_(
                    TrainingLabel.sector_id == sector_id,
                    TrainingLabel.split == split,
                )
            )
            .order_by(TrainingLabel.date)
            .all()
        )

    def get_by_stock_range(
        self,
        stock_code: str,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> List[TrainingLabel]:
        return (
            self.session.query(TrainingLabel)
            .filter(
                and_(
                    TrainingLabel.stock_code == stock_code,
                    TrainingLabel.date >= start_date,
                    TrainingLabel.date <= end_date,
                )
            )
            .order_by(TrainingLabel.date)
            .all()
        )

    def upsert(
        self,
        stock_code: str,
        date: datetime.date,
        label: int,
        split: str,
        sector_id: Optional[int] = None,
    ) -> None:
        stmt = sqlite_insert(TrainingLabel).values(
            stock_code=stock_code,
            date=date,
            label=label,
            split=split,
            sector_id=sector_id,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["stock_code", "date"],
            set_={
                "label": stmt.excluded.label,
                "split": stmt.excluded.split,
                "sector_id": stmt.excluded.sector_id,
            },
        )
        self.session.execute(stmt)
