import datetime
from typing import List, Optional

from sqlalchemy import and_
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from lasps.db.models.short_selling import ShortSelling
from lasps.db.repositories.base_repository import BaseRepository


class ShortSellingRepository(BaseRepository[ShortSelling]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, ShortSelling)

    def get_range(
        self,
        stock_code: str,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> List[ShortSelling]:
        return (
            self.session.query(ShortSelling)
            .filter(
                and_(
                    ShortSelling.stock_code == stock_code,
                    ShortSelling.date >= start_date,
                    ShortSelling.date <= end_date,
                )
            )
            .order_by(ShortSelling.date)
            .all()
        )

    def upsert(
        self,
        stock_code: str,
        date: datetime.date,
        short_volume: int,
        short_ratio: Optional[float] = None,
    ) -> None:
        stmt = sqlite_insert(ShortSelling).values(
            stock_code=stock_code,
            date=date,
            short_volume=short_volume,
            short_ratio=short_ratio,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["stock_code", "date"],
            set_={
                "short_volume": stmt.excluded.short_volume,
                "short_ratio": stmt.excluded.short_ratio,
            },
        )
        self.session.execute(stmt)
