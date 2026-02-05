from typing import List, Optional

from sqlalchemy.orm import Session

from lasps.db.models.stock import Stock
from lasps.db.repositories.base_repository import BaseRepository


class StockRepository(BaseRepository[Stock]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, Stock)

    def get_by_code(self, code: str) -> Optional[Stock]:
        return self.session.query(Stock).filter(Stock.code == code).first()

    def get_by_sector(self, sector_id: int) -> List[Stock]:
        return (
            self.session.query(Stock)
            .filter(Stock.sector_id == sector_id, Stock.is_active.is_(True))
            .all()
        )

    def get_active(self) -> List[Stock]:
        return (
            self.session.query(Stock)
            .filter(Stock.is_active.is_(True))
            .all()
        )

    def upsert(self, code: str, **kwargs: object) -> Stock:
        stock = self.get_by_code(code)
        if stock is None:
            stock = Stock(code=code, **kwargs)
            self.session.add(stock)
        else:
            for k, v in kwargs.items():
                setattr(stock, k, v)
        self.session.flush()
        return stock
