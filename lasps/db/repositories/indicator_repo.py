import datetime
from typing import Dict, List

from sqlalchemy import and_
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from lasps.db.models.technical_indicator import TechnicalIndicator
from lasps.db.repositories.base_repository import BaseRepository

INDICATOR_COLUMNS = [
    "ma5", "ma20", "ma60", "ma120",
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_middle", "bb_lower", "bb_width", "atr",
    "obv", "volume_ma20",
]


class IndicatorRepository(BaseRepository[TechnicalIndicator]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, TechnicalIndicator)

    def get_range(
        self,
        stock_code: str,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> List[TechnicalIndicator]:
        return (
            self.session.query(TechnicalIndicator)
            .filter(
                and_(
                    TechnicalIndicator.stock_code == stock_code,
                    TechnicalIndicator.date >= start_date,
                    TechnicalIndicator.date <= end_date,
                )
            )
            .order_by(TechnicalIndicator.date)
            .all()
        )

    def upsert(self, stock_code: str, date: datetime.date, values: Dict[str, float]) -> None:
        record = {"stock_code": stock_code, "date": date}
        for col in INDICATOR_COLUMNS:
            if col in values:
                record[col] = values[col]
        stmt = sqlite_insert(TechnicalIndicator).values(**record)
        update_cols = {k: getattr(stmt.excluded, k) for k in record if k not in ("stock_code", "date")}
        stmt = stmt.on_conflict_do_update(
            index_elements=["stock_code", "date"],
            set_=update_cols,
        )
        self.session.execute(stmt)
        self.session.expire_all()
