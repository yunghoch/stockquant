import datetime
from typing import Dict, List

from sqlalchemy import and_
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from lasps.db.models.market_sentiment import MarketSentiment
from lasps.db.repositories.base_repository import BaseRepository

SENTIMENT_COLUMNS = [
    "volume_ratio", "volatility_ratio", "gap_direction",
    "rsi_norm", "foreign_inst_flow",
]


class SentimentRepository(BaseRepository[MarketSentiment]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, MarketSentiment)

    def get_range(
        self,
        stock_code: str,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> List[MarketSentiment]:
        return (
            self.session.query(MarketSentiment)
            .filter(
                and_(
                    MarketSentiment.stock_code == stock_code,
                    MarketSentiment.date >= start_date,
                    MarketSentiment.date <= end_date,
                )
            )
            .order_by(MarketSentiment.date)
            .all()
        )

    def upsert(self, stock_code: str, date: datetime.date, values: Dict[str, float]) -> None:
        record = {"stock_code": stock_code, "date": date}
        for col in SENTIMENT_COLUMNS:
            if col in values:
                record[col] = values[col]
        stmt = sqlite_insert(MarketSentiment).values(**record)
        update_cols = {k: getattr(stmt.excluded, k) for k in record if k not in ("stock_code", "date")}
        stmt = stmt.on_conflict_do_update(
            index_elements=["stock_code", "date"],
            set_=update_cols,
        )
        self.session.execute(stmt)
