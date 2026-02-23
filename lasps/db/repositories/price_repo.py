import datetime
from typing import List, Optional

import pandas as pd
from sqlalchemy import and_
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from lasps.db.models.daily_price import DailyPrice
from lasps.db.repositories.base_repository import BaseRepository


class PriceRepository(BaseRepository[DailyPrice]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, DailyPrice)

    def get_range(
        self,
        stock_code: str,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> List[DailyPrice]:
        return (
            self.session.query(DailyPrice)
            .filter(
                and_(
                    DailyPrice.stock_code == stock_code,
                    DailyPrice.date >= start_date,
                    DailyPrice.date <= end_date,
                )
            )
            .order_by(DailyPrice.date)
            .all()
        )

    def get_latest_n_days(self, stock_code: str, n: int) -> List[DailyPrice]:
        return (
            self.session.query(DailyPrice)
            .filter(DailyPrice.stock_code == stock_code)
            .order_by(DailyPrice.date.desc())
            .limit(n)
            .all()
        )[::-1]

    def get_latest_date(self, stock_code: str) -> Optional[datetime.date]:
        row = (
            self.session.query(DailyPrice.date)
            .filter(DailyPrice.stock_code == stock_code)
            .order_by(DailyPrice.date.desc())
            .first()
        )
        return row[0] if row else None

    def upsert_from_dataframe(self, stock_code: str, df: pd.DataFrame) -> int:
        """DataFrame(date,open,high,low,close,volume) → upsert. 반환: 행 수."""
        if df.empty:
            return 0
        records = []
        for _, row in df.iterrows():
            records.append(
                {
                    "stock_code": stock_code,
                    "date": row["date"],
                    "open": int(row["open"]),
                    "high": int(row["high"]),
                    "low": int(row["low"]),
                    "close": int(row["close"]),
                    "volume": int(row["volume"]),
                }
            )
        stmt = sqlite_insert(DailyPrice).values(records)
        stmt = stmt.on_conflict_do_update(
            index_elements=["stock_code", "date"],
            set_={
                "open": stmt.excluded.open,
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "volume": stmt.excluded.volume,
            },
        )
        self.session.execute(stmt)
        return len(records)

    def to_dataframe(self, prices: List[DailyPrice]) -> pd.DataFrame:
        if not prices:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        return pd.DataFrame(
            [
                {
                    "date": p.date,
                    "open": p.open,
                    "high": p.high,
                    "low": p.low,
                    "close": p.close,
                    "volume": p.volume,
                }
                for p in prices
            ]
        )
