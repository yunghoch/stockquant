import datetime
from typing import List

from sqlalchemy import and_
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from lasps.db.models.investor_trading import InvestorTrading
from lasps.db.repositories.base_repository import BaseRepository


class InvestorRepository(BaseRepository[InvestorTrading]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, InvestorTrading)

    def get_range(
        self,
        stock_code: str,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> List[InvestorTrading]:
        return (
            self.session.query(InvestorTrading)
            .filter(
                and_(
                    InvestorTrading.stock_code == stock_code,
                    InvestorTrading.date >= start_date,
                    InvestorTrading.date <= end_date,
                )
            )
            .order_by(InvestorTrading.date)
            .all()
        )

    def upsert(
        self,
        stock_code: str,
        date: datetime.date,
        foreign_net: int,
        inst_net: int,
        individual_net: int | None = None,
    ) -> None:
        stmt = sqlite_insert(InvestorTrading).values(
            stock_code=stock_code,
            date=date,
            foreign_net=foreign_net,
            inst_net=inst_net,
            individual_net=individual_net,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["stock_code", "date"],
            set_={
                "foreign_net": stmt.excluded.foreign_net,
                "inst_net": stmt.excluded.inst_net,
                "individual_net": stmt.excluded.individual_net,
            },
        )
        self.session.execute(stmt)
