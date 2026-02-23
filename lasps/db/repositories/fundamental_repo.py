"""FundamentalHistory 테이블 Repository."""

from typing import List, Optional

from sqlalchemy import and_
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from lasps.db.models.fundamental_history import FundamentalHistory
from lasps.db.repositories.base_repository import BaseRepository


class FundamentalRepository(BaseRepository[FundamentalHistory]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, FundamentalHistory)

    def get_by_stock_year_quarter(
        self,
        stock_code: str,
        fiscal_year: int,
        fiscal_quarter: int,
    ) -> Optional[FundamentalHistory]:
        """특정 종목의 특정 분기 데이터 조회."""
        return (
            self.session.query(FundamentalHistory)
            .filter(
                and_(
                    FundamentalHistory.stock_code == stock_code,
                    FundamentalHistory.fiscal_year == fiscal_year,
                    FundamentalHistory.fiscal_quarter == fiscal_quarter,
                )
            )
            .first()
        )

    def get_range(
        self,
        stock_code: str,
        start_year: int,
        end_year: int,
    ) -> List[FundamentalHistory]:
        """특정 종목의 연도 범위 데이터 조회."""
        return (
            self.session.query(FundamentalHistory)
            .filter(
                and_(
                    FundamentalHistory.stock_code == stock_code,
                    FundamentalHistory.fiscal_year >= start_year,
                    FundamentalHistory.fiscal_year <= end_year,
                )
            )
            .order_by(FundamentalHistory.fiscal_year, FundamentalHistory.fiscal_quarter)
            .all()
        )

    def upsert(
        self,
        stock_code: str,
        fiscal_year: int,
        fiscal_quarter: int,
        revenue: Optional[int] = None,
        operating_income: Optional[int] = None,
        net_income: Optional[int] = None,
        total_equity: Optional[int] = None,
        total_assets: Optional[int] = None,
        total_liabilities: Optional[int] = None,
        shares_outstanding: Optional[int] = None,
        eps: Optional[float] = None,
        bps: Optional[float] = None,
        roe: Optional[float] = None,
        debt_ratio: Optional[float] = None,
    ) -> None:
        """분기 재무 데이터 UPSERT."""
        stmt = sqlite_insert(FundamentalHistory).values(
            stock_code=stock_code,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            revenue=revenue,
            operating_income=operating_income,
            net_income=net_income,
            total_equity=total_equity,
            total_assets=total_assets,
            total_liabilities=total_liabilities,
            shares_outstanding=shares_outstanding,
            eps=eps,
            bps=bps,
            roe=roe,
            debt_ratio=debt_ratio,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["stock_code", "fiscal_year", "fiscal_quarter"],
            set_={
                "revenue": stmt.excluded.revenue,
                "operating_income": stmt.excluded.operating_income,
                "net_income": stmt.excluded.net_income,
                "total_equity": stmt.excluded.total_equity,
                "total_assets": stmt.excluded.total_assets,
                "total_liabilities": stmt.excluded.total_liabilities,
                "shares_outstanding": stmt.excluded.shares_outstanding,
                "eps": stmt.excluded.eps,
                "bps": stmt.excluded.bps,
                "roe": stmt.excluded.roe,
                "debt_ratio": stmt.excluded.debt_ratio,
            },
        )
        self.session.execute(stmt)
