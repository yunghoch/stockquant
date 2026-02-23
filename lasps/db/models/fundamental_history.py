"""분기별 기본적분석 데이터 히스토리 모델."""

from sqlalchemy import BigInteger, Column, Float, Integer, SmallInteger, String, UniqueConstraint
from sqlalchemy.orm import relationship

from lasps.db.base import Base, BigIntPK


class FundamentalHistory(Base):
    """분기별 재무제표 데이터 (DART API)."""

    __tablename__ = "fundamental_history"
    __table_args__ = (
        UniqueConstraint(
            "stock_code", "fiscal_year", "fiscal_quarter",
            name="uq_fundamental_stock_year_quarter"
        ),
    )

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    stock_code = Column(String(6), nullable=False, index=True)
    fiscal_year = Column(SmallInteger, nullable=False)  # 2016~2025
    fiscal_quarter = Column(SmallInteger, nullable=False)  # 1, 2, 3, 4

    # DART에서 직접 수집 (단위: 원)
    revenue = Column(BigInteger, nullable=True)  # 매출액
    operating_income = Column(BigInteger, nullable=True)  # 영업이익
    net_income = Column(BigInteger, nullable=True)  # 당기순이익
    total_equity = Column(BigInteger, nullable=True)  # 자본총계
    total_assets = Column(BigInteger, nullable=True)  # 자산총계
    total_liabilities = Column(BigInteger, nullable=True)  # 부채총계
    shares_outstanding = Column(BigInteger, nullable=True)  # 발행주식수

    # 계산된 지표
    eps = Column(Float, nullable=True)  # 주당순이익
    bps = Column(Float, nullable=True)  # 주당순자산
    roe = Column(Float, nullable=True)  # 자기자본이익률 (%)
    debt_ratio = Column(Float, nullable=True)  # 부채비율 (%)

    def __repr__(self) -> str:
        return f"<FundamentalHistory({self.stock_code} {self.fiscal_year}Q{self.fiscal_quarter})>"
