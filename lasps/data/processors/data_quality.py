"""Data quality filtering utilities for ML training.

Provides functions to filter stocks based on data completeness criteria.
"""

from typing import List, Set
from sqlalchemy import func
from sqlalchemy.orm import Session

from lasps.config.settings import settings
from lasps.db.models.daily_price import DailyPrice
from lasps.db.models.stock import Stock


def get_valid_stock_codes(session: Session, min_rows: int = None) -> Set[str]:
    """Get stock codes that meet the minimum data row requirement.

    Args:
        session: SQLAlchemy session.
        min_rows: Minimum number of daily price rows required.
                  Defaults to settings.MIN_DAILY_PRICE_ROWS (1500).

    Returns:
        Set of stock codes that meet the criteria.
    """
    if min_rows is None:
        min_rows = settings.MIN_DAILY_PRICE_ROWS

    rows = (
        session.query(DailyPrice.stock_code, func.count(DailyPrice.id))
        .group_by(DailyPrice.stock_code)
        .having(func.count(DailyPrice.id) >= min_rows)
        .all()
    )
    return {code for code, _ in rows}


def get_valid_stocks(session: Session, min_rows: int = None) -> List[Stock]:
    """Get Stock objects that meet the minimum data row requirement.

    Args:
        session: SQLAlchemy session.
        min_rows: Minimum number of daily price rows required.
                  Defaults to settings.MIN_DAILY_PRICE_ROWS (1500).

    Returns:
        List of Stock objects that meet the criteria.
    """
    valid_codes = get_valid_stock_codes(session, min_rows)
    return (
        session.query(Stock)
        .filter(Stock.code.in_(valid_codes), Stock.is_active == True)
        .all()
    )


def get_data_quality_stats(session: Session) -> dict:
    """Get data quality statistics for all stocks.

    Returns:
        Dict with statistics about data completeness.
    """
    # Count stocks by row ranges
    rows = (
        session.query(DailyPrice.stock_code, func.count(DailyPrice.id))
        .group_by(DailyPrice.stock_code)
        .all()
    )

    ranges = {
        "2400+": 0,      # 10년+
        "2000-2399": 0,  # 8-10년
        "1500-1999": 0,  # 6-8년
        "1000-1499": 0,  # 4-6년
        "500-999": 0,    # 2-4년
        "<500": 0,       # 2년 미만
    }

    for code, cnt in rows:
        if cnt >= 2400:
            ranges["2400+"] += 1
        elif cnt >= 2000:
            ranges["2000-2399"] += 1
        elif cnt >= 1500:
            ranges["1500-1999"] += 1
        elif cnt >= 1000:
            ranges["1000-1499"] += 1
        elif cnt >= 500:
            ranges["500-999"] += 1
        else:
            ranges["<500"] += 1

    total = len(rows)
    min_rows = settings.MIN_DAILY_PRICE_ROWS
    valid_count = sum(cnt for code, cnt in rows if cnt >= min_rows)

    return {
        "total_stocks": total,
        "min_rows_threshold": min_rows,
        "valid_stocks": sum(1 for _, cnt in rows if cnt >= min_rows),
        "valid_percentage": sum(1 for _, cnt in rows if cnt >= min_rows) / total * 100 if total > 0 else 0,
        "distribution": ranges,
    }
