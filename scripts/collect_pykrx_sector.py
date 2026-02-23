#!/usr/bin/env python
"""pykrx에서 종목별 섹터 정보를 수집하여 DB에 저장.

pykrx 업종 인덱스에서 구성 종목을 조회하여 stocks 테이블의
pykrx_sector_idx, pykrx_sector_name 컬럼에 저장한다.

Usage:
    python scripts/collect_pykrx_sector.py              # 전체 수집
    python scripts/collect_pykrx_sector.py --dry-run    # 미리보기만
    python scripts/collect_pykrx_sector.py --market KOSPI  # KOSPI만
"""

import argparse
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from pykrx import stock as pykrx_stock
from sqlalchemy import text

from lasps.db.engine import get_engine, get_session_factory
from lasps.db.models.stock import Stock
from lasps.utils.logger import setup_logger


# pykrx 업종 인덱스 (KOSPI 기준, KOSDAQ은 2xxx)
# 기존 업종 + IT서비스(1046), 부동산(1045), 오락문화(1047) 추가
PYKRX_SECTOR_INDICES = [
    "1005", "1006", "1007", "1008", "1009", "1010", "1011", "1012",
    "1013", "1014", "1015", "1016", "1017", "1018", "1019", "1020",
    "1021", "1024", "1025", "1026", "1027",
    "1045", "1046", "1047",  # 부동산, IT서비스, 오락문화 (NAVER, 카카오 등)
]


def ensure_columns_exist(engine):
    """pykrx 컬럼이 없으면 추가 (SQLite ALTER TABLE)."""
    with engine.connect() as conn:
        # 컬럼 존재 확인
        result = conn.execute(text("PRAGMA table_info(stocks)"))
        columns = {row[1] for row in result.fetchall()}

        if "pykrx_sector_idx" not in columns:
            conn.execute(text("ALTER TABLE stocks ADD COLUMN pykrx_sector_idx VARCHAR(10)"))
            logger.info("Added column: pykrx_sector_idx")

        if "pykrx_sector_name" not in columns:
            conn.execute(text("ALTER TABLE stocks ADD COLUMN pykrx_sector_name VARCHAR(30)"))
            logger.info("Added column: pykrx_sector_name")

        conn.commit()


def collect_pykrx_sectors(market: str = "ALL") -> dict:
    """pykrx에서 종목별 섹터 정보 수집.

    Args:
        market: "KOSPI", "KOSDAQ", or "ALL"

    Returns:
        Dict[ticker, (sector_idx, sector_name)]
    """
    ticker_to_sector = {}
    sector_stats = defaultdict(int)

    # 기준일 (최근 거래일)
    ref_date = datetime.now().strftime("%Y%m%d")

    # KOSPI 업종
    if market in ("KOSPI", "ALL"):
        logger.info("=== KOSPI 업종 수집 ===")
        for idx in PYKRX_SECTOR_INDICES:
            try:
                name = pykrx_stock.get_index_ticker_name(idx)
                components = pykrx_stock.get_index_portfolio_deposit_file(idx, ref_date)

                for ticker in components:
                    if ticker not in ticker_to_sector:
                        ticker_to_sector[ticker] = (idx, name)
                        sector_stats[f"{idx}:{name}"] += 1

                logger.info(f"  {idx}: {name} ({len(components)} stocks)")
                time.sleep(0.3)  # Rate limiting
            except Exception as e:
                logger.warning(f"  {idx}: Failed - {e}")

    # KOSDAQ 업종 (2xxx)
    if market in ("KOSDAQ", "ALL"):
        logger.info("=== KOSDAQ 업종 수집 ===")
        kosdaq_indices = ["2" + idx[1:] for idx in PYKRX_SECTOR_INDICES]

        for idx in kosdaq_indices:
            try:
                name = pykrx_stock.get_index_ticker_name(idx)
                components = pykrx_stock.get_index_portfolio_deposit_file(idx, ref_date)

                for ticker in components:
                    if ticker not in ticker_to_sector:
                        ticker_to_sector[ticker] = (idx, name)
                        sector_stats[f"{idx}:{name}"] += 1

                if len(components) > 0:
                    logger.info(f"  {idx}: {name} ({len(components)} stocks)")
                time.sleep(0.3)
            except Exception:
                pass  # KOSDAQ은 일부 업종이 없을 수 있음

    logger.info(f"\nTotal: {len(ticker_to_sector)} stocks mapped")

    # 섹터별 통계
    logger.info("\n=== Sector Distribution ===")
    for sector, count in sorted(sector_stats.items(), key=lambda x: -x[1]):
        logger.info(f"  {sector}: {count}")

    return ticker_to_sector


def save_to_db(ticker_to_sector: dict, dry_run: bool = False) -> dict:
    """DB에 pykrx 섹터 정보 저장.

    Returns:
        Dict with update stats.
    """
    engine = get_engine()

    # 컬럼 존재 확인 및 추가 (dry-run에서도 필요)
    ensure_columns_exist(engine)

    SessionLocal = get_session_factory()
    session = SessionLocal()

    stats = {"updated": 0, "not_found": 0, "already_set": 0}

    try:
        # 모든 종목 조회
        stocks = session.query(Stock).all()
        stock_map = {s.code: s for s in stocks}

        logger.info(f"\nDB stocks: {len(stock_map)}, pykrx mapped: {len(ticker_to_sector)}")

        for ticker, (sector_idx, sector_name) in ticker_to_sector.items():
            if ticker in stock_map:
                stock = stock_map[ticker]

                # 이미 같은 값이면 스킵
                if stock.pykrx_sector_idx == sector_idx and stock.pykrx_sector_name == sector_name:
                    stats["already_set"] += 1
                    continue

                if not dry_run:
                    stock.pykrx_sector_idx = sector_idx
                    stock.pykrx_sector_name = sector_name

                stats["updated"] += 1
            else:
                stats["not_found"] += 1

        if not dry_run:
            session.commit()
            logger.info("Changes committed to DB")

    finally:
        session.close()

    return stats


def print_summary(ticker_to_sector: dict, stats: dict, dry_run: bool):
    """결과 요약 출력."""
    print("\n" + "=" * 60)
    print("PYKRX SECTOR COLLECTION SUMMARY")
    print("=" * 60)
    print(f"Total stocks from pykrx: {len(ticker_to_sector)}")
    print(f"Updated in DB:           {stats['updated']}")
    print(f"Already set (skipped):   {stats['already_set']}")
    print(f"Not found in DB:         {stats['not_found']}")
    print("=" * 60)

    if dry_run:
        print("\n[DRY RUN] No changes saved. Run without --dry-run to save.")


def main():
    parser = argparse.ArgumentParser(description="Collect pykrx sector info")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--market", type=str, default="ALL",
                        choices=["KOSPI", "KOSDAQ", "ALL"],
                        help="Market to collect (default: ALL)")
    args = parser.parse_args()

    setup_logger("INFO")

    logger.info("=== pykrx Sector Collection Start ===")
    logger.info(f"Market: {args.market}, Dry-run: {args.dry_run}")

    # 1. pykrx에서 수집
    ticker_to_sector = collect_pykrx_sectors(args.market)

    # 2. DB에 저장
    stats = save_to_db(ticker_to_sector, args.dry_run)

    # 3. 결과 출력
    print_summary(ticker_to_sector, stats, args.dry_run)


if __name__ == "__main__":
    main()
