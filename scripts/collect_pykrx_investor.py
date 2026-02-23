"""PyKRX를 통한 투자자별 매매 데이터 수집 스크립트.

키움 API의 OPT10059는 최근 ~100일만 제공하므로,
PyKRX로 10년치 데이터를 수집하여 investor_trading 테이블을 보완한다.

Usage:
    python scripts/collect_pykrx_investor.py --test        # 10종목 테스트
    python scripts/collect_pykrx_investor.py               # 전체 수집
    python scripts/collect_pykrx_investor.py --resume      # 이어서 수집
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from pykrx import stock as pykrx_stock

from lasps.db.engine import get_engine, get_session_factory
from lasps.db.models.investor_trading import InvestorTrading
from lasps.db.repositories.investor_repo import InvestorRepository
from lasps.db.repositories.stock_repo import StockRepository


# 수집 설정
START_DATE = "20160101"  # 10년 전
END_DATE = datetime.now().strftime("%Y%m%d")
COMMIT_BATCH_SIZE = 1  # 종목당 커밋 (안전)


def get_collected_stock_codes(session) -> set:
    """이미 10년치 데이터가 있는 종목코드를 조회한다."""
    from sqlalchemy import func

    # 2000행 이상 = 약 8년치 이상 (resume용)
    MIN_ROWS = 2000

    rows = (
        session.query(InvestorTrading.stock_code, func.count(InvestorTrading.id))
        .group_by(InvestorTrading.stock_code)
        .having(func.count(InvestorTrading.id) >= MIN_ROWS)
        .all()
    )
    return {code for code, _ in rows}


def collect_single_stock(
    stock_code: str,
    investor_repo: InvestorRepository,
    session,
) -> int:
    """단일 종목의 투자자 데이터를 PyKRX로 수집한다.

    Returns:
        수집된 행 수.
    """
    try:
        # PyKRX로 데이터 조회
        df = pykrx_stock.get_market_trading_value_by_date(
            START_DATE, END_DATE, stock_code
        )

        if df is None or len(df) == 0:
            logger.warning(f"{stock_code}: 데이터 없음")
            return 0

        # DB에 저장
        count = 0
        for date_idx, row in df.iterrows():
            # date_idx는 pandas Timestamp
            date = date_idx.date() if hasattr(date_idx, 'date') else date_idx

            # PyKRX 컬럼: 기관합계, 기타법인, 개인, 외국인합계, 전체
            foreign_net = int(row.get('외국인합계', 0) or 0)
            inst_net = int(row.get('기관합계', 0) or 0)
            individual_net = int(row.get('개인', 0) or 0)

            investor_repo.upsert(
                stock_code=stock_code,
                date=date,
                foreign_net=foreign_net,
                inst_net=inst_net,
                individual_net=individual_net,
            )
            count += 1

        session.commit()
        return count

    except Exception as e:
        session.rollback()
        logger.error(f"{stock_code}: 수집 실패 - {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="PyKRX 투자자 데이터 수집")
    parser.add_argument("--test", action="store_true", help="10종목 테스트")
    parser.add_argument("--resume", action="store_true", help="이어서 수집")
    parser.add_argument("--limit", type=int, help="수집할 종목 수 제한")
    args = parser.parse_args()

    logger.info("=== PyKRX 투자자 데이터 수집 시작 ===")
    logger.info(f"기간: {START_DATE} ~ {END_DATE}")

    # DB 세션
    SessionLocal = get_session_factory()
    session = SessionLocal()

    try:
        stock_repo = StockRepository(session)
        investor_repo = InvestorRepository(session)

        # 활성 종목 조회
        stocks = stock_repo.get_active()
        total_stocks = len(stocks)
        logger.info(f"활성 종목: {total_stocks}개")

        # 테스트 모드
        if args.test:
            # 대표 종목 10개
            test_codes = ['005930', '000660', '035420', '051910', '006400',
                          '035720', '005380', '003550', '017670', '105560']
            stocks = [s for s in stocks if s.code in test_codes]
            logger.info(f"테스트 모드: {len(stocks)}종목")

        # 종목 수 제한
        if args.limit:
            stocks = stocks[:args.limit]
            logger.info(f"제한 모드: {args.limit}종목")

        # Resume 모드: 이미 수집된 종목 스킵
        skip_codes = set()
        if args.resume:
            skip_codes = get_collected_stock_codes(session)
            logger.info(f"Resume: {len(skip_codes)}종목 이미 수집됨")

        # 수집 시작
        collected = 0
        skipped = 0
        failed = 0
        start_time = time.time()

        for i, s in enumerate(stocks, 1):
            if s.code in skip_codes:
                skipped += 1
                continue

            count = collect_single_stock(s.code, investor_repo, session)

            if count > 0:
                collected += 1
                elapsed = time.time() - start_time
                speed = collected / elapsed * 60 if elapsed > 0 else 0
                remaining = (len(stocks) - i) / speed if speed > 0 else 0

                logger.info(
                    f"[{i}/{len(stocks)}] {s.code} ({s.name}): {count}행 "
                    f"(속도: {speed:.1f}종목/분, 남은시간: {remaining:.0f}분)"
                )
            else:
                failed += 1

            # Rate limiting (PyKRX는 KRX 스크래핑이므로 간격 필요)
            time.sleep(0.5)

        elapsed = time.time() - start_time
        logger.info("=== 수집 완료 ===")
        logger.info(f"수집: {collected}종목, 스킵: {skipped}종목, 실패: {failed}종목")
        logger.info(f"소요시간: {elapsed/60:.1f}분")

    finally:
        session.close()


if __name__ == "__main__":
    main()
