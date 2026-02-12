#!/usr/bin/env python
# scripts/historical_data.py

"""
3-Phase 파이프라인: 키움 데이터 수집 → 파생 데이터 계산 → (선택) .npy 내보내기

모든 종목의 10년치 데이터를 수집하고 기술지표/감성/라벨을 계산하여 SQLite DB에 저장한다.
실행 중 중단되면 --resume 으로 이어서 수집할 수 있다.

Usage:
    # Mock 모드 (테스트, 5종목)
    python scripts/historical_data.py --mode mock --days 2600

    # 실제 키움 API (전 종목, 10년치)
    python scripts/historical_data.py --mode real --days 2600

    # 중단 후 이어서 수집
    python scripts/historical_data.py --mode real --days 2600 --resume

    # 특정 종목만 수집
    python scripts/historical_data.py --mode real --stocks 005930,000660,035420

    # Phase 2만 실행 (파생 데이터 계산)
    python scripts/historical_data.py --phase 2

    # NPY 내보내기까지 전체 실행
    python scripts/historical_data.py --mode mock --export-npy --output data/processed
"""

import argparse
import datetime
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Optional, Tuple

# OpenMP 중복 라이브러리 오류 방지 (mplfinance + numpy)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session, sessionmaker

from lasps.config.sector_config import SECTOR_CODES
from lasps.data.collectors.kiwoom_collector import KiwoomCollector
from lasps.data.collectors.kiwoom_mock import MOCK_STOCKS, KiwoomMockAPI
from lasps.data.processors.market_sentiment import MarketSentimentCalculator
from lasps.data.processors.technical_indicators import TechnicalIndicatorCalculator
from lasps.db.base import Base
from lasps.db.models import Sector
from lasps.db.models.daily_price import DailyPrice
from lasps.db.repositories import (
    IndicatorRepository,
    InvestorRepository,
    PriceRepository,
    SentimentRepository,
    ShortSellingRepository,
    StockRepository,
    TrainingLabelRepository,
)
from lasps.utils.constants import (
    INDICATOR_FEATURES,
    OHLCV_FEATURES,
    SENTIMENT_FEATURES,
    TIME_SERIES_LENGTH,
)
from lasps.utils.helpers import compute_label, normalize_time_series
from lasps.utils.logger import setup_logger

SPLIT_CONFIG = {
    "train": ("2015-01-01", "2022-12-31"),
    "val": ("2023-01-01", "2023-12-31"),
    "test": ("2024-01-01", "2024-12-31"),
}

# 종목당 최소 가격 데이터 행 수 (이 미만이면 파생 데이터 계산 스킵)
MIN_PRICE_ROWS = 120

# 종목별 수집 재시도 설정
MAX_RETRIES = 2      # 최대 재시도 횟수 (총 3회 시도)
RETRY_DELAY_S = 5.0  # 재시도 대기 시간(초)

# 키움 API 시간당 제한 준수를 위한 종목간 대기 시간
# 1시간에 1000 TR 제한 → 종목당 4 TR → 250종목/시간 → 14.4초/종목
# 안전 마진 적용하여 18초로 설정 (200종목/시간, 800 TR/시간)
STOCK_DELAY_S = 18.0  # 종목간 대기 시간(초) - real 모드에서만 적용

# 차트 생성 병렬 처리 설정
CHART_WORKERS = max(1, cpu_count() - 1)  # CPU 코어 - 1 (시스템 여유)
CHART_BATCH_SIZE = 100  # 배치당 차트 수 (메모리 관리)


def _generate_and_save_chart(args: Tuple[int, np.ndarray, str]) -> Tuple[int, bool]:
    """단일 차트 이미지 생성 및 파일 저장 (멀티프로세싱용).

    Args:
        args: (인덱스, OHLCV 데이터 배열, 저장 경로) 튜플
              OHLCV는 (60, 6) shape - [date_ordinal, O, H, L, C, V]

    Returns:
        (인덱스, 성공 여부) 튜플
    """
    import pandas as pd
    from lasps.data.processors.chart_generator import ChartGenerator

    idx, ohlcv_arr, save_path = args
    try:
        # date_ordinal을 datetime으로 복원
        dates = [datetime.date.fromordinal(int(d)) for d in ohlcv_arr[:, 0]]
        chart_df = pd.DataFrame({
            "Open": ohlcv_arr[:, 1],
            "High": ohlcv_arr[:, 2],
            "Low": ohlcv_arr[:, 3],
            "Close": ohlcv_arr[:, 4],
            "Volume": ohlcv_arr[:, 5],
        }, index=pd.to_datetime(dates))

        gen = ChartGenerator()
        gen.save_chart(chart_df, save_path)
        return (idx, True)
    except Exception as e:
        # 실패 시 로그만 남김
        return (idx, False)


def get_split(date: datetime.date) -> str:
    """날짜에 해당하는 split을 반환한다."""
    for split, (start, end) in SPLIT_CONFIG.items():
        s = datetime.date.fromisoformat(start)
        e = datetime.date.fromisoformat(end)
        if s <= date <= e:
            return split
    return ""


def seed_sectors(session: Session) -> None:
    """sectors 테이블에 20개 업종을 시딩한다."""
    existing = session.query(Sector).count()
    if existing >= 20:
        logger.info(f"Sectors already seeded ({existing} rows)")
        return
    for code, (sector_id, name, _) in SECTOR_CODES.items():
        if session.get(Sector, sector_id) is None:
            session.add(Sector(id=sector_id, code=code, name=name))
    session.flush()
    session.commit()
    logger.info("Seeded 20 sectors")


def get_collected_stock_codes(session: Session) -> set:
    """이미 가격 데이터가 DB에 있는 종목코드를 조회한다 (resume 용).

    - 일반 주식 (숫자만): MIN_PRICE_ROWS 이상이면 스킵
    - 파생상품 (알파벳 포함): 데이터가 있으면 무조건 스킵 (재수집해도 더 없음)
    """
    rows = (
        session.query(DailyPrice.stock_code, func.count(DailyPrice.id))
        .group_by(DailyPrice.stock_code)
        .all()
    )
    result = set()
    for code, count in rows:
        # 파생상품 (알파벳 포함): 데이터가 있으면 스킵
        if not code.isdigit():
            result.add(code)
        # 일반 주식: MIN_PRICE_ROWS 이상이면 스킵
        elif count >= MIN_PRICE_ROWS:
            result.add(code)
    return result


def create_api_and_collector(mode: str) -> tuple:
    """모드에 따라 API와 Collector를 생성한다.

    Args:
        mode: "real" 또는 "mock".

    Returns:
        (api, collector) 튜플.
    """
    if mode == "real":
        from lasps.data.collectors.kiwoom_real import KiwoomRealAPI

        api = KiwoomRealAPI()
        logger.info("Connecting to Kiwoom OpenAPI+ ...")
        if not api.connect():
            logger.error("Kiwoom API 로그인 실패. 키움 OpenAPI+가 설치되어 있는지 확인하세요.")
            sys.exit(1)
        logger.info("Kiwoom API 로그인 성공")
        collector = KiwoomCollector(api, rate_limit=True)
    else:
        api = KiwoomMockAPI(seed=42)
        collector = KiwoomCollector(api, rate_limit=False)
        logger.info("Mock API 사용 (테스트 모드)")

    return api, collector


def resolve_stock_codes(
    collector: KiwoomCollector,
    mode: str,
    stocks_arg: Optional[str],
    market: str,
) -> List[str]:
    """수집 대상 종목코드 리스트를 결정한다.

    Args:
        collector: KiwoomCollector 인스턴스.
        mode: "real" 또는 "mock".
        stocks_arg: 쉼표로 구분된 종목코드 (없으면 전체).
        market: "all", "kospi", "kosdaq".

    Returns:
        종목코드 리스트.
    """
    if stocks_arg:
        codes = [c.strip() for c in stocks_arg.split(",") if c.strip()]
        logger.info(f"사용자 지정 종목: {len(codes)}개")
        return codes

    if mode == "mock":
        codes = list(MOCK_STOCKS.keys())
        logger.info(f"Mock 종목: {len(codes)}개")
        return codes

    # 실제 API: 시장별 전체 종목 조회
    if market == "kospi":
        codes = collector.api.get_code_list_by_market("0")
        logger.info(f"KOSPI 종목: {len(codes)}개")
    elif market == "kosdaq":
        codes = collector.api.get_code_list_by_market("10")
        logger.info(f"KOSDAQ 종목: {len(codes)}개")
    else:
        codes = collector.get_all_stock_codes()

    return codes


# ── Phase 1: 원시 데이터 수집 ──────────────────────────────

def collect_raw_data(
    session: Session,
    collector: KiwoomCollector,
    stock_codes: List[str],
    days: int,
    resume: bool,
    mode: str = "mock",
) -> None:
    """Phase 1: API에서 원시 데이터를 수집하여 DB에 저장한다.

    Args:
        session: SQLAlchemy session.
        collector: KiwoomCollector 인스턴스.
        stock_codes: 수집 대상 종목코드 리스트.
        days: 수집할 거래일 수.
        resume: True이면 이미 수집된 종목은 건너뛴다.
        mode: "real" 또는 "mock" - real일 때만 종목간 대기 적용.
    """
    total = len(stock_codes)
    logger.info(f"=== Phase 1: 원시 데이터 수집 ({total}종목, {days}일) ===")

    seed_sectors(session)

    stock_repo = StockRepository(session)
    price_repo = PriceRepository(session)
    investor_repo = InvestorRepository(session)
    short_repo = ShortSellingRepository(session)

    # resume: 이미 수집된 종목 스킵
    skip_codes: set = set()
    if resume:
        skip_codes = get_collected_stock_codes(session)
        if skip_codes:
            logger.info(f"Resume: {len(skip_codes)}종목 이미 수집됨, 건너뜀")

    collected = 0
    failed = 0
    failed_codes: List[str] = []
    start_time = time.time()

    for i, code in enumerate(stock_codes, 1):
        if code in skip_codes:
            logger.debug(f"[{i}/{total}] {code} - 이미 수집됨, 스킵")
            collected += 1
            continue

        last_err: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 2):  # 1 ~ MAX_RETRIES+1
            try:
                _collect_single_stock(
                    code, collector, stock_repo, price_repo,
                    investor_repo, short_repo, days,
                )
                session.commit()
                collected += 1

                elapsed = time.time() - start_time
                speed = collected / elapsed if elapsed > 0 else 0
                remaining = (total - i) / speed if speed > 0 else 0
                # 시간당 요청 제한 정보 (real 모드)
                hourly_info = ""
                if mode == "real" and hasattr(collector.api, "get_hourly_request_count"):
                    hourly_count = collector.api.get_hourly_request_count()
                    hourly_info = f", TR: {hourly_count}/1000"

                logger.info(
                    f"[{i}/{total}] {code} 완료 "
                    f"(속도: {speed:.1f}종목/분*60, 남은시간: {remaining/60:.0f}분{hourly_info})"
                )

                # real 모드: 시간당 제한 준수를 위해 종목간 대기
                if mode == "real" and i < total:
                    time.sleep(STOCK_DELAY_S)

                break  # 성공 시 루프 탈출

            except Exception as e:
                session.rollback()
                last_err = e
                if attempt <= MAX_RETRIES:
                    logger.warning(
                        f"[{i}/{total}] {code} 시도 {attempt} 실패, "
                        f"{RETRY_DELAY_S}초 후 재시도: {e}"
                    )
                    time.sleep(RETRY_DELAY_S)
                else:
                    failed += 1
                    failed_codes.append(code)
                    logger.error(
                        f"[{i}/{total}] {code} 최종 실패 "
                        f"({MAX_RETRIES + 1}회 시도): {last_err}"
                    )

    logger.info(
        f"Phase 1 완료: 성공 {collected}/{total}, 실패 {failed}"
    )
    if failed_codes:
        logger.warning(f"실패 종목: {failed_codes[:20]}{'...' if len(failed_codes) > 20 else ''}")


def _collect_single_stock(
    code: str,
    collector: KiwoomCollector,
    stock_repo: StockRepository,
    price_repo: PriceRepository,
    investor_repo: InvestorRepository,
    short_repo: ShortSellingRepository,
    days: int,
) -> None:
    """한 종목의 모든 원시 데이터를 수집한다."""
    # 1) 종목 정보 (OPT10001)
    info = collector.get_stock_info(code)
    stock_repo.upsert(
        code,
        name=info["name"],
        sector_id=info["sector_id"],
        sector_code=info["sector_code"],
        market_cap=int(info["market_cap"]) if info["market_cap"] else None,
        per=float(info["per"]) if info["per"] else None,
        pbr=float(info["pbr"]) if info["pbr"] else None,
        roe=float(info["roe"]) if info["roe"] else None,
        is_active=True,
    )

    # 2) OHLCV (OPT10081)
    ohlcv_df = collector.get_daily_ohlcv(code, days=days)
    count = price_repo.upsert_from_dataframe(code, ohlcv_df)
    logger.debug(f"  {code} OHLCV: {count}행")

    # 3) 투자자별 매매 (OPT10059)
    try:
        investor_df = collector.get_investor_data(code, days=days)
        for _, row in investor_df.iterrows():
            dt = row["date"]
            d = dt.date() if hasattr(dt, "date") and callable(dt.date) else dt
            investor_repo.upsert(code, d, int(row["foreign_net"]), int(row["inst_net"]))
        logger.debug(f"  {code} Investor: {len(investor_df)}행")
    except Exception as e:
        logger.warning(f"  {code} 투자자 데이터 수집 실패 (계속 진행): {e}")

    # 4) 공매도 (OPT10014) - ELW/ETN 등 비주식 종목(코드에 알파벳 포함)은 스킵
    if not code.isdigit():
        logger.debug(f"  {code} 비주식 종목 → 공매도 스킵")
    else:
        try:
            short_df = collector.get_short_selling(code, days=days)
            for _, row in short_df.iterrows():
                dt = row["date"]
                d = dt.date() if hasattr(dt, "date") and callable(dt.date) else dt
                short_repo.upsert(code, d, int(row["short_volume"]), float(row["short_ratio"]))
            logger.debug(f"  {code} Short: {len(short_df)}행")
        except Exception as e:
            logger.warning(f"  {code} 공매도 데이터 수집 실패 (계속 진행): {e}")


# ── Phase 2: 파생 데이터 계산 ──────────────────────────────

def compute_derived_data(session: Session) -> None:
    """Phase 2: 기술지표, 시장감성, 학습라벨을 계산하여 DB에 저장한다."""
    logger.info("=== Phase 2: 파생 데이터 계산 ===")

    stock_repo = StockRepository(session)
    price_repo = PriceRepository(session)
    investor_repo = InvestorRepository(session)
    indicator_repo = IndicatorRepository(session)
    sentiment_repo = SentimentRepository(session)
    label_repo = TrainingLabelRepository(session)

    ind_calc = TechnicalIndicatorCalculator()
    sent_calc = MarketSentimentCalculator()

    stocks = stock_repo.get_active()
    total = len(stocks)
    logger.info(f"{total}종목 처리")

    completed = 0
    failed = 0
    start_time = time.time()

    for i, stock in enumerate(stocks, 1):
        code = stock.code
        sector_id = stock.sector_id

        try:
            _compute_single_stock(
                code, sector_id, price_repo, investor_repo,
                indicator_repo, sentiment_repo, label_repo,
                ind_calc, sent_calc,
            )
            session.commit()
            completed += 1

            if i % 50 == 0 or i == total:
                elapsed = time.time() - start_time
                logger.info(
                    f"Phase 2 진행: [{i}/{total}] "
                    f"(완료: {completed}, 실패: {failed}, "
                    f"경과: {elapsed/60:.1f}분)"
                )

        except Exception as e:
            session.rollback()
            failed += 1
            logger.error(f"[{i}/{total}] {code} 파생 데이터 계산 실패: {e}")

    logger.info(f"Phase 2 완료: 성공 {completed}/{total}, 실패 {failed}")


def _compute_single_stock(
    code: str,
    sector_id: Optional[int],
    price_repo: PriceRepository,
    investor_repo: InvestorRepository,
    indicator_repo: IndicatorRepository,
    sentiment_repo: SentimentRepository,
    label_repo: TrainingLabelRepository,
    ind_calc: TechnicalIndicatorCalculator,
    sent_calc: MarketSentimentCalculator,
) -> None:
    """한 종목의 기술지표, 감성, 라벨을 계산한다."""
    prices = price_repo.get_range(
        code, datetime.date(2014, 1, 1), datetime.date(2025, 12, 31),
    )
    if not prices or len(prices) < MIN_PRICE_ROWS:
        logger.debug(f"  {code}: 가격 데이터 부족 ({len(prices) if prices else 0}행), 스킵")
        return

    ohlcv_df = price_repo.to_dataframe(prices)
    ohlcv_df["date"] = pd.to_datetime(ohlcv_df["date"])

    # 기술지표 15개
    with_indicators = ind_calc.calculate(ohlcv_df)
    ind_count = 0
    for _, row in with_indicators.iterrows():
        dt = row["date"]
        d = dt.date() if hasattr(dt, "date") and callable(dt.date) else dt
        values = {}
        for feat in INDICATOR_FEATURES:
            v = row.get(feat)
            if pd.notna(v):
                values[feat] = float(v)
        if values:
            indicator_repo.upsert(code, d, values)
            ind_count += 1
    logger.debug(f"  {code} Indicators: {ind_count}행")

    # 시장감성 5D
    investor_records = investor_repo.get_range(
        code, datetime.date(2014, 1, 1), datetime.date(2025, 12, 31),
    )
    investor_df = None
    if investor_records:
        investor_df = pd.DataFrame([
            {"date": pd.Timestamp(r.date), "foreign_net": r.foreign_net,
             "inst_net": r.inst_net}
            for r in investor_records
        ])

    sentiment_df = sent_calc.calculate(ohlcv_df, investor_df)
    sent_count = 0
    for _, row in sentiment_df.iterrows():
        dt = row["date"]
        d = dt.date() if hasattr(dt, "date") and callable(dt.date) else dt
        values = {}
        for feat in SENTIMENT_FEATURES:
            v = row.get(feat)
            if pd.notna(v):
                values[feat] = float(v)
        if values:
            sentiment_repo.upsert(code, d, values)
            sent_count += 1
    logger.debug(f"  {code} Sentiment: {sent_count}행")

    # 학습 라벨 (5일 후 수익률 ±3%)
    close_series = ohlcv_df["close"].reset_index(drop=True)
    dates = ohlcv_df["date"].reset_index(drop=True)
    label_count = 0
    for idx in range(len(ohlcv_df)):
        lbl = compute_label(close_series, idx)
        if lbl == -1:
            continue
        dt = dates.iloc[idx]
        d = dt.date() if hasattr(dt, "date") and callable(dt.date) else dt
        split = get_split(d)
        if not split:
            continue
        label_repo.upsert(code, d, lbl, split, sector_id)
        label_count += 1
    logger.debug(f"  {code} Labels: {label_count}행")


# ── Phase 3: NPY 내보내기 (선택) ──────────────────────────

def export_npy(
    session: Session,
    output_dir: Path,
    stride: int,
    no_charts: bool,
    n_workers: Optional[int] = None,
) -> None:
    """Phase 3: DB 데이터를 .npy 파일로 내보낸다.

    차트 이미지는 개별 PNG 파일로 저장하여 메모리 사용량을 최소화한다.

    Args:
        session: SQLAlchemy session.
        output_dir: 출력 디렉토리.
        stride: 슬라이딩 윈도우 간격.
        no_charts: True이면 차트 이미지 생성 건너뛰기.
        n_workers: 차트 생성 병렬 워커 수 (기본: CPU 코어 - 1).

    출력 구조:
        output_dir/
        ├── train/
        │   ├── charts/          # 개별 차트 이미지 (PNG)
        │   │   ├── 000000.png
        │   │   ├── 000001.png
        │   │   └── ...
        │   ├── time_series.npy  # (N, 60, 28)
        │   ├── sector_ids.npy   # (N,)
        │   ├── labels.npy       # (N,)
        │   └── metadata.csv     # stock_code, date, chart_idx 매핑
        ├── val/
        └── test/
    """
    from lasps.config.settings import settings
    from lasps.data.processors.data_quality import get_valid_stock_codes

    if n_workers is None:
        n_workers = CHART_WORKERS

    logger.info(f"=== Phase 3: NPY 내보내기 → {output_dir} (stride={stride}) ===")
    if not no_charts:
        logger.info(f"차트 생성 병렬화: {n_workers} workers (개별 PNG 저장)")

    # 데이터 품질 필터: MIN_DAILY_PRICE_ROWS 이상인 종목만 사용
    valid_codes = get_valid_stock_codes(session, settings.MIN_DAILY_PRICE_ROWS)
    logger.info(f"데이터 품질 필터: {len(valid_codes)}종목 (최소 {settings.MIN_DAILY_PRICE_ROWS}행)")

    stock_repo = StockRepository(session)
    price_repo = PriceRepository(session)
    indicator_repo = IndicatorRepository(session)
    sentiment_repo = SentimentRepository(session)
    label_repo = TrainingLabelRepository(session)

    generate_charts = not no_charts
    feature_cols = list(OHLCV_FEATURES) + list(INDICATOR_FEATURES) + list(SENTIMENT_FEATURES)

    for split, (start_str, end_str) in SPLIT_CONFIG.items():
        start_date = datetime.date.fromisoformat(start_str)
        end_date = datetime.date.fromisoformat(end_str)
        logger.info(f"Exporting {split}: {start_str} ~ {end_str}")

        all_time_series: list = []
        all_chart_tasks: list = []  # (idx, ohlcv_data, save_path) 튜플
        all_sector_ids: list = []
        all_labels: list = []
        all_metadata: list = []  # (stock_code, date, chart_idx)

        # 차트 저장 디렉토리 생성
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        if generate_charts:
            charts_dir = split_dir / "charts"
            charts_dir.mkdir(parents=True, exist_ok=True)

        # 데이터 품질 필터 적용
        all_stocks = stock_repo.get_active()
        stocks = [s for s in all_stocks if s.code in valid_codes]
        logger.info(f"  필터링 후 종목: {len(stocks)}개 (전체 {len(all_stocks)}개)")

        # Step 1: 모든 샘플 데이터 수집 (시계열 + 차트 태스크)
        sample_count = 0
        for stock in stocks:
            code = stock.code
            sector_id = stock.sector_id or 0

            labels = label_repo.get_by_stock_range(code, start_date, end_date)
            if not labels:
                continue
            label_map = {lbl.date: lbl.label for lbl in labels}

            lookback_start = start_date - datetime.timedelta(days=200)
            prices = price_repo.get_range(code, lookback_start, end_date)
            indicators = indicator_repo.get_range(code, lookback_start, end_date)
            sentiments = sentiment_repo.get_range(code, lookback_start, end_date)

            if not prices:
                continue

            price_df = pd.DataFrame([
                {"date": p.date, "open": p.open, "high": p.high,
                 "low": p.low, "close": p.close, "volume": p.volume}
                for p in prices
            ])

            ind_df = pd.DataFrame([
                {"date": ind.date,
                 **{f: getattr(ind, f) for f in INDICATOR_FEATURES}}
                for ind in indicators
            ]) if indicators else pd.DataFrame(
                columns=["date"] + list(INDICATOR_FEATURES)
            )

            sent_df = pd.DataFrame([
                {"date": s.date,
                 **{f: getattr(s, f) for f in SENTIMENT_FEATURES}}
                for s in sentiments
            ]) if sentiments else pd.DataFrame(
                columns=["date"] + list(SENTIMENT_FEATURES)
            )

            merged = price_df.merge(ind_df, on="date", how="left")
            merged = merged.merge(sent_df, on="date", how="left")
            merged = merged.sort_values("date").reset_index(drop=True)

            for col in feature_cols:
                if col in merged.columns:
                    merged[col] = merged[col].fillna(0.0)

            dates_in_split = sorted(label_map.keys())

            for target_date in dates_in_split[::stride]:
                date_idx = merged.index[merged["date"] == target_date]
                if len(date_idx) == 0:
                    continue
                idx = date_idx[0]
                if idx < TIME_SERIES_LENGTH - 1:
                    continue

                window_start = idx - TIME_SERIES_LENGTH + 1
                window = merged.iloc[window_start:idx + 1]
                if len(window) < TIME_SERIES_LENGTH:
                    continue

                # 시계열 데이터
                ts_data = window[feature_cols].values.astype(np.float32)
                ts_normalized = normalize_time_series(ts_data)
                all_time_series.append(ts_normalized)

                # 차트 태스크 생성 (메모리에 이미지 저장 안 함)
                if generate_charts:
                    chart_path = str(charts_dir / f"{sample_count:06d}.png")
                    chart_data = np.column_stack([
                        [d.toordinal() for d in window["date"]],
                        window["open"].values,
                        window["high"].values,
                        window["low"].values,
                        window["close"].values,
                        window["volume"].values,
                    ]).astype(np.float64)
                    all_chart_tasks.append((sample_count, chart_data, chart_path))

                # 메타데이터
                all_metadata.append({
                    "stock_code": code,
                    "date": target_date.isoformat(),
                    "chart_idx": sample_count,
                })

                all_sector_ids.append(sector_id)
                all_labels.append(label_map[target_date])
                sample_count += 1

        if not all_time_series:
            logger.warning(f"No samples for {split}, skipping")
            continue

        logger.info(f"  수집 완료: {sample_count}개 샘플")

        # Step 2: 차트 이미지 병렬 생성 (개별 파일로 저장)
        if generate_charts and all_chart_tasks:
            logger.info(f"  차트 이미지 생성 시작 ({len(all_chart_tasks)}개 → 개별 PNG)...")
            chart_start_time = time.time()

            success_count = 0
            fail_count = 0

            # 멀티프로세싱으로 차트 생성 및 저장
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                # 배치 단위로 처리
                completed = 0
                for batch_start in range(0, len(all_chart_tasks), CHART_BATCH_SIZE):
                    batch_end = min(batch_start + CHART_BATCH_SIZE, len(all_chart_tasks))
                    batch_tasks = all_chart_tasks[batch_start:batch_end]

                    futures = {
                        executor.submit(_generate_and_save_chart, task): task[0]
                        for task in batch_tasks
                    }

                    for future in as_completed(futures):
                        idx, success = future.result()
                        if success:
                            success_count += 1
                        else:
                            fail_count += 1
                        completed += 1

                    # 진행 상황 로깅 (배치 완료 시)
                    elapsed = time.time() - chart_start_time
                    speed = completed / elapsed if elapsed > 0 else 0
                    remaining = (len(all_chart_tasks) - completed) / speed if speed > 0 else 0
                    logger.info(
                        f"  차트 생성 진행: {completed}/{len(all_chart_tasks)} "
                        f"({speed:.1f}/초, 남은시간: {remaining:.0f}초)"
                    )

            chart_elapsed = time.time() - chart_start_time
            logger.info(
                f"  차트 생성 완료: {chart_elapsed:.1f}초 "
                f"(성공: {success_count}, 실패: {fail_count})"
            )

        # Step 3: NPY 및 메타데이터 저장
        ts_arr = np.stack(all_time_series).astype(np.float32)
        np.save(str(split_dir / "time_series.npy"), ts_arr)
        logger.info(f"  time_series: {ts_arr.shape}")

        sector_arr = np.array(all_sector_ids, dtype=np.int64)
        np.save(str(split_dir / "sector_ids.npy"), sector_arr)
        logger.info(f"  sector_ids: {sector_arr.shape}")

        label_arr = np.array(all_labels, dtype=np.int64)
        np.save(str(split_dir / "labels.npy"), label_arr)
        logger.info(f"  labels: {label_arr.shape}")

        # 메타데이터 CSV 저장
        metadata_df = pd.DataFrame(all_metadata)
        metadata_df.to_csv(split_dir / "metadata.csv", index=False)
        logger.info(f"  metadata: {len(metadata_df)}행")

        if generate_charts:
            logger.info(f"  charts: {split_dir / 'charts'}/ ({sample_count}개 PNG)")

        unique, counts = np.unique(label_arr, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(unique, counts)}
        logger.info(f"  label distribution: {dist}")

    logger.info("Phase 3 완료")


# ── 통계 출력 ──────────────────────────────────────────────

def print_db_stats(session: Session) -> None:
    """DB에 저장된 데이터 통계를 출력한다."""
    from lasps.db.models.daily_price import DailyPrice
    from lasps.db.models.technical_indicator import TechnicalIndicator
    from lasps.db.models.market_sentiment import MarketSentiment
    from lasps.db.models.training_label import TrainingLabel
    from lasps.db.models.investor_trading import InvestorTrading
    from lasps.db.models.short_selling import ShortSelling
    from lasps.db.models.stock import Stock

    stock_count = session.query(Stock).filter(Stock.is_active.is_(True)).count()
    price_count = session.query(DailyPrice).count()
    indicator_count = session.query(TechnicalIndicator).count()
    sentiment_count = session.query(MarketSentiment).count()
    label_count = session.query(TrainingLabel).count()
    investor_count = session.query(InvestorTrading).count()
    short_count = session.query(ShortSelling).count()

    logger.info("=" * 50)
    logger.info("DB 통계:")
    logger.info(f"  종목 수:       {stock_count:>10,}")
    logger.info(f"  가격 데이터:   {price_count:>10,}")
    logger.info(f"  투자자 데이터: {investor_count:>10,}")
    logger.info(f"  공매도 데이터: {short_count:>10,}")
    logger.info(f"  기술지표:      {indicator_count:>10,}")
    logger.info(f"  시장감성:      {sentiment_count:>10,}")
    logger.info(f"  학습라벨:      {label_count:>10,}")

    # 라벨 분포
    if label_count > 0:
        for split_name in ["train", "val", "test"]:
            count = session.query(TrainingLabel).filter(
                TrainingLabel.split == split_name
            ).count()
            logger.info(f"  라벨 ({split_name}): {count:>10,}")
    logger.info("=" * 50)


# ── 메인 ──────────────────────────────────────────────────

def main() -> None:
    """CLI 진입점: 3-Phase 파이프라인을 실행한다."""
    parser = argparse.ArgumentParser(
        description="LASPS v7a Historical Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python scripts/historical_data.py --mode mock                    # Mock 테스트
  python scripts/historical_data.py --mode real --days 2600        # 실제 전종목 10년치
  python scripts/historical_data.py --mode real --resume           # 중단 후 이어서
  python scripts/historical_data.py --phase 2                      # 파생 데이터만
  python scripts/historical_data.py --stocks 005930,000660         # 특정 종목만
        """,
    )

    parser.add_argument(
        "--mode", choices=["real", "mock"], default="mock",
        help="API 모드: real=실제 키움, mock=테스트용 (기본: mock)",
    )
    parser.add_argument(
        "--phase", type=int, choices=[1, 2, 3], default=None,
        help="특정 Phase만 실행 (미지정시 1→2 순차 실행)",
    )
    parser.add_argument(
        "--days", type=int, default=2600,
        help="수집할 거래일 수 (기본: 2600 ≈ 10년)",
    )
    parser.add_argument(
        "--stocks", type=str, default=None,
        help="쉼표 구분 종목코드 (예: 005930,000660)",
    )
    parser.add_argument(
        "--market", choices=["all", "kospi", "kosdaq"], default="all",
        help="수집 시장 (기본: all)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="이미 수집된 종목은 건너뛰고 이어서 수집",
    )
    parser.add_argument(
        "--export-npy", action="store_true",
        help="Phase 3 (NPY 내보내기)도 실행",
    )
    parser.add_argument(
        "--output", type=str, default="data/processed",
        help="NPY 출력 디렉토리 (기본: data/processed)",
    )
    parser.add_argument(
        "--stride", type=int, default=5,
        help="NPY 슬라이딩 윈도우 간격 (기본: 5)",
    )
    parser.add_argument(
        "--no-charts", action="store_true",
        help="NPY 내보내기 시 차트 이미지 생성 건너뛰기",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help=f"차트 생성 병렬 워커 수 (기본: CPU 코어 - 1 = {CHART_WORKERS})",
    )
    parser.add_argument(
        "--db-path", type=str, default=None,
        help="SQLite DB 경로 (기본: data/lasps.db)",
    )

    args = parser.parse_args()

    setup_logger("INFO")
    logger.info("LASPS v7a Historical Data Pipeline 시작")
    logger.info(f"모드: {args.mode}, 일수: {args.days}, 시장: {args.market}")

    # DB 설정
    if args.db_path:
        db_url = f"sqlite:///{args.db_path}"
    else:
        from lasps.config.settings import settings
        db_url = settings.DATABASE_URL

    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = session_factory()

    pipeline_start = time.time()

    try:
        run_phase_1 = args.phase is None or args.phase == 1
        run_phase_2 = args.phase is None or args.phase == 2
        run_phase_3 = args.export_npy and (args.phase is None or args.phase == 3)

        if run_phase_1:
            api, collector = create_api_and_collector(args.mode)
            stock_codes = resolve_stock_codes(
                collector, args.mode, args.stocks, args.market,
            )
            collect_raw_data(session, collector, stock_codes, args.days, args.resume, args.mode)

            # 실제 API 연결 해제
            if args.mode == "real" and hasattr(api, "disconnect"):
                api.disconnect()

        if run_phase_2:
            compute_derived_data(session)

        if run_phase_3:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            export_npy(session, output_dir, args.stride, args.no_charts, args.workers)

        # 최종 통계 출력
        print_db_stats(session)

    finally:
        session.close()

    elapsed = time.time() - pipeline_start
    logger.info(f"파이프라인 완료! (총 소요: {elapsed/60:.1f}분)")


if __name__ == "__main__":
    main()
