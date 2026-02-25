#!/usr/bin/env python3
"""Alpha101 일일 통합 분석 스크립트.

1) pykrx로 DB 업데이트 (daily_prices + index_prices)
2) 전체 알파 계산 → 1년 백테스트 → Top 10 알파 선정
3) Top 10 알파 투표 → 최종 Top 10 종목 선정
4) 마크다운 리포트 저장 (docs/result/YYYY-MM-DD_alpha.md)

Usage:
    python scripts/daily_alpha.py
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
import sqlite3
import time
from collections import Counter
from datetime import datetime, timedelta
from loguru import logger
import warnings

warnings.filterwarnings('ignore')

# === 설정 (동적) ===
TODAY = datetime.now().strftime("%Y-%m-%d")
EVAL_START = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
EVAL_END = TODAY
LOOKBACK_START = (datetime.now() - timedelta(days=int(365 * 2.5))).strftime("%Y-%m-%d")
MIN_TRADING_DAYS = 200
TOP_N_STOCKS = 10
TOP_N_PER_ALPHA = 10
TOP_N_ALPHAS = 10
REBALANCE_DAYS = 5
HOLDING_DAYS = 5
DB_PATH = Path('data/lasps.db')
REPORT_DIR = Path('docs/result')


def update_db(today: str) -> dict:
    """Step 1: pykrx로 daily_prices + index_prices 업데이트.

    Args:
        today: 오늘 날짜 (YYYY-MM-DD)

    Returns:
        dict with 'period', 'daily_inserted', 'index_inserted'
    """
    from pykrx import stock

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # daily_prices 마지막 날짜
    cursor.execute("SELECT MAX(date) FROM daily_prices")
    last_daily = cursor.fetchone()[0]

    # index_prices 마지막 날짜
    cursor.execute("SELECT MAX(date) FROM index_prices")
    last_index = cursor.fetchone()[0]

    last_date = min(last_daily or '2015-01-01', last_index or '2015-01-01')
    start_date = (pd.Timestamp(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")

    if start_date > today:
        logger.info(f"  DB 이미 최신 (마지막: {last_date})")
        conn.close()
        return {'period': f'{last_date} (최신)', 'daily_inserted': 0, 'index_inserted': 0}

    logger.info(f"  DB 업데이트: {start_date} ~ {today}")

    # 날짜 범위 생성
    dates = pd.date_range(start=start_date, end=today, freq='B')  # 영업일
    daily_inserted = 0
    index_inserted = 0

    for i, date in enumerate(dates):
        date_str = date.strftime("%Y-%m-%d")
        pykrx_date = date_str.replace('-', '')

        try:
            # --- daily_prices ---
            df = stock.get_market_ohlcv_by_ticker(pykrx_date, market='ALL')
            if df is not None and len(df) > 0:
                for ticker in df.index:
                    row = df.loc[ticker]
                    if pd.isna(row['종가']) or int(row['종가']) == 0:
                        continue
                    try:
                        cursor.execute(
                            "INSERT OR IGNORE INTO daily_prices "
                            "(stock_code, date, open, high, low, close, volume) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (ticker, date_str,
                             int(row['시가']), int(row['고가']),
                             int(row['저가']), int(row['종가']),
                             int(row['거래량']))
                        )
                        daily_inserted += cursor.rowcount
                    except Exception:
                        pass

            # --- index_prices ---
            for idx_code, idx_name in [('1001', 'KOSPI'), ('2001', 'KOSDAQ')]:
                try:
                    df_idx = stock.get_index_ohlcv(pykrx_date, pykrx_date, idx_code)
                    if df_idx is not None and len(df_idx) > 0:
                        r = df_idx.iloc[0]
                        cursor.execute(
                            "INSERT OR IGNORE INTO index_prices "
                            "(index_code, index_name, date, open, high, low, close, volume, trading_value) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (idx_code, idx_name, date_str,
                             float(r['시가']), float(r['고가']),
                             float(r['저가']), float(r['종가']),
                             int(r['거래량']), int(r['거래대금']))
                        )
                        index_inserted += cursor.rowcount
                except Exception:
                    pass

            if (i + 1) % 10 == 0:
                conn.commit()
                logger.info(f"  [{i+1}/{len(dates)}] {date_str}: daily +{daily_inserted}, index +{index_inserted}")

        except Exception as e:
            logger.warning(f"  {date_str} 스킵: {e}")

        time.sleep(0.5)

    conn.commit()
    conn.close()

    logger.info(f"  DB 업데이트 완료: daily +{daily_inserted}건, index +{index_inserted}건")
    return {
        'period': f'{start_date} ~ {today}',
        'daily_inserted': daily_inserted,
        'index_inserted': index_inserted,
    }


def load_data(today: str) -> tuple:
    """Step 2: DB에서 데이터 로드 + 필터.

    Args:
        today: 오늘 날짜

    Returns:
        (wide, df) - wide format dict와 원본 DataFrame
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT stock_code, date, open, high, low, close, volume "
        "FROM daily_prices WHERE date >= ? AND date <= ? "
        "ORDER BY date, stock_code",
        conn, params=[LOOKBACK_START, today]
    )
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    logger.info(f"  데이터: {len(df):,}건, {df['stock_code'].nunique()}개 종목")

    # 거래일 200일 미만 종목 필터링
    counts = df.groupby('stock_code').size()
    valid = counts[counts >= MIN_TRADING_DAYS].index
    df = df[df['stock_code'].isin(valid)].copy()
    logger.info(f"  필터 후: {len(valid)}개 종목")

    # Wide format pivot
    wide = {}
    for col in ['open', 'high', 'low', 'close', 'volume']:
        wide[col] = df.pivot(index='date', columns='stock_code', values=col).sort_index()

    return wide, df


def compute_alphas(wide: dict) -> dict:
    """Step 3: 전체 알파 계산.

    Args:
        wide: wide format dict (open, high, low, close, volume)

    Returns:
        valid_alphas dict (alpha_id -> DataFrame)
    """
    from lasps.data.processors.alpha101 import Alpha101Calculator

    calc = Alpha101Calculator(
        open_=wide['open'], high=wide['high'], low=wide['low'],
        close=wide['close'], volume=wide['volume'],
    )
    t1 = time.time()
    all_alphas = calc.compute_simple(skip_errors=True)
    valid_alphas = {k: v for k, v in all_alphas.items() if v is not None}
    logger.info(f"  계산 완료: {len(valid_alphas)}개 알파 ({time.time()-t1:.0f}초)")
    return valid_alphas


def evaluate_alphas(valid_alphas: dict, wide: dict) -> tuple:
    """Step 4: 알파 평가 + Top 10 선정.

    Args:
        valid_alphas: alpha_id -> DataFrame
        wide: wide format dict

    Returns:
        (df_rank, top10_config) - 전체 순위 DataFrame, Top 10 설정 리스트
    """
    close = wide['close']
    fwd_returns = close.shift(-HOLDING_DAYS) / close - 1
    eval_start = pd.Timestamp(EVAL_START)
    eval_end = pd.Timestamp(EVAL_END)
    all_dates = close.index
    eval_dates = all_dates[(all_dates >= eval_start) & (all_dates <= eval_end)]
    rebalance_dates = eval_dates[::REBALANCE_DAYS]

    logger.info(f"  평가: {eval_dates[0].date()} ~ {eval_dates[-1].date()}, "
                f"{len(eval_dates)}일, 리밸런싱 {len(rebalance_dates)}회")

    results_list = []
    for alpha_id, alpha_df in sorted(valid_alphas.items()):
        for reverse in [False, True]:
            period_returns = []
            for date in rebalance_dates:
                if date not in alpha_df.index or date not in fwd_returns.index:
                    continue
                row = alpha_df.loc[date].dropna()
                fwd_row = fwd_returns.loc[date]
                if len(row) < TOP_N_PER_ALPHA:
                    continue
                top_n = row.sort_values(ascending=reverse).head(TOP_N_PER_ALPHA).index
                rets = fwd_row[top_n].dropna()
                if len(rets) > 0:
                    period_returns.append(rets.mean())

            if not period_returns:
                continue

            cum_ret = sum(period_returns)
            compound = np.prod([1 + r for r in period_returns]) - 1
            win_rate = np.mean([r > 0 for r in period_returns])
            suffix = '_rev' if reverse else ''
            results_list.append({
                'alpha_id': alpha_id,
                'alpha_name': f'alpha_{alpha_id:03d}{suffix}',
                'reversed': reverse,
                'cumulative_return_pct': cum_ret * 100,
                'compound_return_pct': compound * 100,
                'win_rate_pct': win_rate * 100,
                'n_periods': len(period_returns),
            })

    df_rank = pd.DataFrame(results_list).sort_values(
        'cumulative_return_pct', ascending=False
    ).reset_index(drop=True)

    # Top 10 출력
    top10 = df_rank.head(TOP_N_ALPHAS)
    logger.info(f"\n{'='*70}")
    logger.info(f"Top {TOP_N_ALPHAS} 알파 ({EVAL_START} ~ {EVAL_END} 기준)")
    logger.info(f"{'='*70}")
    logger.info(
        f"{'순위':>4} {'알파':>16} {'누적수익률':>12} "
        f"{'복리수익률':>12} {'승률':>8} {'횟수':>6}"
    )
    logger.info("-" * 65)

    for i, (_, row) in enumerate(top10.iterrows(), 1):
        logger.info(
            f"{i:>4} {row['alpha_name']:>16} "
            f"{row['cumulative_return_pct']:>+11.1f}% "
            f"{row['compound_return_pct']:>+11.1f}% "
            f"{row['win_rate_pct']:>7.0f}% "
            f"{row['n_periods']:>5.0f}"
        )

    top10_config = [
        (int(row['alpha_id']), bool(row['reversed']))
        for _, row in top10.iterrows()
    ]

    return df_rank, top10_config


def select_stocks_by_voting(
    valid_alphas: dict,
    top10_config: list,
    wide: dict,
    today: str,
) -> dict:
    """Step 5: 투표 종목 선정.

    Args:
        valid_alphas: alpha_id -> DataFrame
        top10_config: [(alpha_id, reversed), ...] Top 10 설정
        wide: wide format dict
        today: 오늘 날짜

    Returns:
        dict with 'alpha_picks', 'votes', 'final_picks', 'stock_info'
    """
    close = wide['close']

    # 알파 준비
    alphas_for_vote = {}
    for aid, rev in top10_config:
        d = valid_alphas.get(aid)
        if d is None:
            continue
        name = f"alpha_{aid:03d}_rev" if rev else f"alpha_{aid:03d}"
        alphas_for_vote[name] = -d if rev else d

    # 오늘 날짜 (또는 가장 최근 거래일)
    today_ts = pd.Timestamp(today)
    if today_ts not in close.index:
        today_ts = close.index[close.index <= today_ts][-1]
        logger.info(f"  (오늘 데이터 없음, 최근 거래일 사용: {today_ts.date()})")

    volume_row = wide['volume'].loc[today_ts] if today_ts in wide['volume'].index else pd.Series()
    tradeable = volume_row.dropna()
    tradeable = tradeable[tradeable > 0].index

    # 각 알파별 Top 10 + 투표
    votes = Counter()
    alpha_picks = {}

    for name, alpha_df in alphas_for_vote.items():
        if today_ts not in alpha_df.index:
            continue
        row = alpha_df.loc[today_ts].dropna()
        row = row[row.index.isin(tradeable)]
        if len(row) < TOP_N_PER_ALPHA:
            continue
        picks = row.sort_values(ascending=False).head(TOP_N_PER_ALPHA).index.tolist()
        alpha_picks[name] = picks
        votes.update(picks)

    # 종목명 로딩
    conn = sqlite3.connect(DB_PATH)
    stock_names = pd.read_sql(
        "SELECT code, name FROM stocks", conn
    ).set_index('code')['name'].to_dict()

    # 최종 Top N 종목
    final_picks = [code for code, _ in votes.most_common()][:TOP_N_STOCKS]

    # 종목별 상세 정보
    stock_info = []
    for code in final_picks:
        price_row = pd.read_sql(
            "SELECT close, volume FROM daily_prices "
            "WHERE stock_code = ? AND date = ? LIMIT 1",
            conn, params=[code, str(today_ts.date())]
        )
        price = int(price_row['close'].iloc[0]) if len(price_row) > 0 else 0
        vol = int(price_row['volume'].iloc[0]) if len(price_row) > 0 else 0
        recommending = [a for a, picks in alpha_picks.items() if code in picks]
        stock_info.append({
            'code': code,
            'name': stock_names.get(code, '?'),
            'votes': votes[code],
            'price': price,
            'volume': vol,
            'alphas': recommending,
        })

    conn.close()

    # 콘솔 출력
    logger.info(f"\n{'='*70}")
    logger.info(f"투표 전략 종목 선정 ({today})")
    logger.info(f"{'='*70}")

    logger.info(f"\n  [각 알파별 Top {TOP_N_PER_ALPHA} 추천]")
    for name in sorted(alpha_picks.keys()):
        picks = alpha_picks[name]
        picks_str = ', '.join(f"{c}({stock_names.get(c, '?')})" for c in picks)
        logger.info(f"  {name:>16}: {picks_str}")

    logger.info(f"\n  [투표 집계]")
    for code, count in votes.most_common(20):
        name = stock_names.get(code, '?')
        bar = '#' * count
        logger.info(f"  {code}({name:>10}) {count:>2}표 {bar}")

    logger.info(f"\n{'='*70}")
    logger.info(f"  최종 매수 추천 ({today}, 투표 Top {TOP_N_STOCKS})")
    logger.info(f"{'='*70}")
    for i, info in enumerate(stock_info, 1):
        logger.info(f"  {i}. {info['code']} ({info['name']}) - {info['votes']}표")
        logger.info(f"     현재가: {info['price']:,}원, 거래량: {info['volume']:,}")
        logger.info(f"     추천 알파: {', '.join(info['alphas'])}")

    return {
        'alpha_picks': alpha_picks,
        'votes': votes,
        'final_picks': final_picks,
        'stock_info': stock_info,
        'stock_names': stock_names,
        'today_ts': today_ts,
    }


def generate_report(
    today: str,
    db_result: dict,
    df_rank: pd.DataFrame,
    top10_config: list,
    voting_result: dict,
    elapsed: float,
) -> Path:
    """Step 6: 마크다운 리포트 생성.

    Args:
        today: 오늘 날짜
        db_result: DB 업데이트 결과
        df_rank: 알파 순위 DataFrame
        top10_config: Top 10 알파 설정
        voting_result: 투표 결과
        elapsed: 총 소요시간(초)

    Returns:
        리포트 파일 경로
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"{today}_alpha.md"

    stock_names = voting_result['stock_names']
    alpha_picks = voting_result['alpha_picks']
    votes = voting_result['votes']
    stock_info = voting_result['stock_info']

    lines = []
    lines.append(f"# Alpha101 일일 분석 리포트 ({today})")
    lines.append("")

    # 1. DB 업데이트
    lines.append("## 1. DB 업데이트")
    lines.append(f"- 업데이트 기간: {db_result['period']}")
    lines.append(f"- 추가된 레코드: daily {db_result['daily_inserted']:,}건, index {db_result['index_inserted']:,}건")
    lines.append("")

    # 2. Top 10 알파
    lines.append(f"## 2. Top {TOP_N_ALPHAS} 알파 (평가기간: {EVAL_START} ~ {EVAL_END})")
    lines.append("")
    lines.append("| 순위 | 알파 | 누적수익률 | 복리수익률 | 승률 | 횟수 |")
    lines.append("|------|------|-----------|-----------|------|------|")

    top10 = df_rank.head(TOP_N_ALPHAS)
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        lines.append(
            f"| {i} | {row['alpha_name']} | "
            f"{row['cumulative_return_pct']:+.1f}% | "
            f"{row['compound_return_pct']:+.1f}% | "
            f"{row['win_rate_pct']:.0f}% | "
            f"{row['n_periods']:.0f} |"
        )
    lines.append("")

    # 3. 각 알파별 추천 종목
    lines.append("## 3. 각 알파별 추천 종목")
    lines.append("")
    lines.append("| 알파 | 추천 종목 |")
    lines.append("|------|----------|")

    for name in sorted(alpha_picks.keys()):
        picks = alpha_picks[name]
        picks_str = ', '.join(f"{c}({stock_names.get(c, '?')})" for c in picks)
        lines.append(f"| {name} | {picks_str} |")
    lines.append("")

    # 4. 투표 집계
    lines.append("## 4. 투표 집계")
    lines.append("")
    lines.append("| 종목코드 | 종목명 | 득표 |")
    lines.append("|---------|--------|------|")

    for code, count in votes.most_common(20):
        name = stock_names.get(code, '?')
        lines.append(f"| {code} | {name} | {count}표 |")
    lines.append("")

    # 5. 최종 매수 추천
    lines.append(f"## 5. 최종 매수 추천 (Top {TOP_N_STOCKS})")
    lines.append("")
    lines.append("| 순위 | 종목코드 | 종목명 | 득표 | 현재가 | 거래량 | 추천 알파 |")
    lines.append("|------|---------|--------|------|--------|--------|----------|")

    for i, info in enumerate(stock_info, 1):
        alphas_str = ', '.join(info['alphas'])
        lines.append(
            f"| {i} | {info['code']} | {info['name']} | "
            f"{info['votes']}표 | {info['price']:,} | "
            f"{info['volume']:,} | {alphas_str} |"
        )
    lines.append("")

    # Footer
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append("---")
    lines.append(f"*생성: {now}*  ")
    lines.append(f"*소요시간: {elapsed:.0f}초*")
    lines.append("")

    report_path.write_text('\n'.join(lines), encoding='utf-8')
    logger.info(f"  리포트 저장: {report_path}")
    return report_path


def main():
    t0 = time.time()
    today = datetime.now().strftime("%Y-%m-%d")

    logger.info("=" * 70)
    logger.info(f"Alpha101 일일 통합 분석 ({today})")
    logger.info(f"평가기간: {EVAL_START} ~ {EVAL_END}")
    logger.info(f"LOOKBACK: {LOOKBACK_START}")
    logger.info("=" * 70)

    # Step 1: DB 업데이트
    logger.info("\n[Step 1] DB 업데이트")
    db_result = update_db(today)

    # Step 2: 데이터 로드
    logger.info("\n[Step 2] 데이터 로드")
    wide, df = load_data(today)

    # Step 3: 알파 계산
    logger.info("\n[Step 3] 전체 알파 계산")
    valid_alphas = compute_alphas(wide)

    # Step 4: 알파 평가 + Top 10
    logger.info("\n[Step 4] 알파 평가")
    df_rank, top10_config = evaluate_alphas(valid_alphas, wide)

    # Step 5: 투표 종목 선정
    logger.info("\n[Step 5] 투표 종목 선정")
    voting_result = select_stocks_by_voting(valid_alphas, top10_config, wide, today)

    # Step 6: 리포트 생성
    elapsed = time.time() - t0
    logger.info("\n[Step 6] 리포트 생성")
    report_path = generate_report(today, db_result, df_rank, top10_config, voting_result, elapsed)

    logger.info(f"\n총 소요: {elapsed:.0f}초")
    logger.info(f"리포트: {report_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
