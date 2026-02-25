#!/usr/bin/env python3
"""2024년 기준 Top 10 알파 선정 + 투표 전략 시뮬레이션.

Phase 1: 전체 알파를 2024년 데이터로 평가 → Top 10 선정
Phase 2: 선정된 Top 10으로 투표 전략 시뮬레이션
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
import sqlite3
from collections import Counter
from loguru import logger
import warnings
import time

warnings.filterwarnings('ignore')

from scripts.simulate_trading import (
    Portfolio, select_voting, run_simulation, compute_metrics
)

DB_PATH = Path('data/lasps.db')
EVAL_START = '2024-01-01'
EVAL_END = '2024-12-31'
LOOKBACK_START = '2022-06-01'
MIN_TRADING_DAYS = 200
TOP_N = 5
REBALANCE_DAYS = 5
HOLDING_DAYS = 5
INITIAL_CAPITAL = 100_000_000


def main():
    logger.info("=" * 70)
    logger.info("Phase 1: 2024년 기준 Top 10 알파 선정")
    logger.info("=" * 70)
    t0 = time.time()

    # 데이터 로딩
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT stock_code, date, open, high, low, close, volume "
        "FROM daily_prices WHERE date >= ? AND date <= ? "
        "ORDER BY date, stock_code",
        conn, params=[LOOKBACK_START, EVAL_END]
    )
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    logger.info(f"  데이터: {len(df):,}건, {df['stock_code'].nunique()}개 종목")

    counts = df.groupby('stock_code').size()
    valid = counts[counts >= MIN_TRADING_DAYS].index
    df = df[df['stock_code'].isin(valid)].copy()
    logger.info(f"  필터 후: {len(valid)}개 종목")

    wide = {}
    for col in ['open', 'high', 'low', 'close', 'volume']:
        wide[col] = df.pivot(
            index='date', columns='stock_code', values=col
        ).sort_index()

    # 전체 알파 계산
    logger.info("\n  전체 알파 계산 중 (시간 소요)...")
    from lasps.data.processors.alpha101 import Alpha101Calculator
    calc = Alpha101Calculator(
        open_=wide['open'], high=wide['high'], low=wide['low'],
        close=wide['close'], volume=wide['volume'],
    )
    t1 = time.time()
    all_alphas = calc.compute_simple(skip_errors=True)
    valid_alphas = {k: v for k, v in all_alphas.items() if v is not None}
    logger.info(f"  계산 완료: {len(valid_alphas)}개 알파 ({time.time()-t1:.0f}초)")

    # 5일 리밸런싱 백테스트로 알파 순위
    close = wide['close']
    fwd_returns = close.shift(-HOLDING_DAYS) / close - 1
    eval_start = pd.Timestamp(EVAL_START)
    eval_end = pd.Timestamp(EVAL_END)
    all_dates = close.index
    eval_dates = all_dates[(all_dates >= eval_start) & (all_dates <= eval_end)]
    rebalance_dates = eval_dates[::REBALANCE_DAYS]

    logger.info(f"  평가기간: {eval_dates[0].date()} ~ {eval_dates[-1].date()}")
    logger.info(f"  리밸런싱: {len(rebalance_dates)}회")

    results_list = []
    for alpha_id, alpha_df in sorted(valid_alphas.items()):
        for reverse in [False, True]:
            period_returns = []
            for date in rebalance_dates:
                if date not in alpha_df.index or date not in fwd_returns.index:
                    continue
                row = alpha_df.loc[date].dropna()
                fwd_row = fwd_returns.loc[date]
                if len(row) < TOP_N:
                    continue
                top5 = row.sort_values(ascending=reverse).head(TOP_N).index
                rets = fwd_row[top5].dropna()
                if len(rets) > 0:
                    period_returns.append(rets.mean())

            if len(period_returns) == 0:
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
    )

    # Top 10 출력
    logger.info(f"\n{'='*70}")
    logger.info("2024년 기준 Top 10 알파")
    logger.info(f"{'='*70}")
    logger.info(
        f"{'순위':>4} {'알파':>16} {'누적수익률':>12} "
        f"{'복리수익률':>12} {'승률':>8} {'횟수':>6}"
    )
    logger.info("-" * 65)

    top10 = df_rank.head(10)
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
    logger.info(
        f"\n  선정: "
        f"{[(f'alpha_{a:03d}' + ('_rev' if r else '')) for a, r in top10_config]}"
    )

    # 2025 Top10과 비교
    top10_2025 = [
        'alpha_049_rev', 'alpha_083_rev', 'alpha_031_rev', 'alpha_038_rev',
        'alpha_010', 'alpha_013', 'alpha_018', 'alpha_045',
        'alpha_017', 'alpha_009',
    ]
    top10_2024_names = [
        f"alpha_{a:03d}" + ('_rev' if r else '') for a, r in top10_config
    ]
    overlap = set(top10_2025) & set(top10_2024_names)
    logger.info(f"\n  2025 Top10과 겹치는 알파: {len(overlap)}개")
    if overlap:
        logger.info(f"    {sorted(overlap)}")

    # ================================================================
    # Phase 2: 투표 시뮬레이션
    # ================================================================
    logger.info(f"\n{'='*70}")
    logger.info("Phase 2: 투표 전략 시뮬레이션 (2024 기준 Top 10)")
    logger.info(f"{'='*70}")

    alphas_for_sim = {}
    for aid, rev in top10_config:
        d = valid_alphas.get(aid)
        if d is None:
            continue
        name = f"alpha_{aid:03d}_rev" if rev else f"alpha_{aid:03d}"
        alphas_for_sim[name] = -d if rev else d

    close_wide = wide['close']
    volume_wide = wide['volume']

    # KOSPI
    conn = sqlite3.connect(DB_PATH)
    kospi = pd.read_sql(
        "SELECT date, close FROM index_prices "
        "WHERE index_code='1001' AND date >= ? AND date <= ? ORDER BY date",
        conn, params=[EVAL_START, EVAL_END]
    )
    conn.close()
    kospi['date'] = pd.to_datetime(kospi['date'])
    kospi = kospi.set_index('date')['close']

    def voting_fn(date, volume_row):
        return select_voting(alphas_for_sim, date, volume_row, TOP_N)

    pf = run_simulation(
        "voting_2024_v2", voting_fn,
        close_wide, volume_wide,
        rebalance_dates, eval_dates,
    )
    metrics = compute_metrics(pf, kospi)

    # 결과
    logger.info(f"\n{'='*70}")
    logger.info("투표 전략 결과 (2024 기준 Top 10)")
    logger.info(f"{'='*70}")
    logger.info(f"  초기 자본:    {INITIAL_CAPITAL:>18,.0f}원")
    logger.info(f"  최종 자산:    {metrics['final_value']:>18,.0f}원")
    logger.info(f"  총수익률:     {metrics['total_return_pct']:>+17.1f}%")
    logger.info(f"  CAGR:         {metrics['cagr_pct']:>+17.1f}%")
    logger.info(f"  Sharpe:       {metrics['sharpe']:>+17.2f}")
    logger.info(f"  MDD:          {metrics['mdd_pct']:>17.1f}%")
    logger.info(f"  승률:         {metrics['win_rate_pct']:>17.1f}%")
    logger.info(f"  회전율:       {metrics['turnover']:>17.1f}x")
    logger.info(f"  총 수수료:    {metrics['total_fees']:>18,.0f}원")
    logger.info(f"  KOSPI 수익률: {metrics['kospi_return_pct']:>+17.1f}%")
    logger.info(f"  초과수익률:   {metrics['excess_return_pct']:>+17.1f}%")

    # 월별
    daily_df = pd.DataFrame(pf.daily_values)
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df = daily_df.set_index('date')
    monthly = daily_df['total_value'].resample('ME').last()
    monthly_ret = monthly.pct_change()
    monthly_ret.iloc[0] = monthly.iloc[0] / INITIAL_CAPITAL - 1

    logger.info(f"\n{'='*70}")
    logger.info("월별 수익률")
    logger.info(f"{'='*70}")
    logger.info(f"  {'월':>8s}  {'수익률':>10s}  {'자산':>16s}")
    logger.info("-" * 42)
    for dt, ret in monthly_ret.items():
        val = monthly.loc[dt]
        logger.info(
            f"  {dt.strftime('%Y-%m'):>8s}  {ret*100:>+9.1f}%  {val:>15,.0f}원"
        )

    # 비교표
    logger.info(f"\n{'='*70}")
    logger.info("2025 Top10 vs 2024 Top10 투표 전략 비교 (2024년 시뮬)")
    logger.info(f"{'='*70}")
    logger.info(f"  {'지표':>12s}  {'2025 Top10':>12s}  {'2024 Top10':>12s}")
    logger.info("-" * 42)
    prev = {
        'total_return_pct': -35.7, 'sharpe': -1.62,
        'mdd_pct': -42.9, 'win_rate_pct': 48.1, 'kospi_return_pct': -10.1,
    }
    curr = metrics
    rows = [
        ('총수익률', f"{prev['total_return_pct']:+.1f}%",
         f"{curr['total_return_pct']:+.1f}%"),
        ('Sharpe', f"{prev['sharpe']:+.2f}",
         f"{curr['sharpe']:+.2f}"),
        ('MDD', f"{prev['mdd_pct']:.1f}%",
         f"{curr['mdd_pct']:.1f}%"),
        ('승률', f"{prev['win_rate_pct']:.1f}%",
         f"{curr['win_rate_pct']:.1f}%"),
        ('KOSPI', f"{prev['kospi_return_pct']:+.1f}%",
         f"{curr['kospi_return_pct']:+.1f}%"),
        ('초과수익',
         f"{prev['total_return_pct'] - prev['kospi_return_pct']:+.1f}%",
         f"{curr['excess_return_pct']:+.1f}%"),
    ]
    for label, v1, v2 in rows:
        logger.info(f"  {label:>12s}  {v1:>12s}  {v2:>12s}")

    logger.info(f"\n총 소요: {time.time()-t0:.0f}초")


if __name__ == "__main__":
    main()
