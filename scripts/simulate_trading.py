#!/usr/bin/env python3
"""Top 10 Alpha 실전 매매 시뮬레이션.

3가지 전략 비교:
  A. 개별 알파 (10개): 각 알파 독립적으로 Top 5 종목 선정
  B. 앙상블: 10개 알파 z-score 합산 → Top 5
  C. 투표: 각 알파 Top 5 후보 → 중복 많은 순 Top 5

설정:
  - 초기 자본: 1억원
  - 수수료: 매수 0.015%, 매도 0.015% + 거래세 0.23%
  - 리밸런싱: 5거래일 간격, 종목당 동일 비중
  - 평가 기간: 2025-02-20 ~ 2026-02-20
  - 벤치마크: KOSPI
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import Counter
from loguru import logger
import warnings
import time

warnings.filterwarnings('ignore')

# === Constants ===
DB_PATH = Path('data/lasps.db')
EVAL_START = '2025-02-20'
EVAL_END = '2026-02-20'
LOOKBACK_START = '2024-01-01'
MIN_TRADING_DAYS = 200
TOP_N = 5
REBALANCE_DAYS = 5

INITIAL_CAPITAL = 100_000_000  # 1억원
BUY_FEE_RATE = 0.00015        # 매수 수수료 0.015%
SELL_FEE_RATE = 0.00015       # 매도 수수료 0.015%
SELL_TAX_RATE = 0.0023        # 거래세 0.23%
RISK_FREE_RATE = 0.035        # 무위험이자율 3.5%

# Top 10 알파 (alpha_top10_allstocks.csv 기준)
TOP10_ALPHAS = [
    (49, True),   # alpha_049_rev
    (83, True),   # alpha_083_rev
    (31, True),   # alpha_031_rev
    (38, True),   # alpha_038_rev
    (10, False),  # alpha_010
    (13, False),  # alpha_013
    (18, False),  # alpha_018
    (45, False),  # alpha_045
    (17, False),  # alpha_017
    (9, False),   # alpha_009
]

OUTPUT_DIR = Path('data/simulation_results')


@dataclass
class Position:
    """보유 종목 정보."""
    stock_code: str
    shares: int
    avg_price: float
    buy_date: str


class Portfolio:
    """포트폴리오 시뮬레이션 클래스.

    현금, 보유종목, 매매기록, 일별 평가액을 추적한다.
    """

    def __init__(self, initial_capital: float):
        self.cash: float = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_log: List[dict] = []
        self.daily_values: List[dict] = []
        self.initial_capital = initial_capital

    def buy(self, stock_code: str, price: float, allocation: float,
            date: str) -> bool:
        """주식 매수.

        Args:
            stock_code: 종목코드
            price: 매수가격
            allocation: 배정금액
            date: 매수일

        Returns:
            매수 성공 여부
        """
        if price <= 0 or allocation <= 0:
            return False

        cost_per_share = price * (1 + BUY_FEE_RATE)
        shares = int(allocation / cost_per_share)
        if shares <= 0:
            return False

        total_cost = shares * price
        fee = int(total_cost * BUY_FEE_RATE)
        total_outlay = total_cost + fee

        if total_outlay > self.cash:
            shares = int(self.cash / cost_per_share)
            if shares <= 0:
                return False
            total_cost = shares * price
            fee = int(total_cost * BUY_FEE_RATE)
            total_outlay = total_cost + fee

        self.cash -= total_outlay
        self.positions[stock_code] = Position(
            stock_code=stock_code,
            shares=shares,
            avg_price=price,
            buy_date=date,
        )

        self.trade_log.append({
            'date': date,
            'stock_code': stock_code,
            'action': 'BUY',
            'shares': shares,
            'price': price,
            'fee': fee,
            'tax': 0,
            'total': total_outlay,
        })
        return True

    def sell(self, stock_code: str, price: float, date: str) -> bool:
        """주식 전량 매도.

        Args:
            stock_code: 종목코드
            price: 매도가격
            date: 매도일

        Returns:
            매도 성공 여부
        """
        if stock_code not in self.positions:
            return False

        pos = self.positions[stock_code]
        if price <= 0:
            return False

        gross = pos.shares * price
        fee = int(gross * SELL_FEE_RATE)
        tax = int(gross * SELL_TAX_RATE)
        net = gross - fee - tax

        self.cash += net
        pnl = net - (pos.shares * pos.avg_price)

        self.trade_log.append({
            'date': date,
            'stock_code': stock_code,
            'action': 'SELL',
            'shares': pos.shares,
            'price': price,
            'fee': fee,
            'tax': tax,
            'total': net,
            'pnl': pnl,
        })

        del self.positions[stock_code]
        return True

    def get_total_value(self, prices: Dict[str, float]) -> float:
        """현금 + 보유주식 시가평가 합계."""
        stock_value = sum(
            pos.shares * prices.get(pos.stock_code, pos.avg_price)
            for pos in self.positions.values()
        )
        return self.cash + stock_value

    def record_daily(self, date: str, prices: Dict[str, float]) -> None:
        """일별 평가액 기록."""
        total = self.get_total_value(prices)
        stock_value = total - self.cash
        self.daily_values.append({
            'date': date,
            'total_value': total,
            'cash': self.cash,
            'stock_value': stock_value,
            'n_positions': len(self.positions),
        })

    def validate(self, prices: Dict[str, float]) -> bool:
        """포트폴리오 일관성 검증."""
        assert self.cash >= 0, f"마이너스 현금: {self.cash}"
        total = self.get_total_value(prices)
        stock_value = sum(
            pos.shares * prices.get(pos.stock_code, pos.avg_price)
            for pos in self.positions.values()
        )
        expected = self.cash + stock_value
        assert abs(total - expected) < 1, f"불일치: {total} != {expected}"
        return True


# ============================================================
# Data Loading (evaluate_top_alphas_allstocks.py 패턴 재사용)
# ============================================================

def load_daily_prices() -> pd.DataFrame:
    """DB에서 전종목 일봉 데이터를 로드한다."""
    logger.info("DB에서 전종목 일봉 데이터 로딩 중...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT stock_code, date, open, high, low, close, volume "
        "FROM daily_prices WHERE date >= ? AND date <= ? "
        "ORDER BY date, stock_code",
        conn, params=[LOOKBACK_START, EVAL_END]
    )
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    logger.info(f"  로딩 완료: {len(df):,}건, "
                f"{df['stock_code'].nunique()}개 종목, "
                f"{df['date'].min().date()} ~ {df['date'].max().date()}")
    return df


def load_kospi_index() -> pd.Series:
    """KOSPI 지수 종가를 로드한다."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT date, close FROM index_prices "
        "WHERE index_code = '1001' AND date >= ? AND date <= ? "
        "ORDER BY date",
        conn, params=[EVAL_START, EVAL_END]
    )
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    series = df.set_index('date')['close']
    logger.info(f"  KOSPI 로딩: {len(series)}일, "
                f"{series.index.min().date()} ~ {series.index.max().date()}")
    return series


def filter_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """최소 거래일 + 유동성 기준으로 종목을 필터링한다."""
    counts = df.groupby('stock_code').size()
    valid_codes = counts[counts >= MIN_TRADING_DAYS].index
    filtered = df[df['stock_code'].isin(valid_codes)].copy()
    logger.info(f"  종목 필터: {df['stock_code'].nunique()}개 → "
                f"{len(valid_codes)}개 (최소 {MIN_TRADING_DAYS}거래일)")
    return filtered


def prepare_wide_data(df: pd.DataFrame) -> dict:
    """Wide-format DataFrame 생성."""
    wide = {}
    for col in ['open', 'high', 'low', 'close', 'volume']:
        pivot = df.pivot(index='date', columns='stock_code', values=col)
        pivot = pivot.sort_index()
        wide[col] = pivot
    logger.info(f"  Wide 데이터: {wide['close'].shape} (날짜 x 종목)")
    return wide


def compute_top10_alphas(wide: dict) -> Dict[str, pd.DataFrame]:
    """Top 10 알파만 계산한다.

    Returns:
        Dict[alpha_name, DataFrame] (역방향이면 부호 반전 포함)
    """
    from lasps.data.processors.alpha101 import Alpha101Calculator

    calc = Alpha101Calculator(
        open_=wide['open'],
        high=wide['high'],
        low=wide['low'],
        close=wide['close'],
        volume=wide['volume'],
    )

    alpha_ids = list(set(aid for aid, _ in TOP10_ALPHAS))
    logger.info(f"  알파 계산 중: {sorted(alpha_ids)} ({len(alpha_ids)}개)...")
    start = time.time()
    raw = calc.compute_batch(alpha_ids, skip_errors=True)
    logger.info(f"  알파 계산 완료 ({time.time() - start:.1f}초)")

    alphas = {}
    for aid, rev in TOP10_ALPHAS:
        df = raw.get(aid)
        if df is None:
            logger.warning(f"  alpha_{aid:03d} 계산 실패, 건너뜀")
            continue
        name = f"alpha_{aid:03d}_rev" if rev else f"alpha_{aid:03d}"
        alphas[name] = -df if rev else df

    logger.info(f"  유효 알파: {len(alphas)}개")
    return alphas


# ============================================================
# Stock Selection Strategies
# ============================================================

def select_individual(alpha_df: pd.DataFrame, date: pd.Timestamp,
                      volume_row: pd.Series, top_n: int = TOP_N) -> List[str]:
    """단일 알파 시그널 상위 Top N 종목 선정."""
    if date not in alpha_df.index:
        return []
    row = alpha_df.loc[date].dropna()
    # 거래량 0인 종목 제외
    tradeable = volume_row.dropna()
    tradeable = tradeable[tradeable > 0].index
    row = row[row.index.isin(tradeable)]
    if len(row) < top_n:
        return []
    return row.sort_values(ascending=False).head(top_n).index.tolist()


def select_ensemble(alphas: Dict[str, pd.DataFrame], date: pd.Timestamp,
                    volume_row: pd.Series, top_n: int = TOP_N) -> List[str]:
    """10개 알파 z-score 정규화 → 합산 → Top N."""
    tradeable = volume_row.dropna()
    tradeable = tradeable[tradeable > 0].index

    z_sum = None
    count = 0

    for name, alpha_df in alphas.items():
        if date not in alpha_df.index:
            continue
        row = alpha_df.loc[date].dropna()
        row = row[row.index.isin(tradeable)]
        if len(row) < 50:
            continue

        mean_v = row.mean()
        std_v = row.std()
        if std_v > 0:
            z = (row - mean_v) / std_v
        else:
            z = row * 0.0

        if z_sum is None:
            z_sum = z
        else:
            z_sum = z_sum.add(z, fill_value=0)
        count += 1

    if z_sum is None or count == 0:
        return []

    z_sum = z_sum.dropna()
    if len(z_sum) < top_n:
        return []
    return z_sum.sort_values(ascending=False).head(top_n).index.tolist()


def select_voting(alphas: Dict[str, pd.DataFrame], date: pd.Timestamp,
                  volume_row: pd.Series, top_n: int = TOP_N) -> List[str]:
    """각 알파 Top 5 후보 → 투표수 내림차순 → Top N."""
    tradeable = volume_row.dropna()
    tradeable = tradeable[tradeable > 0].index

    votes: Counter = Counter()

    for name, alpha_df in alphas.items():
        if date not in alpha_df.index:
            continue
        row = alpha_df.loc[date].dropna()
        row = row[row.index.isin(tradeable)]
        if len(row) < top_n:
            continue
        candidates = row.sort_values(ascending=False).head(top_n).index.tolist()
        votes.update(candidates)

    if not votes:
        return []

    # 투표수 내림차순, 동점이면 첫 등장 순서
    ranked = [code for code, _ in votes.most_common()]
    return ranked[:top_n]


# ============================================================
# Simulation Engine
# ============================================================

def run_simulation(
    strategy_name: str,
    select_fn,
    close_wide: pd.DataFrame,
    volume_wide: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    all_eval_dates: pd.DatetimeIndex,
) -> Portfolio:
    """단일 전략 시뮬레이션을 실행한다.

    Args:
        strategy_name: 전략 이름 (로깅용)
        select_fn: 종목 선정 함수 (date, volume_row) -> List[str]
        close_wide: 종가 wide DataFrame
        volume_wide: 거래량 wide DataFrame
        rebalance_dates: 리밸런싱 날짜들
        all_eval_dates: 전체 평가 기간 거래일

    Returns:
        Portfolio 객체
    """
    pf = Portfolio(INITIAL_CAPITAL)

    # 종가에서 ffill (최대 5일)
    close_filled = close_wide.ffill(limit=5)

    for date in all_eval_dates:
        date_str = str(date.date())
        prices_row = close_filled.loc[date].dropna()
        prices_dict = prices_row.to_dict()

        if date in rebalance_dates:
            volume_row = volume_wide.loc[date] if date in volume_wide.index else pd.Series()

            # 1) 전량 매도
            for code in list(pf.positions.keys()):
                price = prices_dict.get(code, 0)
                if price > 0:
                    pf.sell(code, price, date_str)

            # 2) 종목 선정
            selected = select_fn(date, volume_row)

            # 3) 동일 비중 매수
            if selected:
                allocation = pf.cash / len(selected)
                for code in selected:
                    price = prices_dict.get(code, 0)
                    if price > 0:
                        pf.buy(code, price, allocation, date_str)

        # 일별 기록
        pf.record_daily(date_str, prices_dict)
        pf.validate(prices_dict)

    # 마지막 날 전량 매도 (청산)
    last_date = str(all_eval_dates[-1].date())
    last_prices = close_filled.loc[all_eval_dates[-1]].dropna().to_dict()
    for code in list(pf.positions.keys()):
        price = last_prices.get(code, 0)
        if price > 0:
            pf.sell(code, price, last_date)

    return pf


# ============================================================
# Performance Metrics
# ============================================================

def compute_metrics(pf: Portfolio, kospi: pd.Series) -> dict:
    """포트폴리오 성과 지표를 계산한다."""
    df = pd.DataFrame(pf.daily_values)
    if df.empty:
        return {}

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # 일별 수익률
    daily_ret = df['total_value'].pct_change().dropna()
    total_value_series = df['total_value']

    # 총수익률
    total_return = (total_value_series.iloc[-1] / pf.initial_capital) - 1

    # 거래일 수
    n_days = len(total_value_series)

    # CAGR (연환산)
    years = n_days / 252
    if years > 0 and total_return > -1:
        cagr = (1 + total_return) ** (1 / years) - 1
    else:
        cagr = 0.0

    # Sharpe Ratio (연환산)
    if len(daily_ret) > 1 and daily_ret.std() > 0:
        ann_ret = daily_ret.mean() * 252
        ann_vol = daily_ret.std() * np.sqrt(252)
        sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol
    else:
        sharpe = 0.0

    # MDD
    cummax = total_value_series.cummax()
    drawdown = (total_value_series - cummax) / cummax
    mdd = drawdown.min()

    # 승률 (리밸런싱 기간별)
    trades_df = pd.DataFrame(pf.trade_log)
    if not trades_df.empty:
        sells = trades_df[trades_df['action'] == 'SELL']
        if 'pnl' in sells.columns and len(sells) > 0:
            win_rate = (sells['pnl'] > 0).mean()
        else:
            win_rate = 0.0
        total_fees = trades_df['fee'].sum() + trades_df['tax'].sum()
        n_trades = len(trades_df)
    else:
        win_rate = 0.0
        total_fees = 0
        n_trades = 0

    # 회전율 (매매대금 / 평균 자산)
    if not trades_df.empty:
        buys = trades_df[trades_df['action'] == 'BUY']
        buy_amount = (buys['shares'] * buys['price']).sum() if len(buys) > 0 else 0
        avg_value = total_value_series.mean()
        turnover = buy_amount / avg_value if avg_value > 0 else 0
    else:
        turnover = 0.0

    # vs KOSPI
    common_dates = df.index.intersection(kospi.index)
    if len(common_dates) >= 2:
        k_start = kospi.loc[common_dates[0]]
        k_end = kospi.loc[common_dates[-1]]
        kospi_return = (k_end / k_start) - 1
    else:
        kospi_return = 0.0

    return {
        'initial_capital': pf.initial_capital,
        'final_value': total_value_series.iloc[-1],
        'total_return_pct': total_return * 100,
        'cagr_pct': cagr * 100,
        'sharpe': sharpe,
        'mdd_pct': mdd * 100,
        'win_rate_pct': win_rate * 100,
        'turnover': turnover,
        'total_fees': total_fees,
        'n_trades': n_trades,
        'n_days': n_days,
        'kospi_return_pct': kospi_return * 100,
        'excess_return_pct': (total_return - kospi_return) * 100,
    }


# ============================================================
# Main
# ============================================================

def main():
    logger.info("=" * 70)
    logger.info("Top 10 Alpha 실전 매매 시뮬레이션")
    logger.info(f"기간: {EVAL_START} ~ {EVAL_END}")
    logger.info(f"초기자본: {INITIAL_CAPITAL:,.0f}원, 리밸런싱: {REBALANCE_DAYS}일")
    logger.info(f"수수료: 매수 {BUY_FEE_RATE*100:.3f}%, "
                f"매도 {SELL_FEE_RATE*100:.3f}% + 거래세 {SELL_TAX_RATE*100:.2f}%")
    logger.info("=" * 70)

    start_time = time.time()

    # === Step 1: 데이터 로딩 ===
    logger.info(f"\n{'='*70}")
    logger.info("Step 1: 데이터 로딩 및 필터링")
    logger.info(f"{'='*70}")

    df = load_daily_prices()
    df = filter_stocks(df)
    wide = prepare_wide_data(df)
    kospi = load_kospi_index()

    # === Step 2: 알파 계산 ===
    logger.info(f"\n{'='*70}")
    logger.info("Step 2: Top 10 알파 계산")
    logger.info(f"{'='*70}")

    alphas = compute_top10_alphas(wide)

    # === Step 3: 리밸런싱 날짜 ===
    close_wide = wide['close']
    volume_wide = wide['volume']
    eval_start = pd.Timestamp(EVAL_START)
    eval_end = pd.Timestamp(EVAL_END)
    all_dates = close_wide.index
    eval_dates = all_dates[(all_dates >= eval_start) & (all_dates <= eval_end)]
    rebalance_dates = eval_dates[::REBALANCE_DAYS]

    logger.info(f"\n  평가 기간: {eval_dates[0].date()} ~ {eval_dates[-1].date()}")
    logger.info(f"  거래일: {len(eval_dates)}일, 리밸런싱: {len(rebalance_dates)}회")

    # === Step 4: 시뮬레이션 실행 ===
    logger.info(f"\n{'='*70}")
    logger.info("Step 3: 시뮬레이션 실행")
    logger.info(f"{'='*70}")

    results = {}
    all_daily = {}

    # A) 개별 알파 10개
    for name, alpha_df in sorted(alphas.items()):
        logger.info(f"  [A] {name} 시뮬레이션 중...")

        def make_select_fn(adf):
            def fn(date, volume_row):
                return select_individual(adf, date, volume_row, TOP_N)
            return fn

        pf = run_simulation(
            name,
            make_select_fn(alpha_df),
            close_wide, volume_wide,
            rebalance_dates, eval_dates,
        )
        metrics = compute_metrics(pf, kospi)
        results[name] = metrics
        all_daily[name] = pf.daily_values

        # 매매기록 저장용
        if name == sorted(alphas.keys())[0]:
            first_trade_log = pf.trade_log

    # B) 앙상블
    logger.info(f"  [B] 앙상블 시뮬레이션 중...")

    def ensemble_fn(date, volume_row):
        return select_ensemble(alphas, date, volume_row, TOP_N)

    pf_ensemble = run_simulation(
        "ensemble",
        ensemble_fn,
        close_wide, volume_wide,
        rebalance_dates, eval_dates,
    )
    results['ensemble'] = compute_metrics(pf_ensemble, kospi)
    all_daily['ensemble'] = pf_ensemble.daily_values

    # C) 투표
    logger.info(f"  [C] 투표 시뮬레이션 중...")

    def voting_fn(date, volume_row):
        return select_voting(alphas, date, volume_row, TOP_N)

    pf_voting = run_simulation(
        "voting",
        voting_fn,
        close_wide, volume_wide,
        rebalance_dates, eval_dates,
    )
    results['voting'] = compute_metrics(pf_voting, kospi)
    all_daily['voting'] = pf_voting.daily_values

    # === Step 5: 결과 출력 ===
    logger.info(f"\n{'='*70}")
    logger.info("Step 4: 시뮬레이션 결과 비교")
    logger.info(f"{'='*70}")

    # 종목명 로딩
    conn = sqlite3.connect(DB_PATH)
    stock_names = pd.read_sql(
        "SELECT code, name FROM stocks", conn
    ).set_index('code')['name'].to_dict()
    conn.close()

    header = (f"{'전략':<18s} {'최종자산':>14s} {'수익률':>9s} {'CAGR':>8s} "
              f"{'Sharpe':>8s} {'MDD':>8s} {'승률':>7s} {'회전율':>7s} "
              f"{'수수료':>12s} {'vs KOSPI':>10s}")
    logger.info(f"\n{header}")
    logger.info("-" * 115)

    # KOSPI 수익률은 공통
    kospi_ret_pct = list(results.values())[0].get('kospi_return_pct', 0)

    summary_rows = []
    for name in sorted(results.keys()):
        m = results[name]
        if not m:
            continue

        display_name = name
        row_str = (
            f"{display_name:<18s} "
            f"{m['final_value']:>13,.0f}원 "
            f"{m['total_return_pct']:>+8.1f}% "
            f"{m['cagr_pct']:>+7.1f}% "
            f"{m['sharpe']:>+7.2f} "
            f"{m['mdd_pct']:>7.1f}% "
            f"{m['win_rate_pct']:>6.1f}% "
            f"{m['turnover']:>6.1f}x "
            f"{m['total_fees']:>11,.0f}원 "
            f"{m['excess_return_pct']:>+9.1f}%"
        )
        logger.info(row_str)

        summary_rows.append({'strategy': name, **m})

    logger.info("-" * 115)
    logger.info(f"  KOSPI: {kospi_ret_pct:+.1f}%")
    logger.info(f"  무위험이자율: {RISK_FREE_RATE*100:.1f}%")

    # === Step 6: CSV 저장 ===
    logger.info(f"\n{'='*70}")
    logger.info("Step 5: 결과 저장")
    logger.info(f"{'='*70}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) simulation_summary.csv
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / 'simulation_summary.csv'
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    logger.info(f"  요약: {summary_path}")

    # 2) daily_portfolio_values.csv
    daily_rows = []
    for strat_name, dvs in all_daily.items():
        for d in dvs:
            daily_rows.append({'strategy': strat_name, **d})
    daily_df = pd.DataFrame(daily_rows)
    daily_path = OUTPUT_DIR / 'daily_portfolio_values.csv'
    daily_df.to_csv(daily_path, index=False, encoding='utf-8-sig')
    logger.info(f"  일별 평가액: {daily_path}")

    # 3) trade_log.csv (앙상블 전략)
    if pf_ensemble.trade_log:
        trade_df = pd.DataFrame(pf_ensemble.trade_log)
        # 종목명 추가
        trade_df['stock_name'] = trade_df['stock_code'].map(
            lambda c: stock_names.get(c, '?')
        )
        trade_path = OUTPUT_DIR / 'trade_log.csv'
        trade_df.to_csv(trade_path, index=False, encoding='utf-8-sig')
        logger.info(f"  매매기록 (앙상블): {trade_path}")

    elapsed = time.time() - start_time
    logger.info(f"\n총 소요 시간: {elapsed:.0f}초")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
