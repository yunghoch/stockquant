#!/usr/bin/env python3
"""Backtest Alpha101 strategies.

This script:
1. Loads alpha signals and forward returns
2. Constructs Long-Short portfolios based on alpha deciles
3. Calculates performance metrics (Sharpe, Max Drawdown, etc.)
4. Compares single alphas vs combined strategy
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
from pathlib import Path
import sqlite3
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Constants
PREDICTION_HORIZON = 5  # 5-day holding period
TEST_START = '2024-01-01'
TEST_END = '2026-12-31'
TOP_PCT = 0.1  # Top 10% for long
BOTTOM_PCT = 0.1  # Bottom 10% for short

DATA_DIR = Path('data/alpha101')
DB_PATH = Path('data/lasps.db')

# Top alphas to backtest
TOP_ALPHAS = ['alpha_044', 'alpha_016', 'alpha_026', 'alpha_015', 'alpha_013']
REVERSE_ALPHAS = ['alpha_042', 'alpha_100', 'alpha_048', 'alpha_094']  # Negative IC


def load_alphas(alpha_names: list) -> dict:
    """Load specified alpha parquet files."""
    logger.info(f"Loading {len(alpha_names)} alphas...")

    alphas = {}
    for name in alpha_names:
        path = DATA_DIR / f'{name}.parquet'
        if path.exists():
            df = pd.read_parquet(path)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            alphas[name] = df
        else:
            logger.warning(f"Alpha file not found: {path}")

    return alphas


def compute_forward_returns(horizon: int = 5) -> pd.DataFrame:
    """Compute forward returns from daily_prices table."""
    logger.info(f"Computing {horizon}-day forward returns...")

    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT stock_code, date, close
    FROM daily_prices
    ORDER BY stock_code, date
    """
    df = pd.read_sql(query, conn)
    conn.close()

    df['date'] = pd.to_datetime(df['date'])
    close_pivot = df.pivot(index='date', columns='stock_code', values='close')
    forward_returns = close_pivot.shift(-horizon) / close_pivot - 1

    return forward_returns


def backtest_single_alpha(alpha_df: pd.DataFrame, returns_df: pd.DataFrame,
                          reverse: bool = False) -> pd.DataFrame:
    """Backtest a single alpha using Long-Short strategy.

    Strategy:
    - Long: Top 10% alpha stocks
    - Short: Bottom 10% alpha stocks
    - Equal weight within each leg
    - Rebalance daily (with 5-day holding)

    Returns:
        DataFrame with daily portfolio returns
    """
    # Align dates
    common_dates = alpha_df.index.intersection(returns_df.index)
    common_stocks = alpha_df.columns.intersection(returns_df.columns)

    alpha_aligned = alpha_df.loc[common_dates, common_stocks]
    returns_aligned = returns_df.loc[common_dates, common_stocks]

    if reverse:
        alpha_aligned = -alpha_aligned  # Reverse signal for negative IC alphas

    daily_returns = []

    for i, date in enumerate(common_dates[:-PREDICTION_HORIZON]):
        alpha_row = alpha_aligned.loc[date].dropna()

        if len(alpha_row) < 50:  # Need enough stocks
            continue

        # Get top and bottom deciles
        n_stocks = len(alpha_row)
        n_long = max(1, int(n_stocks * TOP_PCT))
        n_short = max(1, int(n_stocks * BOTTOM_PCT))

        sorted_stocks = alpha_row.sort_values(ascending=False)
        long_stocks = sorted_stocks.head(n_long).index
        short_stocks = sorted_stocks.tail(n_short).index

        # Get 5-day forward return
        future_date = common_dates[i + PREDICTION_HORIZON]

        # Calculate portfolio return
        long_ret = returns_aligned.loc[date, long_stocks].mean()
        short_ret = returns_aligned.loc[date, short_stocks].mean()

        if pd.notna(long_ret) and pd.notna(short_ret):
            portfolio_ret = (long_ret - short_ret) / 2  # Long-Short
            long_only_ret = long_ret

            daily_returns.append({
                'date': date,
                'long_short': portfolio_ret,
                'long_only': long_only_ret,
                'short_only': -short_ret,
                'n_long': n_long,
                'n_short': n_short
            })

    return pd.DataFrame(daily_returns).set_index('date')


def backtest_combined_alpha(alphas: dict, returns_df: pd.DataFrame,
                            weights: dict = None) -> pd.DataFrame:
    """Backtest combined alpha strategy.

    Combines multiple alphas by averaging their signals (or weighted average).
    """
    logger.info("Backtesting combined alpha strategy...")

    # Get common dates and stocks
    all_dates = None
    all_stocks = None

    for name, df in alphas.items():
        if all_dates is None:
            all_dates = set(df.index)
            all_stocks = set(df.columns)
        else:
            all_dates &= set(df.index)
            all_stocks &= set(df.columns)

    all_dates = sorted(all_dates)
    all_stocks = sorted(all_stocks)

    # Combine alphas
    combined = pd.DataFrame(0, index=all_dates, columns=all_stocks)

    if weights is None:
        weights = {name: 1.0 for name in alphas.keys()}

    total_weight = sum(weights.values())

    for name, df in alphas.items():
        # Handle reverse alphas
        if name in REVERSE_ALPHAS:
            combined += -df.loc[all_dates, all_stocks] * weights[name] / total_weight
        else:
            combined += df.loc[all_dates, all_stocks] * weights[name] / total_weight

    return backtest_single_alpha(combined, returns_df, reverse=False)


def calculate_metrics(returns_series: pd.Series) -> dict:
    """Calculate performance metrics."""
    # Annualization factor (assuming 5-day holding, ~50 trades per year)
    ann_factor = 252 / PREDICTION_HORIZON

    total_return = (1 + returns_series).prod() - 1
    ann_return = (1 + total_return) ** (ann_factor / len(returns_series)) - 1

    volatility = returns_series.std() * np.sqrt(ann_factor)
    sharpe = ann_return / volatility if volatility > 0 else 0

    # Max Drawdown
    cumulative = (1 + returns_series).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Win rate
    win_rate = (returns_series > 0).mean()

    # Profit factor
    gains = returns_series[returns_series > 0].sum()
    losses = abs(returns_series[returns_series < 0].sum())
    profit_factor = gains / losses if losses > 0 else np.inf

    return {
        'total_return': total_return,
        'ann_return': ann_return,
        'volatility': volatility,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'n_trades': len(returns_series)
    }


def main():
    logger.info("=" * 70)
    logger.info("Alpha101 Backtest")
    logger.info("=" * 70)

    # Load data
    all_alphas = TOP_ALPHAS + REVERSE_ALPHAS
    alphas = load_alphas(all_alphas)
    returns = compute_forward_returns(PREDICTION_HORIZON)

    # Filter test period
    returns = returns[(returns.index >= TEST_START) & (returns.index <= TEST_END)]
    logger.info(f"Test period: {returns.index.min()} ~ {returns.index.max()} ({len(returns)} days)")

    # Backtest individual alphas
    logger.info("\n" + "=" * 70)
    logger.info("Individual Alpha Backtest Results")
    logger.info("=" * 70)

    results = {}

    for alpha_name in all_alphas:
        if alpha_name not in alphas:
            continue

        alpha_df = alphas[alpha_name]
        alpha_df = alpha_df[(alpha_df.index >= TEST_START) & (alpha_df.index <= TEST_END)]

        reverse = alpha_name in REVERSE_ALPHAS
        bt_result = backtest_single_alpha(alpha_df, returns, reverse=reverse)

        if len(bt_result) > 0:
            metrics = calculate_metrics(bt_result['long_short'])
            metrics['name'] = alpha_name
            metrics['reverse'] = reverse
            results[alpha_name] = {
                'returns': bt_result,
                'metrics': metrics
            }

    # Print results table
    logger.info(f"\n{'Alpha':<12} {'Total%':>8} {'Ann%':>8} {'Sharpe':>8} {'MDD%':>8} {'Win%':>8} {'PF':>8}")
    logger.info("-" * 76)

    for name, data in sorted(results.items(), key=lambda x: x[1]['metrics']['sharpe'], reverse=True):
        m = data['metrics']
        logger.info(
            f"{name:<12} "
            f"{m['total_return']*100:>+7.1f}% "
            f"{m['ann_return']*100:>+7.1f}% "
            f"{m['sharpe']:>+8.2f} "
            f"{m['max_drawdown']*100:>7.1f}% "
            f"{m['win_rate']*100:>7.1f}% "
            f"{m['profit_factor']:>8.2f}"
        )

    # Backtest combined strategy (Top 3)
    logger.info("\n" + "=" * 70)
    logger.info("Combined Strategy Backtest (alpha_044 + alpha_016 + alpha_026)")
    logger.info("=" * 70)

    top3_alphas = {k: v for k, v in alphas.items() if k in ['alpha_044', 'alpha_016', 'alpha_026']}

    if len(top3_alphas) == 3:
        combined_result = backtest_combined_alpha(top3_alphas, returns)
        combined_metrics = calculate_metrics(combined_result['long_short'])

        logger.info(f"\nCombined Strategy Metrics:")
        logger.info(f"  Total Return:    {combined_metrics['total_return']*100:+.2f}%")
        logger.info(f"  Annual Return:   {combined_metrics['ann_return']*100:+.2f}%")
        logger.info(f"  Sharpe Ratio:    {combined_metrics['sharpe']:+.2f}")
        logger.info(f"  Max Drawdown:    {combined_metrics['max_drawdown']*100:.2f}%")
        logger.info(f"  Win Rate:        {combined_metrics['win_rate']*100:.1f}%")
        logger.info(f"  Profit Factor:   {combined_metrics['profit_factor']:.2f}")
        logger.info(f"  # Trades:        {combined_metrics['n_trades']}")

        # Save cumulative returns
        cumulative = (1 + combined_result['long_short']).cumprod()
        cumulative.to_csv('data/alpha101_backtest_cumulative.csv')
        logger.info(f"\nCumulative returns saved to data/alpha101_backtest_cumulative.csv")

    # Long-only comparison
    logger.info("\n" + "=" * 70)
    logger.info("Long-Only Strategy Comparison")
    logger.info("=" * 70)

    logger.info(f"\n{'Alpha':<12} {'Total%':>8} {'Ann%':>8} {'Sharpe':>8} {'MDD%':>8}")
    logger.info("-" * 50)

    for name, data in sorted(results.items(), key=lambda x: calculate_metrics(x[1]['returns']['long_only'])['sharpe'], reverse=True):
        m = calculate_metrics(data['returns']['long_only'])
        logger.info(
            f"{name:<12} "
            f"{m['total_return']*100:>+7.1f}% "
            f"{m['ann_return']*100:>+7.1f}% "
            f"{m['sharpe']:>+8.2f} "
            f"{m['max_drawdown']*100:>7.1f}%"
        )

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    best_single = max(results.items(), key=lambda x: x[1]['metrics']['sharpe'])
    logger.info(f"\nBest Single Alpha: {best_single[0]}")
    logger.info(f"  Sharpe: {best_single[1]['metrics']['sharpe']:.2f}")
    logger.info(f"  Annual Return: {best_single[1]['metrics']['ann_return']*100:.1f}%")

    if len(top3_alphas) == 3:
        logger.info(f"\nCombined Strategy (Top 3):")
        logger.info(f"  Sharpe: {combined_metrics['sharpe']:.2f}")
        logger.info(f"  Annual Return: {combined_metrics['ann_return']*100:.1f}%")


if __name__ == "__main__":
    main()
