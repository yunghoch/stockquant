#!/usr/bin/env python3
"""Evaluate top 5 alphas over full 1-year and select 5 stocks each.

Filters:
  - Binary alphas (<=5 unique values) are excluded from top-5 ranking
    because stock selection from tied scores is arbitrary.
  - ETFs/stocks without price data on signal_date or return_end_date
    are excluded from stock selection candidates.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sqlite3
import numpy as np
import pandas as pd
import sys, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from lasps.data.processors.alpha101 import Alpha101Calculator

REBAL_STEP = 5
TOP_PCT = 0.10
MIN_STOCKS = 50
MIN_UNIQUE_VALUES = 10  # Minimum unique values to be considered continuous alpha


def _is_continuous_alpha(adf: pd.DataFrame, min_unique: int = MIN_UNIQUE_VALUES) -> bool:
    """Check if alpha produces continuous (non-binary) scores.

    Args:
        adf: Alpha DataFrame with dates as index, stocks as columns.
        min_unique: Minimum number of unique values required.

    Returns:
        True if alpha has enough unique values to be meaningful for ranking.
    """
    # Sample a few dates to check
    sample_dates = adf.index[::max(1, len(adf) // 5)][:5]
    for d in sample_dates:
        vals = adf.loc[d].dropna()
        if vals.nunique() >= min_unique:
            return True
    return False


def main():
    # Step 1: Load 1-year data
    print("=" * 70)
    print("Step 1: Load 1-year OHLCV (before 2026-02-16)")
    print("=" * 70)

    conn = sqlite3.connect('data/lasps.db')

    df = pd.read_sql("""
        SELECT stock_code, date, open, high, low, close, volume, trading_value, market_cap_daily
        FROM daily_prices
        WHERE date >= '2025-02-15' AND date <= '2026-02-15'
        ORDER BY date, stock_code
    """, conn, parse_dates=['date'])

    industry_df = pd.read_sql("SELECT code, sector_id FROM stocks", conn)
    stock_names = pd.read_sql("SELECT code, name FROM stocks", conn).set_index('code')['name']

    close_feb20 = pd.read_sql(
        "SELECT stock_code, close FROM daily_prices WHERE date = '2026-02-20'",
        conn
    ).set_index('stock_code')['close']
    conn.close()

    print(f"  Rows: {len(df):,}")
    print(f"  Date range: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"  Trading days: {df['date'].nunique()}")

    ohlcv = {}
    for col in ['open', 'high', 'low', 'close', 'volume']:
        ohlcv[col] = df.pivot(index='date', columns='stock_code', values=col)
    ohlcv['volume'] = ohlcv['volume'].astype(float)
    tv_pivot = df.pivot(index='date', columns='stock_code', values='trading_value').astype(float)
    ohlcv['vwap'] = tv_pivot / ohlcv['volume'].replace(0, np.nan)
    ohlcv['cap'] = df.pivot(index='date', columns='stock_code', values='market_cap_daily').astype(float)
    industry = industry_df.set_index('code')['sector_id']

    close_pivot = ohlcv['close']
    dates = close_pivot.index
    print(f"  Panel: {close_pivot.shape}")

    # Step 2: Compute all 101 alphas
    print("\n" + "=" * 70)
    print("Step 2: Compute all 101 alphas")
    print("=" * 70)

    calc = Alpha101Calculator(
        open_=ohlcv['open'], high=ohlcv['high'], low=ohlcv['low'],
        close=ohlcv['close'], volume=ohlcv['volume'],
        vwap=ohlcv['vwap'], cap=ohlcv['cap'], industry=industry,
    )

    simple_ids = calc.simple_alphas.get_implemented_alphas()
    industry_ids = calc.industry_alphas.get_implemented_alphas() if calc.industry_alphas else []
    all_ids = sorted(set(simple_ids + industry_ids))

    alphas = {}
    binary_ids = []
    continuous_ids = []
    for i, aid in enumerate(all_ids):
        try:
            result = calc.compute(aid)
            valid_pct = result.notna().sum().sum() / result.size * 100
            if valid_pct > 5:
                alphas[aid] = result
                if _is_continuous_alpha(result):
                    continuous_ids.append(aid)
                else:
                    binary_ids.append(aid)
            if (i+1) % 20 == 0:
                print(f"  [{i+1}/{len(all_ids)}] computed...")
                calc.clear_cache()
        except:
            pass

    print(f"  Valid alphas: {len(alphas)}")
    print(f"  Continuous alphas: {len(continuous_ids)} (used for ranking)")
    print(f"  Binary/discrete alphas: {len(binary_ids)} (excluded from ranking)")

    # Step 3: FULL 1-year evaluation (continuous alphas only)
    print("\n" + "=" * 70)
    print("Step 3: Evaluate over FULL 1-year (ALL rebalancing periods)")
    print("        (continuous alphas only - binary alphas excluded)")
    print("=" * 70)

    fwd_returns = close_pivot.shift(-REBAL_STEP) / close_pivot - 1

    all_rebal_dates = list(dates[::REBAL_STEP])
    valid_rebal = [d for d in all_rebal_dates
                   if d in fwd_returns.index and fwd_returns.loc[d].notna().sum() > MIN_STOCKS]

    print(f"  Rebalancing periods: {len(valid_rebal)}")
    print(f"  From {valid_rebal[0].date()} to {valid_rebal[-1].date()}")

    alpha_perf = {}

    # Only evaluate continuous alphas for ranking
    for aid in continuous_ids:
        adf = alphas[aid]
        orig_rets = []
        rev_rets = []

        for rd in valid_rebal:
            if rd not in adf.index or rd not in fwd_returns.index:
                continue

            scores = adf.loc[rd].dropna()
            rets = fwd_returns.loc[rd]
            common = scores.index.intersection(rets.dropna().index)

            if len(common) < MIN_STOCKS:
                continue

            n_sel = max(1, int(len(common) * TOP_PCT))

            top_orig = scores[common].sort_values(ascending=False).head(n_sel).index
            ret_orig = rets[top_orig].mean()
            if pd.notna(ret_orig):
                orig_rets.append(ret_orig)

            top_rev = scores[common].sort_values(ascending=True).head(n_sel).index
            ret_rev = rets[top_rev].mean()
            if pd.notna(ret_rev):
                rev_rets.append(ret_rev)

        if orig_rets:
            alpha_perf[f'alpha_{aid:03d}'] = {
                'mean': np.mean(orig_rets), 'n_periods': len(orig_rets), 'reversed': False,
                'alpha_id': aid,
            }
        if rev_rets:
            alpha_perf[f'alpha_{aid:03d}_rev'] = {
                'mean': np.mean(rev_rets), 'n_periods': len(rev_rets), 'reversed': True,
                'alpha_id': aid,
            }

    ranking = sorted(alpha_perf.items(), key=lambda x: x[1]['mean'], reverse=True)

    print(f"\n  Total continuous variants evaluated: {len(ranking)}")
    print(f"\n--- Top 10 (Full 1-Year, Continuous Only) ---")
    print(f"{'Rank':<6} {'Alpha':<22} {'Avg 5d Return':>14} {'Periods':>9}")
    print("-" * 54)
    for r, (name, info) in enumerate(ranking[:10], 1):
        print(f"{r:<6} {name:<22} {info['mean']*100:>+13.2f}% {info['n_periods']:>8}")

    # Step 4: Top 5 -> 5 stocks each (exclude ETFs/stocks without price data)
    print("\n" + "=" * 70)
    print("Step 4: Top 5 alphas -> select 5 stocks each")
    print("        (ETFs/stocks without 2/13 or 2/20 price data excluded)")
    print("=" * 70)

    signal_date = dates[dates <= pd.Timestamp('2026-02-13')][-1]
    print(f"  Signal date: {signal_date.date()}")
    print(f"  Return period: {signal_date.date()} -> 2026-02-20 (5 trading days)")

    # Build valid stock set: must have close on both signal_date and 2/20
    close_on_signal = close_pivot.loc[signal_date].dropna()
    valid_stocks = close_on_signal.index.intersection(close_feb20.dropna().index)
    print(f"  Valid stocks (have price on both dates): {len(valid_stocks)}")

    # Filter ranking: alpha must have differentiated scores on signal_date
    # Top 5 selected stocks must have meaningfully different scores
    # (rounded to 4 decimal places, at least 3 unique values)
    MIN_TOP5_UNIQUE = 3
    filtered_ranking = []
    skipped_signal = []
    for alpha_name, info in ranking:
        base_id = info['alpha_id']
        is_rev = info['reversed']
        adf = alphas[base_id]
        if signal_date not in adf.index:
            skipped_signal.append(alpha_name)
            continue
        scores = adf.loc[signal_date].dropna()
        scores = scores[scores.index.isin(valid_stocks)]
        if is_rev:
            scores = -scores
        top5_vals = scores.sort_values(ascending=False).head(5)
        # Round to 4 decimal places to check practical uniqueness
        rounded_unique = len(set(round(v, 4) for v in top5_vals.values))
        if rounded_unique >= MIN_TOP5_UNIQUE:
            filtered_ranking.append((alpha_name, info))
        else:
            skipped_signal.append(alpha_name)
        if len(filtered_ranking) >= 5:
            break

    if skipped_signal:
        print(f"  Skipped (tied scores on signal_date): {skipped_signal[:10]}...")

    top5_alphas = filtered_ranking[:5]
    all_results = {}

    for r, (alpha_name, info) in enumerate(top5_alphas, 1):
        base_id = info['alpha_id']
        is_rev = info['reversed']

        adf = alphas[base_id]
        scores = adf.loc[signal_date].dropna()

        # Filter to valid stocks only (exclude ETFs with no price data)
        scores = scores[scores.index.isin(valid_stocks)]

        if is_rev:
            scores = -scores

        top5_stocks = scores.sort_values(ascending=False).head(5)

        print(f"\n  #{r} {alpha_name} (1yr avg: {info['mean']*100:+.2f}%, {info['n_periods']} periods)")
        print(f"  {'Code':<8} {'Name':<16} {'Score':>10} {'2/13 Close':>12} {'2/20 Close':>12} {'5d Return':>10}")
        print("  " + "-" * 72)

        stock_results = []
        for code, score in top5_stocks.items():
            name = str(stock_names.get(code, '?'))[:14]
            p13 = close_pivot.loc[signal_date].get(code, np.nan)
            p20 = close_feb20.get(code, np.nan)
            ret = (p20/p13 - 1)*100 if pd.notna(p13) and pd.notna(p20) and p13 > 0 else np.nan

            p13s = f"{int(p13):,}" if pd.notna(p13) else "N/A"
            p20s = f"{int(p20):,}" if pd.notna(p20) else "N/A"
            rets = f"{ret:+.2f}%" if pd.notna(ret) else "N/A"

            print(f"  {code:<8} {name:<16} {score:>10.4f} {p13s:>12} {p20s:>12} {rets:>10}")

            stock_results.append({
                'code': code, 'name': name, 'score': round(float(score), 4),
                'close_0213': int(p13) if pd.notna(p13) else None,
                'close_0220': int(p20) if pd.notna(p20) else None,
                'return_5d_pct': round(ret, 2) if pd.notna(ret) else None,
            })

        rets_list = [s['return_5d_pct'] for s in stock_results if s['return_5d_pct'] is not None]
        avg_ret = round(np.mean(rets_list), 2) if rets_list else None
        if rets_list:
            print(f"  {'':8} {'Average':<16} {'':>10} {'':>12} {'':>12} {np.mean(rets_list):>+9.2f}%")

        all_results[alpha_name] = {
            'rank': r,
            'avg_1yr_return_pct': round(info['mean']*100, 2),
            'n_eval_periods': info['n_periods'],
            'reversed': is_rev,
            'stocks': stock_results,
            'portfolio_avg_5d_return_pct': avg_ret,
        }

    # Save verification data
    output = {
        'description': 'Top 5 continuous alphas selected by full 1-year evaluation, 5 stocks each',
        'signal_date': '2026-02-13 (last trading day before 2026-02-16)',
        'return_end_date': '2026-02-20',
        'eval_period': f"{valid_rebal[0].date()} ~ {valid_rebal[-1].date()}",
        'n_rebal_periods': len(valid_rebal),
        'total_alphas_computed': len(alphas),
        'continuous_alphas': len(continuous_ids),
        'binary_alphas_excluded': len(binary_ids),
        'total_variants_evaluated': len(ranking),
        'filters_applied': [
            f'Binary/discrete alphas excluded (unique_values < {MIN_UNIQUE_VALUES})',
            'ETFs/stocks without price on signal_date or return_end_date excluded from stock selection',
            f'Alphas with tied top-5 scores on signal_date excluded (min {MIN_TOP5_UNIQUE} unique values in top 5)',
        ],
        'top5': all_results,
    }

    with open('data/alpha_top5_verification.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nVerification data: data/alpha_top5_verification.json")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Alpha':<22} {'1yr Avg':>10} {'Periods':>9} {'5d Actual':>12}")
    print("-" * 56)
    for aname, info in all_results.items():
        r5d = f"{info['portfolio_avg_5d_return_pct']:+.2f}%" if info['portfolio_avg_5d_return_pct'] is not None else "N/A"
        print(f"{aname:<22} {info['avg_1yr_return_pct']:>+9.2f}% {info['n_eval_periods']:>8} {r5d:>12}")


if __name__ == '__main__':
    main()
