#!/usr/bin/env python3
"""Compute Alpha101 factors from database with optimized operators.

This script:
1. Loads OHLCV data from SQLite database
2. Optimizes slow operators (ts_rank, ts_argmax, ts_argmin) using Numba
3. Computes all 82 simple Alpha101 factors + 19 industry Alpha factors
4. Saves to parquet files with validation

Key improvements over compute_alpha101_from_db.py:
- Numba-accelerated ts_rank, ts_argmax, ts_argmin (10-20x faster)
- Industry Alpha support using pykrx_sector_idx
- Batch processing with progress tracking
- Checkpoint/resume capability
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from loguru import logger
from datetime import datetime
import gc
import json
import argparse
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

# Numba optimization for slow operators
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not available - using slow fallback")


# =============================================================================
# Numba-Optimized Time Series Operators
# =============================================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def _numba_ts_rank_1d(arr: np.ndarray, d: int) -> np.ndarray:
        """Numba-accelerated ts_rank for 1D array."""
        n = len(arr)
        result = np.empty(n)
        result[:d-1] = np.nan

        for i in range(d-1, n):
            window = arr[i-d+1:i+1]

            # Count valid (non-NaN) values
            valid_count = 0
            for v in window:
                if not np.isnan(v):
                    valid_count += 1

            if valid_count < 2:
                result[i] = np.nan
                continue

            # Get last value
            last_val = window[-1]
            if np.isnan(last_val):
                result[i] = np.nan
                continue

            # Count how many values are less than last_val
            rank = 1
            for v in window:
                if not np.isnan(v) and v < last_val:
                    rank += 1

            result[i] = rank / valid_count

        return result

    @jit(nopython=True, cache=True)
    def _numba_ts_argmax_1d(arr: np.ndarray, d: int) -> np.ndarray:
        """Numba-accelerated ts_argmax for 1D array."""
        n = len(arr)
        result = np.empty(n)
        result[:d-1] = np.nan

        for i in range(d-1, n):
            window = arr[i-d+1:i+1]

            # Find argmax
            max_val = -np.inf
            max_idx = -1
            all_nan = True

            for j in range(d):
                v = window[j]
                if not np.isnan(v):
                    all_nan = False
                    if v > max_val:
                        max_val = v
                        max_idx = j

            if all_nan:
                result[i] = np.nan
            else:
                # Convert to "days from most recent" (d=most recent, 1=oldest)
                result[i] = d - max_idx

        return result

    @jit(nopython=True, cache=True)
    def _numba_ts_argmin_1d(arr: np.ndarray, d: int) -> np.ndarray:
        """Numba-accelerated ts_argmin for 1D array."""
        n = len(arr)
        result = np.empty(n)
        result[:d-1] = np.nan

        for i in range(d-1, n):
            window = arr[i-d+1:i+1]

            # Find argmin
            min_val = np.inf
            min_idx = -1
            all_nan = True

            for j in range(d):
                v = window[j]
                if not np.isnan(v):
                    all_nan = False
                    if v < min_val:
                        min_val = v
                        min_idx = j

            if all_nan:
                result[i] = np.nan
            else:
                # Convert to "days from most recent" (d=most recent, 1=oldest)
                result[i] = d - min_idx

        return result


def ts_rank_fast(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Fast ts_rank using Numba."""
    if not NUMBA_AVAILABLE:
        # Fallback to slow version
        from lasps.data.processors.alpha101.operators import ts_rank
        return ts_rank(x, d)

    result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    for col in x.columns:
        result[col] = _numba_ts_rank_1d(x[col].values.astype(np.float64), d)
    return result


def ts_argmax_fast(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Fast ts_argmax using Numba."""
    if not NUMBA_AVAILABLE:
        from lasps.data.processors.alpha101.operators import ts_argmax
        return ts_argmax(x, d)

    result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    for col in x.columns:
        result[col] = _numba_ts_argmax_1d(x[col].values.astype(np.float64), d)
    return result


def ts_argmin_fast(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Fast ts_argmin using Numba."""
    if not NUMBA_AVAILABLE:
        from lasps.data.processors.alpha101.operators import ts_argmin
        return ts_argmin(x, d)

    result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    for col in x.columns:
        result[col] = _numba_ts_argmin_1d(x[col].values.astype(np.float64), d)
    return result


def indneutralize_fast(x: pd.DataFrame, industry: pd.Series) -> pd.DataFrame:
    """Fast industry-neutralize using pandas groupby.

    Much faster than row-by-row iteration.
    """
    # Filter industry to only include stocks in x
    common_stocks = x.columns.intersection(industry.index)
    industry_filtered = industry[common_stocks]

    # Create a mapping: stock_code -> industry_code
    # For stocks not in industry, they will be NaN and excluded from means
    stock_industry = pd.Series(index=x.columns, dtype=object)
    stock_industry[common_stocks] = industry_filtered.values

    # Compute industry means for each date using stack/groupby
    # This is vectorized and much faster
    result = x.copy()

    # Stack to long format
    stacked = x.stack()
    stacked.index.names = ['date', 'stock']

    # Add industry
    stacked_df = stacked.reset_index()
    stacked_df.columns = ['date', 'stock', 'value']
    stacked_df['industry'] = stacked_df['stock'].map(stock_industry)

    # Compute industry mean for each (date, industry) pair
    ind_means = stacked_df.groupby(['date', 'industry'])['value'].transform('mean')
    stacked_df['neutralized'] = stacked_df['value'] - ind_means

    # Handle stocks without industry (keep original value)
    no_industry_mask = stacked_df['industry'].isna()
    stacked_df.loc[no_industry_mask, 'neutralized'] = stacked_df.loc[no_industry_mask, 'value']

    # Pivot back to wide format
    result_df = stacked_df.pivot(index='date', columns='stock', values='neutralized')

    # Reorder columns to match original
    result_df = result_df.reindex(columns=x.columns)

    return result_df


def patch_operators():
    """Patch slow operators with fast versions.

    This patches both the operators module AND the alpha classes that have
    already imported the operators.
    """
    import lasps.data.processors.alpha101.operators as ops
    import lasps.data.processors.alpha101.simple_alphas as simple_mod
    import lasps.data.processors.alpha101.industry_alphas as industry_mod
    import lasps.data.processors.alpha101.alpha_base as base_mod

    # Store original operators
    ops._original_ts_rank = ops.ts_rank
    ops._original_ts_argmax = ops.ts_argmax
    ops._original_ts_argmin = ops.ts_argmin
    ops._original_indneutralize = ops.indneutralize

    # Replace in operators module
    ops.ts_rank = ts_rank_fast
    ops.ts_argmax = ts_argmax_fast
    ops.ts_argmin = ts_argmin_fast
    ops.indneutralize = indneutralize_fast

    # Also patch in modules that have already imported
    simple_mod.ts_rank = ts_rank_fast
    simple_mod.ts_argmax = ts_argmax_fast
    simple_mod.ts_argmin = ts_argmin_fast

    industry_mod.ts_rank = ts_rank_fast
    industry_mod.ts_argmax = ts_argmax_fast
    industry_mod.ts_argmin = ts_argmin_fast
    industry_mod.indneutralize = indneutralize_fast

    base_mod.ts_rank = ts_rank_fast
    base_mod.ts_argmax = ts_argmax_fast
    base_mod.ts_argmin = ts_argmin_fast
    base_mod.indneutralize = indneutralize_fast

    logger.info("Patched slow operators with Numba-accelerated versions")


def restore_operators():
    """Restore original operators."""
    import lasps.data.processors.alpha101.operators as ops
    import lasps.data.processors.alpha101.simple_alphas as simple_mod
    import lasps.data.processors.alpha101.industry_alphas as industry_mod
    import lasps.data.processors.alpha101.alpha_base as base_mod

    if hasattr(ops, '_original_ts_rank'):
        ops.ts_rank = ops._original_ts_rank
        ops.ts_argmax = ops._original_ts_argmax
        ops.ts_argmin = ops._original_ts_argmin
        ops.indneutralize = ops._original_indneutralize

        simple_mod.ts_rank = ops._original_ts_rank
        simple_mod.ts_argmax = ops._original_ts_argmax
        simple_mod.ts_argmin = ops._original_ts_argmin

        industry_mod.ts_rank = ops._original_ts_rank
        industry_mod.ts_argmax = ops._original_ts_argmax
        industry_mod.ts_argmin = ops._original_ts_argmin
        industry_mod.indneutralize = ops._original_indneutralize

        base_mod.ts_rank = ops._original_ts_rank
        base_mod.ts_argmax = ops._original_ts_argmax
        base_mod.ts_argmin = ops._original_ts_argmin
        base_mod.indneutralize = ops._original_indneutralize

        logger.info("Restored original operators")


# =============================================================================
# Database Loading
# =============================================================================

def load_ohlcv_from_db(
    db_path: str,
    start_date: str = None,
    end_date: str = None,
) -> Tuple[dict, pd.Series]:
    """Load OHLCV data and sector info from SQLite.

    Returns:
        Tuple of:
        - Dict with 'open', 'high', 'low', 'close', 'volume' DataFrames
        - Series mapping stock_code to sector_id (pykrx_sector_idx)
    """
    logger.info(f"Loading data from {db_path}...")

    conn = sqlite3.connect(db_path)

    # Build OHLCV query
    query = "SELECT stock_code, date, open, high, low, close, volume FROM daily_prices"
    conditions = []
    if start_date:
        conditions.append(f"date >= '{start_date}'")
    if end_date:
        conditions.append(f"date <= '{end_date}'")
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY date, stock_code"

    logger.info("Loading OHLCV data...")
    df = pd.read_sql(query, conn, parse_dates=['date'])

    logger.info(f"Loaded {len(df):,} rows")
    logger.info(f"Date range: {df['date'].min()} ~ {df['date'].max()}")
    logger.info(f"Stocks: {df['stock_code'].nunique():,}")

    # Load sector info
    logger.info("Loading sector info...")
    sector_df = pd.read_sql(
        "SELECT code, pykrx_sector_idx FROM stocks WHERE pykrx_sector_idx IS NOT NULL",
        conn
    )
    industry = sector_df.set_index('code')['pykrx_sector_idx']
    logger.info(f"Loaded {len(industry):,} stocks with sector info")
    logger.info(f"Unique sectors: {industry.nunique()}")

    conn.close()

    # Pivot to panel format
    logger.info("Pivoting to panel format...")

    ohlcv = {}
    for col in ['open', 'high', 'low', 'close', 'volume']:
        logger.info(f"  Pivoting {col}...")
        ohlcv[col] = df.pivot(index='date', columns='stock_code', values=col)

    # Convert volume to float
    ohlcv['volume'] = ohlcv['volume'].astype(float)

    logger.info(f"Panel shape: {ohlcv['close'].shape} (dates x stocks)")

    return ohlcv, industry


# =============================================================================
# Alpha Computation
# =============================================================================

def compute_and_save_alphas(
    ohlcv: dict,
    industry: pd.Series,
    output_dir: Path,
    compute_industry: bool = True,
    alpha_ids: Optional[List[int]] = None,
    resume: bool = True,
) -> Tuple[List[dict], List[dict], dict]:
    """Compute Alpha101 factors and save to disk.

    Args:
        ohlcv: Dict with OHLCV DataFrames
        industry: Series mapping stock_code to sector_id
        output_dir: Directory to save alpha files
        compute_industry: Whether to compute industry alphas
        alpha_ids: Specific alpha IDs to compute (None = all)
        resume: Skip already computed alphas

    Returns:
        Tuple of (computed, failed, metadata)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Patch operators with fast versions
    if NUMBA_AVAILABLE:
        patch_operators()

    # Import calculator after patching
    from lasps.data.processors.alpha101 import Alpha101Calculator

    logger.info("Initializing Alpha101Calculator...")
    calc = Alpha101Calculator(
        open_=ohlcv['open'],
        high=ohlcv['high'],
        low=ohlcv['low'],
        close=ohlcv['close'],
        volume=ohlcv['volume'],
        industry=industry if compute_industry else None,
    )

    # Get list of alphas to compute
    if alpha_ids is None:
        simple_ids = calc.simple_alphas.get_implemented_alphas()
        if compute_industry and calc.industry_alphas is not None:
            industry_ids = calc.industry_alphas.get_implemented_alphas()
        else:
            industry_ids = []
        all_alpha_ids = simple_ids + industry_ids
    else:
        all_alpha_ids = alpha_ids

    logger.info(f"Total alphas to compute: {len(all_alpha_ids)}")

    # Check for already computed (resume)
    if resume:
        existing = set()
        for f in output_dir.glob('alpha_*.parquet'):
            alpha_id = int(f.stem.split('_')[1])
            existing.add(alpha_id)

        all_alpha_ids = [a for a in all_alpha_ids if a not in existing]
        logger.info(f"Skipping {len(existing)} already computed alphas")
        logger.info(f"Remaining: {len(all_alpha_ids)} alphas")

    # Track results
    computed = []
    failed = []

    # Compute alphas
    industry_alphas_set = {48, 56, 58, 59, 63, 67, 69, 70, 76, 79, 80, 82, 87, 89, 90, 91, 93, 97, 100}

    for i, alpha_id in enumerate(all_alpha_ids):
        is_industry = alpha_id in industry_alphas_set
        alpha_type = "Industry" if is_industry else "Simple"

        try:
            logger.info(f"[{i+1}/{len(all_alpha_ids)}] Computing Alpha #{alpha_id} ({alpha_type})...")

            import time
            start = time.time()
            alpha_df = calc.compute(alpha_id)
            elapsed = time.time() - start

            # Save to parquet
            filename = output_dir / f"alpha_{alpha_id:03d}.parquet"
            alpha_df.to_parquet(filename)

            # Track stats
            valid_pct = alpha_df.notna().sum().sum() / alpha_df.size * 100
            computed.append({
                'alpha_id': alpha_id,
                'type': alpha_type,
                'valid_pct': valid_pct,
                'min': float(alpha_df.min().min()),
                'max': float(alpha_df.max().max()),
                'elapsed_sec': elapsed,
                'file': filename.name,
            })

            logger.info(f"  -> Saved {filename.name} (valid: {valid_pct:.1f}%, time: {elapsed:.1f}s)")

            # Memory cleanup every 10 alphas
            if (i + 1) % 10 == 0:
                calc.clear_cache()
                gc.collect()

        except Exception as e:
            import traceback
            logger.error(f"  -> Failed: {e}")
            failed.append({
                'alpha_id': alpha_id,
                'type': alpha_type,
                'error': str(e),
                'traceback': traceback.format_exc(),
            })

    # Restore operators
    if NUMBA_AVAILABLE:
        restore_operators()

    # Save metadata
    metadata = {
        'computed_at': datetime.now().isoformat(),
        'date_range': [
            str(ohlcv['close'].index.min()),
            str(ohlcv['close'].index.max())
        ],
        'num_dates': len(ohlcv['close']),
        'num_stocks': len(ohlcv['close'].columns),
        'num_sectors': int(industry.nunique()) if industry is not None else 0,
        'total_alphas': len(all_alpha_ids),
        'computed': len(computed),
        'failed': len(failed),
        'numba_enabled': NUMBA_AVAILABLE,
    }

    # Save stats
    if computed:
        stats_df = pd.DataFrame(computed)
        stats_df.to_csv(output_dir / 'alpha_stats.csv', index=False)

    if failed:
        failed_df = pd.DataFrame(failed)
        failed_df.to_csv(output_dir / 'alpha_failed.csv', index=False)

    # Save metadata
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    return computed, failed, metadata


def validate_alphas(output_dir: Path) -> dict:
    """Validate saved alpha files.

    Returns:
        Validation report dict
    """
    logger.info("Validating saved alphas...")

    report = {
        'total_files': 0,
        'valid_files': 0,
        'corrupted_files': [],
        'empty_files': [],
        'alpha_coverage': {},
    }

    parquet_files = list(output_dir.glob('alpha_*.parquet'))
    report['total_files'] = len(parquet_files)

    for f in parquet_files:
        try:
            df = pd.read_parquet(f)
            alpha_id = int(f.stem.split('_')[1])

            # Check for corruption
            if df.empty:
                report['empty_files'].append(f.name)
                continue

            # Check coverage
            valid_pct = df.notna().sum().sum() / df.size * 100
            report['alpha_coverage'][alpha_id] = valid_pct
            report['valid_files'] += 1

        except Exception as e:
            report['corrupted_files'].append({'file': f.name, 'error': str(e)})

    return report


def test_performance():
    """Test Numba optimization performance."""
    logger.info("Testing Numba optimization performance...")

    # Generate test data
    n_days = 500
    n_stocks = 100
    data = pd.DataFrame(
        np.random.randn(n_days, n_stocks),
        index=pd.date_range('2020-01-01', periods=n_days),
        columns=[f'S{i:04d}' for i in range(n_stocks)]
    )

    # Test ts_rank
    import time

    # Slow version
    from lasps.data.processors.alpha101.operators import ts_rank as ts_rank_slow
    start = time.time()
    _ = ts_rank_slow(data.iloc[:100, :20], 10)
    slow_time = time.time() - start

    # Fast version (warm-up Numba JIT)
    if NUMBA_AVAILABLE:
        _ = ts_rank_fast(data.iloc[:10, :5], 5)

        start = time.time()
        _ = ts_rank_fast(data.iloc[:100, :20], 10)
        fast_time = time.time() - start

        logger.info(f"ts_rank (100x20, d=10):")
        logger.info(f"  Slow: {slow_time:.3f}s")
        logger.info(f"  Fast: {fast_time:.3f}s")
        logger.info(f"  Speedup: {slow_time/fast_time:.1f}x")
    else:
        logger.info("Numba not available for performance test")


def main():
    parser = argparse.ArgumentParser(description='Compute Alpha101 factors from database')
    parser.add_argument('--db', type=str, default='data/lasps.db', help='Database path')
    parser.add_argument('--output', type=str, default='data/alpha101', help='Output directory')
    parser.add_argument('--start-date', type=str, default='2015-01-01', help='Start date')
    parser.add_argument('--end-date', type=str, default='2026-02-14', help='End date')
    parser.add_argument('--no-industry', action='store_true', help='Skip industry alphas')
    parser.add_argument('--alphas', type=str, default=None, help='Comma-separated alpha IDs')
    parser.add_argument('--no-resume', action='store_true', help='Recompute all alphas')
    parser.add_argument('--test', action='store_true', help='Run performance test only')
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Alpha101 Computation v2 (Numba-Optimized)")
    logger.info("=" * 70)

    if args.test:
        test_performance()
        return

    # Parse alpha IDs
    alpha_ids = None
    if args.alphas:
        alpha_ids = [int(x.strip()) for x in args.alphas.split(',')]
        logger.info(f"Computing specific alphas: {alpha_ids}")

    # Paths
    db_path = args.db
    output_dir = Path(args.output)

    # Check database exists
    if not Path(db_path).exists():
        logger.error(f"Database not found: {db_path}")
        return

    # Load OHLCV data
    ohlcv, industry = load_ohlcv_from_db(
        db_path,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # Compute and save alphas
    computed, failed, metadata = compute_and_save_alphas(
        ohlcv=ohlcv,
        industry=industry,
        output_dir=output_dir,
        compute_industry=not args.no_industry,
        alpha_ids=alpha_ids,
        resume=not args.no_resume,
    )

    # Clear memory
    del ohlcv
    gc.collect()

    # Validate
    validation = validate_alphas(output_dir)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\nOutput Directory: {output_dir.absolute()}")
    logger.info(f"Date Range: {metadata['date_range'][0]} ~ {metadata['date_range'][1]}")
    logger.info(f"Panel Size: {metadata['num_dates']:,} dates x {metadata['num_stocks']:,} stocks")
    logger.info(f"Sectors: {metadata['num_sectors']}")
    logger.info(f"Numba Enabled: {metadata['numba_enabled']}")

    logger.info(f"\nComputation Results:")
    logger.info(f"  - Success: {len(computed)}/{metadata['total_alphas']} alphas")
    logger.info(f"  - Failed: {len(failed)} alphas")

    if computed:
        avg_time = sum(c['elapsed_sec'] for c in computed) / len(computed)
        logger.info(f"  - Avg time per alpha: {avg_time:.1f}s")

    logger.info(f"\nValidation Results:")
    logger.info(f"  - Valid files: {validation['valid_files']}/{validation['total_files']}")
    logger.info(f"  - Corrupted: {len(validation['corrupted_files'])}")
    logger.info(f"  - Empty: {len(validation['empty_files'])}")

    # Coverage stats
    if validation['alpha_coverage']:
        coverages = list(validation['alpha_coverage'].values())
        logger.info(f"\nData Coverage:")
        logger.info(f"  - Average: {np.mean(coverages):.1f}%")
        logger.info(f"  - Min: {np.min(coverages):.1f}%")
        logger.info(f"  - Max: {np.max(coverages):.1f}%")

    # List failed alphas
    if failed:
        logger.info(f"\nFailed Alphas:")
        for f in failed[:10]:  # Show first 10
            logger.info(f"  - Alpha #{f['alpha_id']} ({f['type']}): {f['error'][:80]}")
        if len(failed) > 10:
            logger.info(f"  ... and {len(failed) - 10} more")

    logger.info("\n" + "=" * 70)
    logger.info("Complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
