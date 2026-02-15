#!/usr/bin/env python3
"""Compute Alpha101 factors from database and save to disk.

This script:
1. Loads OHLCV data from SQLite database
2. Pivots to panel format (dates × stocks)
3. Computes all 82 simple Alpha101 factors
4. Saves to parquet files
5. Validates completeness and integrity
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from lasps.data.processors.alpha101 import Alpha101Calculator


def load_ohlcv_from_db(db_path: str, start_date: str = None, end_date: str = None) -> dict:
    """Load OHLCV data from SQLite and pivot to panel format.

    Returns:
        Dict with 'open', 'high', 'low', 'close', 'volume' DataFrames
        Each DataFrame has index=dates, columns=stock_codes
    """
    logger.info(f"Loading data from {db_path}...")

    conn = sqlite3.connect(db_path)

    # Build query
    query = "SELECT stock_code, date, open, high, low, close, volume FROM daily_prices"
    conditions = []
    if start_date:
        conditions.append(f"date >= '{start_date}'")
    if end_date:
        conditions.append(f"date <= '{end_date}'")
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY date, stock_code"

    logger.info("Executing query...")
    df = pd.read_sql(query, conn, parse_dates=['date'])
    conn.close()

    logger.info(f"Loaded {len(df):,} rows")
    logger.info(f"Date range: {df['date'].min()} ~ {df['date'].max()}")
    logger.info(f"Stocks: {df['stock_code'].nunique():,}")

    # Pivot to panel format
    logger.info("Pivoting to panel format...")

    ohlcv = {}
    for col in ['open', 'high', 'low', 'close', 'volume']:
        logger.info(f"  Pivoting {col}...")
        ohlcv[col] = df.pivot(index='date', columns='stock_code', values=col)

    # Convert volume to float
    ohlcv['volume'] = ohlcv['volume'].astype(float)

    logger.info(f"Panel shape: {ohlcv['close'].shape} (dates × stocks)")

    return ohlcv


def compute_and_save_alphas(ohlcv: dict, output_dir: Path, batch_size: int = 20):
    """Compute Alpha101 factors and save to disk.

    Args:
        ohlcv: Dict with OHLCV DataFrames
        output_dir: Directory to save alpha files
        batch_size: Number of alphas to compute before saving (memory management)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing Alpha101Calculator...")
    calc = Alpha101Calculator(
        open_=ohlcv['open'],
        high=ohlcv['high'],
        low=ohlcv['low'],
        close=ohlcv['close'],
        volume=ohlcv['volume'],
    )

    # Get list of simple alphas (no industry neutralization needed)
    simple_alpha_ids = calc.simple_alphas.get_implemented_alphas()
    logger.info(f"Computing {len(simple_alpha_ids)} simple alphas...")

    # Track results
    computed = []
    failed = []

    # Compute alphas in batches
    for i, alpha_id in enumerate(simple_alpha_ids):
        try:
            logger.info(f"[{i+1}/{len(simple_alpha_ids)}] Computing Alpha #{alpha_id}...")
            alpha_df = calc.compute(alpha_id)

            # Save to parquet
            filename = output_dir / f"alpha_{alpha_id:03d}.parquet"
            alpha_df.to_parquet(filename)

            # Track stats
            valid_pct = alpha_df.notna().sum().sum() / alpha_df.size * 100
            computed.append({
                'alpha_id': alpha_id,
                'valid_pct': valid_pct,
                'min': alpha_df.min().min(),
                'max': alpha_df.max().max(),
                'file': filename.name,
            })

            logger.info(f"  -> Saved {filename.name} (valid: {valid_pct:.1f}%)")

            # Memory cleanup
            if (i + 1) % batch_size == 0:
                calc.clear_cache()
                gc.collect()

        except Exception as e:
            logger.error(f"  -> Failed: {e}")
            failed.append({'alpha_id': alpha_id, 'error': str(e)})

    # Save metadata
    metadata = {
        'computed_at': datetime.now().isoformat(),
        'date_range': [str(ohlcv['close'].index.min()), str(ohlcv['close'].index.max())],
        'num_dates': len(ohlcv['close']),
        'num_stocks': len(ohlcv['close'].columns),
        'total_alphas': len(simple_alpha_ids),
        'computed': len(computed),
        'failed': len(failed),
    }

    # Save stats
    stats_df = pd.DataFrame(computed)
    stats_df.to_csv(output_dir / 'alpha_stats.csv', index=False)

    if failed:
        failed_df = pd.DataFrame(failed)
        failed_df.to_csv(output_dir / 'alpha_failed.csv', index=False)

    # Save metadata
    import json
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


def main():
    logger.info("=" * 70)
    logger.info("Alpha101 Computation from Database")
    logger.info("=" * 70)

    # Paths
    db_path = "data/lasps.db"
    output_dir = Path("data/alpha101")

    # Check database exists
    if not Path(db_path).exists():
        logger.error(f"Database not found: {db_path}")
        return

    # Load OHLCV data
    # Use date range for memory efficiency (full range ~7M rows)
    ohlcv = load_ohlcv_from_db(
        db_path,
        start_date='2015-01-01',
        end_date='2026-02-06',
    )

    # Compute and save alphas
    computed, failed, metadata = compute_and_save_alphas(ohlcv, output_dir)

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
    logger.info(f"Panel Size: {metadata['num_dates']:,} dates × {metadata['num_stocks']:,} stocks")

    logger.info(f"\n계산 결과:")
    logger.info(f"  - 성공: {len(computed)}/{metadata['total_alphas']} alphas")
    logger.info(f"  - 실패: {len(failed)} alphas")

    logger.info(f"\n검증 결과:")
    logger.info(f"  - 유효 파일: {validation['valid_files']}/{validation['total_files']}")
    logger.info(f"  - 손상 파일: {len(validation['corrupted_files'])}")
    logger.info(f"  - 빈 파일: {len(validation['empty_files'])}")

    # Coverage stats
    if validation['alpha_coverage']:
        coverages = list(validation['alpha_coverage'].values())
        logger.info(f"\n데이터 커버리지:")
        logger.info(f"  - 평균: {np.mean(coverages):.1f}%")
        logger.info(f"  - 최소: {np.min(coverages):.1f}%")
        logger.info(f"  - 최대: {np.max(coverages):.1f}%")

    # List failed alphas
    if failed:
        logger.info(f"\n실패한 Alpha:")
        for f in failed[:10]:  # Show first 10
            logger.info(f"  - Alpha #{f['alpha_id']}: {f['error'][:50]}")

    logger.info("\n" + "=" * 70)
    logger.info("완료!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
