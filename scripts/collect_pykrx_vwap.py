#!/usr/bin/env python3
"""Collect VWAP (trading_value) and market_cap from pykrx for all trading days.

Updates daily_prices table with:
- trading_value: 거래대금 (VWAP = trading_value / volume)
- market_cap_daily: 일별 시가총액

Usage:
    python scripts/collect_pykrx_vwap.py
    python scripts/collect_pykrx_vwap.py --start 2024-01-01  # resume from date
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sqlite3
import time
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
from pykrx import stock
from loguru import logger

DB_PATH = Path('data/lasps.db')
BATCH_SIZE = 50  # commit every N days


def get_trading_dates(conn: sqlite3.Connection, start_date: str = None) -> list:
    """Get all unique trading dates from daily_prices."""
    query = "SELECT DISTINCT date FROM daily_prices"
    if start_date:
        query += f" WHERE date >= '{start_date}'"
    query += " ORDER BY date"

    cursor = conn.cursor()
    cursor.execute(query)
    return [row[0] for row in cursor.fetchall()]


def get_dates_already_collected(conn: sqlite3.Connection) -> set:
    """Get dates that already have trading_value data."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT date FROM daily_prices
        WHERE trading_value IS NOT NULL
    """)
    return {row[0] for row in cursor.fetchall()}


def collect_and_update(conn: sqlite3.Connection, dates: list):
    """Collect pykrx data and update daily_prices for each date."""
    cursor = conn.cursor()

    total = len(dates)
    updated_total = 0
    matched_total = 0
    errors = 0
    start_time = time.time()

    for i, date_str in enumerate(dates):
        try:
            # Convert date format: 2024-01-02 -> 20240102
            pykrx_date = date_str.replace('-', '')

            # Fetch from pykrx (KOSPI + KOSDAQ)
            df = stock.get_market_ohlcv_by_ticker(pykrx_date, market='ALL')

            if df is None or len(df) == 0:
                logger.warning(f"[{i+1}/{total}] {date_str}: no data from pykrx")
                errors += 1
                continue

            # Also get market cap separately (more reliable)
            df_cap = stock.get_market_cap_by_ticker(pykrx_date, market='ALL')

            # Update each stock
            updated = 0
            for ticker in df.index:
                trading_val = int(df.loc[ticker, '거래대금']) if pd.notna(df.loc[ticker, '거래대금']) else None

                # Get market cap
                mcap = None
                if df_cap is not None and ticker in df_cap.index:
                    mcap = int(df_cap.loc[ticker, '시가총액']) if pd.notna(df_cap.loc[ticker, '시가총액']) else None
                elif '시가총액' in df.columns and pd.notna(df.loc[ticker, '시가총액']):
                    mcap = int(df.loc[ticker, '시가총액'])

                if trading_val is not None or mcap is not None:
                    cursor.execute("""
                        UPDATE daily_prices
                        SET trading_value = ?, market_cap_daily = ?
                        WHERE stock_code = ? AND date = ?
                    """, (trading_val, mcap, ticker, date_str))

                    if cursor.rowcount > 0:
                        updated += 1

            updated_total += updated
            matched_total += len(df)

            # Commit in batches
            if (i + 1) % BATCH_SIZE == 0:
                conn.commit()

            # Progress
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate if rate > 0 else 0

            if (i + 1) % 100 == 0 or i == 0:
                logger.info(
                    f"[{i+1}/{total}] {date_str}: "
                    f"pykrx={len(df)}, updated={updated} | "
                    f"total_updated={updated_total:,} | "
                    f"elapsed={elapsed/60:.1f}min, remaining={remaining/60:.1f}min"
                )

            # Rate limiting
            time.sleep(0.3)

        except Exception as e:
            logger.error(f"[{i+1}/{total}] {date_str}: {e}")
            errors += 1
            time.sleep(1)  # longer pause on error
            continue

    # Final commit
    conn.commit()

    elapsed = time.time() - start_time
    logger.info(f"\nCollection complete:")
    logger.info(f"  Days processed: {total}")
    logger.info(f"  Total rows updated: {updated_total:,}")
    logger.info(f"  Errors: {errors}")
    logger.info(f"  Time: {elapsed/60:.1f} minutes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--no-resume', action='store_true', help='Do not skip already collected dates')
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)

    # Get all trading dates
    all_dates = get_trading_dates(conn, args.start)
    logger.info(f"Total trading dates in DB: {len(all_dates)}")

    if not args.no_resume:
        # Skip dates already collected
        collected = get_dates_already_collected(conn)
        dates = [d for d in all_dates if d not in collected]
        logger.info(f"Already collected: {len(collected)} dates")
        logger.info(f"Remaining: {len(dates)} dates")
    else:
        dates = all_dates

    if not dates:
        logger.info("Nothing to collect. All dates already have data.")
        conn.close()
        return

    logger.info(f"Collecting {len(dates)} dates: {dates[0]} ~ {dates[-1]}")
    logger.info(f"Estimated time: {len(dates) * 1.5 / 60:.0f}~{len(dates) * 2 / 60:.0f} minutes")

    collect_and_update(conn, dates)

    # Verify
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM daily_prices WHERE trading_value IS NOT NULL")
    count = cursor.fetchone()[0]
    logger.info(f"\nVerification: {count:,} rows now have trading_value")

    cursor.execute("SELECT COUNT(*) FROM daily_prices WHERE market_cap_daily IS NOT NULL")
    count = cursor.fetchone()[0]
    logger.info(f"Verification: {count:,} rows now have market_cap_daily")

    conn.close()


if __name__ == '__main__':
    main()
