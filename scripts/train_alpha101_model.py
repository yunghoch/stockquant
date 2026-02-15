#!/usr/bin/env python3
"""Train Alpha101 model and compute IC.

This script:
1. Loads all 101 alpha parquet files
2. Computes 5-day forward returns from daily_prices
3. Calculates Daily IC for each alpha
4. Trains Linear Regression model on alpha combination
5. Evaluates out-of-sample performance
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
from pathlib import Path
import sqlite3
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Constants
PREDICTION_HORIZON = 5  # 5-day forward return
TRAIN_END = '2022-12-31'
VAL_END = '2023-12-31'
# Test: 2024-01-01 ~

DATA_DIR = Path('data/alpha101')
DB_PATH = Path('data/lasps.db')


def load_all_alphas() -> dict[str, pd.DataFrame]:
    """Load all alpha parquet files."""
    logger.info("Loading alpha101 parquet files...")

    alpha_files = sorted(DATA_DIR.glob('alpha_*.parquet'))
    logger.info(f"Found {len(alpha_files)} alpha files")

    alphas = {}
    for f in alpha_files:
        alpha_name = f.stem  # e.g., 'alpha_001'
        df = pd.read_parquet(f)
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        alphas[alpha_name] = df

    logger.info(f"Loaded {len(alphas)} alphas")
    return alphas


def compute_forward_returns(horizon: int = 5) -> pd.DataFrame:
    """Compute forward returns from daily_prices table."""
    logger.info(f"Computing {horizon}-day forward returns from DB...")

    conn = sqlite3.connect(DB_PATH)

    # Load daily prices
    query = """
    SELECT stock_code, date, close
    FROM daily_prices
    ORDER BY stock_code, date
    """
    df = pd.read_sql(query, conn)
    conn.close()

    logger.info(f"Loaded {len(df):,} price records")

    # Convert to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Pivot to wide format (date x stock_code)
    close_pivot = df.pivot(index='date', columns='stock_code', values='close')

    # Compute forward returns: (close[t+horizon] - close[t]) / close[t]
    forward_returns = close_pivot.shift(-horizon) / close_pivot - 1

    logger.info(f"Forward returns shape: {forward_returns.shape}")
    return forward_returns


def compute_daily_ic(alpha_df: pd.DataFrame, returns_df: pd.DataFrame) -> dict:
    """Compute daily Information Coefficient (Spearman correlation).

    Returns:
        dict with 'daily_ics', 'mean_ic', 'ic_std', 'ic_ir', 'hit_rate'
    """
    # Align dates
    common_dates = alpha_df.index.intersection(returns_df.index)
    common_stocks = alpha_df.columns.intersection(returns_df.columns)

    if len(common_dates) < 10 or len(common_stocks) < 10:
        return {'mean_ic': np.nan, 'ic_std': np.nan, 'ic_ir': np.nan, 'hit_rate': np.nan}

    alpha_aligned = alpha_df.loc[common_dates, common_stocks]
    returns_aligned = returns_df.loc[common_dates, common_stocks]

    daily_ics = []

    for date in common_dates:
        alpha_row = alpha_aligned.loc[date].dropna()
        return_row = returns_aligned.loc[date].dropna()

        common = alpha_row.index.intersection(return_row.index)

        if len(common) < 30:  # Need enough samples
            continue

        a = alpha_row[common].values
        r = return_row[common].values

        # Filter inf/nan
        mask = np.isfinite(a) & np.isfinite(r)
        if mask.sum() < 30:
            continue

        ic, _ = spearmanr(a[mask], r[mask])
        if np.isfinite(ic):
            daily_ics.append(ic)

    if len(daily_ics) < 10:
        return {'mean_ic': np.nan, 'ic_std': np.nan, 'ic_ir': np.nan, 'hit_rate': np.nan}

    daily_ics = np.array(daily_ics)
    mean_ic = daily_ics.mean()
    ic_std = daily_ics.std()
    ic_ir = mean_ic / ic_std if ic_std > 0 else 0  # Information Ratio
    hit_rate = (daily_ics > 0).mean()  # Fraction of positive IC days

    return {
        'daily_ics': daily_ics,
        'mean_ic': mean_ic,
        'ic_std': ic_std,
        'ic_ir': ic_ir,
        'hit_rate': hit_rate,
        'n_days': len(daily_ics)
    }


def prepare_training_data(alphas: dict, returns: pd.DataFrame,
                          start_date: str, end_date: str) -> tuple:
    """Prepare X, y for training.

    Converts panel data to sample-based format:
    - Each row is (date, stock) observation
    - Features are alpha values
    - Target is forward return
    """
    logger.info(f"Preparing data from {start_date} to {end_date}...")

    # Filter returns by date range
    returns_period = returns[(returns.index >= start_date) & (returns.index <= end_date)]

    # Stack to long format: (date, stock) -> return
    returns_stacked = returns_period.stack()
    returns_stacked.name = 'return'

    # Build feature matrix - filter each alpha by its own date range
    feature_dfs = []
    for alpha_name, alpha_df in alphas.items():
        # Filter alpha by date range (using alpha's own index)
        alpha_mask = (alpha_df.index >= start_date) & (alpha_df.index <= end_date)
        alpha_period = alpha_df[alpha_mask]
        alpha_stacked = alpha_period.stack()
        alpha_stacked.name = alpha_name
        feature_dfs.append(alpha_stacked)

    # Combine features (join on common (date, stock) pairs)
    features = pd.concat(feature_dfs, axis=1, join='inner')

    # Set proper names for MultiIndex
    features.index.names = ['date', 'stock']
    returns_stacked.index.names = ['date', 'stock']

    # Merge with returns (inner join to keep only common observations)
    features_reset = features.reset_index()
    returns_reset = returns_stacked.reset_index()

    data = pd.merge(
        features_reset,
        returns_reset,
        on=['date', 'stock'],
        how='inner'
    )
    data = data.set_index(['date', 'stock'])

    alpha_names = sorted(alphas.keys())

    # Replace inf with nan, then fill nan with 0
    for col in alpha_names:
        data[col] = data[col].replace([np.inf, -np.inf], np.nan)
        data[col] = data[col].fillna(0)
        # Clip extreme values
        data[col] = data[col].clip(-10, 10)

    # Only drop rows where return is NaN
    data_clean = data.dropna(subset=['return'])

    # Remove extreme return values (potential data errors)
    data_clean = data_clean[data_clean['return'].abs() < 1.0]  # < 100% return

    logger.info(f"Clean samples: {len(data_clean):,}")

    if len(data_clean) == 0:
        raise ValueError("No clean samples after filtering!")

    X = data_clean[alpha_names].values
    y = data_clean['return'].values

    return X, y, data_clean.index, alpha_names


def train_linear_model(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       alpha_names: list) -> dict:
    """Train Linear Regression and evaluate."""
    logger.info("Training Linear Regression model...")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)

    # Compute IC (Spearman correlation)
    train_ic, _ = spearmanr(y_train_pred, y_train)
    val_ic, _ = spearmanr(y_val_pred, y_val)

    # Feature importance (coefficients)
    coef_df = pd.DataFrame({
        'alpha': alpha_names,
        'coefficient': model.coef_
    }).sort_values('coefficient', key=abs, ascending=False)

    return {
        'model': model,
        'scaler': scaler,
        'train_ic': train_ic,
        'val_ic': val_ic,
        'coefficients': coef_df,
        'train_r2': model.score(X_train_scaled, y_train),
        'val_r2': model.score(X_val_scaled, y_val)
    }


def evaluate_on_test(model, scaler, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate model on test set."""
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    test_ic, _ = spearmanr(y_pred, y_test)
    test_r2 = model.score(X_test_scaled, y_test)

    # Decile analysis
    deciles = pd.qcut(y_pred, 10, labels=False, duplicates='drop')
    decile_returns = pd.DataFrame({'decile': deciles, 'return': y_test}).groupby('decile')['return'].mean()

    return {
        'test_ic': test_ic,
        'test_r2': test_r2,
        'decile_returns': decile_returns,
        'spread': decile_returns.iloc[-1] - decile_returns.iloc[0] if len(decile_returns) >= 2 else 0
    }


def main():
    logger.info("=" * 70)
    logger.info("Alpha101 Model Training and IC Evaluation")
    logger.info("=" * 70)

    # 1. Load alphas
    alphas = load_all_alphas()
    alpha_names = sorted(alphas.keys())

    # 2. Compute forward returns
    returns = compute_forward_returns(PREDICTION_HORIZON)

    # 3. Split periods
    train_returns = returns[returns.index <= TRAIN_END]
    val_returns = returns[(returns.index > TRAIN_END) & (returns.index <= VAL_END)]
    test_returns = returns[returns.index > VAL_END]

    logger.info(f"\nData splits:")
    logger.info(f"  Train: {train_returns.index.min()} ~ {train_returns.index.max()} ({len(train_returns)} days)")
    logger.info(f"  Val:   {val_returns.index.min()} ~ {val_returns.index.max()} ({len(val_returns)} days)")
    logger.info(f"  Test:  {test_returns.index.min()} ~ {test_returns.index.max()} ({len(test_returns)} days)")

    # 4. Compute Daily IC for each alpha
    logger.info("\n" + "=" * 70)
    logger.info("Computing Daily IC for each alpha...")
    logger.info("=" * 70)

    ic_results = {}

    for alpha_name in alpha_names:
        alpha_df = alphas[alpha_name]

        # Train IC
        train_ic = compute_daily_ic(
            alpha_df[alpha_df.index <= TRAIN_END],
            train_returns
        )

        # Val IC
        val_ic = compute_daily_ic(
            alpha_df[(alpha_df.index > TRAIN_END) & (alpha_df.index <= VAL_END)],
            val_returns
        )

        # Test IC
        test_ic = compute_daily_ic(
            alpha_df[alpha_df.index > VAL_END],
            test_returns
        )

        ic_results[alpha_name] = {
            'train': train_ic,
            'val': val_ic,
            'test': test_ic
        }

    # 5. Print IC summary
    logger.info("\n" + "=" * 70)
    logger.info("IC Summary (sorted by |Test IC|)")
    logger.info("=" * 70)

    ic_summary = []
    for alpha_name, result in ic_results.items():
        ic_summary.append({
            'alpha': alpha_name,
            'train_ic': result['train']['mean_ic'],
            'train_ir': result['train']['ic_ir'],
            'val_ic': result['val']['mean_ic'],
            'val_ir': result['val']['ic_ir'],
            'test_ic': result['test']['mean_ic'],
            'test_ir': result['test']['ic_ir'],
            'test_hit': result['test'].get('hit_rate', np.nan)
        })

    ic_df = pd.DataFrame(ic_summary)
    ic_df['abs_test_ic'] = ic_df['test_ic'].abs()
    ic_df = ic_df.sort_values('abs_test_ic', ascending=False)

    logger.info(f"\n{'Alpha':<15} {'Train IC':>10} {'Val IC':>10} {'Test IC':>10} {'Test IR':>10} {'Hit%':>8}")
    logger.info("-" * 75)

    for _, row in ic_df.head(30).iterrows():
        logger.info(
            f"{row['alpha']:<15} "
            f"{row['train_ic']:>+10.4f} "
            f"{row['val_ic']:>+10.4f} "
            f"{row['test_ic']:>+10.4f} "
            f"{row['test_ir']:>+10.4f} "
            f"{row['test_hit']*100:>7.1f}%"
        )

    # Save IC results
    ic_df.to_csv('data/alpha101_ic_results.csv', index=False)
    logger.info(f"\nIC results saved to data/alpha101_ic_results.csv")

    # 6. Train Linear Regression model
    logger.info("\n" + "=" * 70)
    logger.info("Training Linear Regression Model")
    logger.info("=" * 70)

    # Prepare data
    X_train, y_train, _, alpha_names_train = prepare_training_data(
        alphas, returns, '2015-01-01', TRAIN_END
    )
    X_val, y_val, _, _ = prepare_training_data(
        alphas, returns, '2023-01-01', VAL_END
    )
    X_test, y_test, _, _ = prepare_training_data(
        alphas, returns, '2024-01-01', '2026-12-31'
    )
    alpha_names = alpha_names_train  # Use consistent names

    logger.info(f"\nDataset sizes:")
    logger.info(f"  Train: {len(y_train):,} samples")
    logger.info(f"  Val:   {len(y_val):,} samples")
    logger.info(f"  Test:  {len(y_test):,} samples")

    # Train model
    model_result = train_linear_model(X_train, y_train, X_val, y_val, alpha_names)

    logger.info(f"\nModel Performance:")
    logger.info(f"  Train IC: {model_result['train_ic']:+.4f}")
    logger.info(f"  Val IC:   {model_result['val_ic']:+.4f}")
    logger.info(f"  Train R²: {model_result['train_r2']:.6f}")
    logger.info(f"  Val R²:   {model_result['val_r2']:.6f}")

    # 7. Evaluate on test
    logger.info("\n" + "=" * 70)
    logger.info("Test Set Evaluation")
    logger.info("=" * 70)

    test_result = evaluate_on_test(
        model_result['model'],
        model_result['scaler'],
        X_test, y_test
    )

    logger.info(f"\nTest Performance:")
    logger.info(f"  Test IC: {test_result['test_ic']:+.4f}")
    logger.info(f"  Test R²: {test_result['test_r2']:.6f}")
    logger.info(f"  Long-Short Spread (D10-D1): {test_result['spread']*100:+.2f}%")

    logger.info(f"\nDecile Returns (5-day forward):")
    for decile, ret in test_result['decile_returns'].items():
        logger.info(f"  D{decile+1}: {ret*100:+.3f}%")

    # 8. Top features
    logger.info("\n" + "=" * 70)
    logger.info("Top 20 Alpha Coefficients (by magnitude)")
    logger.info("=" * 70)

    top_coef = model_result['coefficients'].head(20)
    for _, row in top_coef.iterrows():
        logger.info(f"  {row['alpha']:<15}: {row['coefficient']:+.6f}")

    # 9. Final Summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)

    # Best single alphas
    top_5_alphas = ic_df.head(5)
    logger.info("\nTop 5 Single Alphas (by |Test IC|):")
    for i, (_, row) in enumerate(top_5_alphas.iterrows(), 1):
        logger.info(f"  {i}. {row['alpha']}: Test IC = {row['test_ic']:+.4f}")

    logger.info(f"\nCombined Model (Linear Regression with {len(alpha_names)} alphas):")
    logger.info(f"  Train IC: {model_result['train_ic']:+.4f}")
    logger.info(f"  Val IC:   {model_result['val_ic']:+.4f}")
    logger.info(f"  Test IC:  {test_result['test_ic']:+.4f}")

    # Interpretation
    if abs(test_result['test_ic']) > 0.05:
        logger.info("\n[+] Strong predictive signal found (IC > 0.05)")
    elif abs(test_result['test_ic']) > 0.02:
        logger.info("\n[*] Moderate predictive signal (IC 0.02~0.05)")
    else:
        logger.info("\n[!] Weak predictive signal (IC < 0.02)")


if __name__ == "__main__":
    main()
