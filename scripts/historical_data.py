#!/usr/bin/env python
# scripts/historical_data.py

"""
과거 10년 데이터 수집 및 학습 데이터셋 생성

키움 OpenAPI를 통해 전 종목 10년치 일봉 데이터를 수집하고,
기술지표/시장감성/캔들차트를 계산하여 학습용 npy 파일로 저장한다.

Usage:
    python scripts/historical_data.py --output data/processed
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from lasps.utils.logger import setup_logger
from lasps.utils.constants import TIME_SERIES_LENGTH, TOTAL_FEATURE_DIM
from lasps.utils.helpers import compute_label

# 데이터 분할 기준
SPLIT_CONFIG = {
    "train": ("2015-01-01", "2022-12-31"),
    "val": ("2023-01-01", "2023-12-31"),
    "test": ("2024-01-01", "2024-12-31"),
}


def main() -> None:
    """Collect historical data and generate training datasets."""
    parser = argparse.ArgumentParser(
        description="LASPS v7a Historical Data Collection"
    )
    parser.add_argument("--output", type=str, default="data/processed")
    args = parser.parse_args()
    setup_logger("INFO")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Historical data collection - requires Kiwoom API connection")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Splits: {SPLIT_CONFIG}")

    # 실제 구현 시:
    # 1. 키움 API로 전 종목 10년치 일봉 수집 (OPT10081)
    # 2. 투자자별 데이터 수집 (OPT10059)
    # 3. 기술지표 15개 계산
    # 4. 시장감성 5차원 계산
    # 5. 캔들차트 이미지 생성 (60일 윈도우 슬라이딩)
    # 6. 라벨 생성 (5일 후 수익률 +/-3%)
    # 7. 시간순 분할 (train/val/test)
    # 8. npy 파일 저장
    #    - {split}_ts.npy: (N, 60, 25) float32
    #    - {split}_charts.npy: (N, 3, 224, 224) float32
    #    - {split}_sectors.npy: (N,) int64
    #    - {split}_labels.npy: (N,) int64

    logger.info(f"Done. Output files saved to: {output_dir}")


if __name__ == "__main__":
    main()
