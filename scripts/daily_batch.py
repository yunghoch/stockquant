#!/usr/bin/env python
# scripts/daily_batch.py

"""
LASPS v7a 일일 배치 프로세스 (장 마감 후 50분)

전체 파이프라인: 데이터 수집 -> QVM 스크리닝 -> 예측 -> LLM 분석 -> 리포트

Usage:
    python scripts/daily_batch.py
"""

import torch
from datetime import datetime
from pathlib import Path
from loguru import logger

from lasps.config.settings import settings
from lasps.models.sector_aware_model import SectorAwareFusionModel
from lasps.models.qvm_screener import QVMScreener
from lasps.services.predictor import SectorAwarePredictor
from lasps.services.llm_analyst import LLMAnalyst
from lasps.utils.logger import setup_logger


def main() -> None:
    """Run daily batch prediction pipeline."""
    setup_logger("INFO")
    today = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"=== LASPS v7a Daily Batch: {today} ===")

    # 1. 키움 로그인
    logger.info("Step 1: Kiwoom login")
    # kiwoom_api = KiwoomAPI()  # 실제 키움 API 연결

    # 2. 전체 종목 기본정보 수집
    logger.info("Step 2: Collecting stock info")

    # 3. DART 부채비율
    logger.info("Step 3: Collecting DART debt ratios")

    # 4. QVM 스크리닝 -> 50종목
    logger.info("Step 4: QVM screening -> 50 stocks")

    # 5. 상세 데이터 수집
    logger.info("Step 5: Collecting detailed data for 50 stocks")

    # 6. 시장 감성 계산
    logger.info("Step 6: Computing market sentiment")

    # 7. 기술지표 + 차트 이미지
    logger.info("Step 7: Computing indicators + generating charts")

    # 8. 시계열 구성
    logger.info("Step 8: Assembling time series (60, 28)")

    # 9. Sector-Aware 예측
    logger.info("Step 9: Running Sector-Aware predictions")
    model = SectorAwareFusionModel()
    checkpoint = settings.MODEL_PATH / "phase3_final.pt"
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))

    predictor = SectorAwarePredictor(model, device="cpu")

    # 10. LLM 상세 분석 (Top 10)
    logger.info("Step 10: LLM analysis for Top 10")
    if settings.ANTHROPIC_API_KEY:
        analyst = LLMAnalyst(api_key=settings.ANTHROPIC_API_KEY)

    # 11. 리포트 생성
    logger.info("Step 11: Generating report")
    logger.info(f"=== Daily Batch Complete: {today} ===")


if __name__ == "__main__":
    main()
