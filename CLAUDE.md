# LASPS v7a - Sector-Aware Stock Prediction System

## Project Overview

LASPS (LLM-Augmented Stock Prediction System) v7a는 한국 주식시장을 대상으로 한 섹터 인식 딥러닝 예측 시스템이다. 뉴스 텍스트 감성 대신 시장 기반 5차원 감성 지표를 사용하여 비용을 50% 절감한다.

## Architecture

- **2-Branch Fusion Model**: Linear Transformer(시계열) + CNN(차트 이미지)
- **20 Sector-Specific Heads**: 섹터별 전용 분류기 (128 → 64 → 3)
- **3-Phase Training**: Backbone → Sector Heads → End-to-End Fine-tune
- **입력**: (60, 28) 시계열 + (3, 224, 224) 캔들차트 + sector_id
- **출력**: 3-class 분류 (SELL=0, HOLD=1, BUY=2)

## Key Constants

```
TIME_SERIES_SHAPE = (60, 28)      # OHLCV(5) + 지표(15) + 감성(5) + temporal(3)
CHART_IMAGE_SHAPE = (3, 224, 224)
NUM_SECTORS = 20
NUM_CLASSES = 3
PREDICTION_HORIZON = 5 days
LABEL_THRESHOLD = ±3%
```

## Project Structure

```
lasps/
├── config/          # settings, model_config, sector_config, tr_config
├── data/
│   ├── collectors/  # kiwoom, dart, integrated
│   ├── processors/  # technical_indicators, chart_generator, market_sentiment
│   └── datasets/    # stock_dataset (PyTorch Dataset)
├── models/          # linear_transformer, chart_cnn, sector_aware_model, qvm_screener
├── training/        # trainer (ThreePhaseTrainer), loss_functions
├── services/        # predictor, llm_analyst
├── api/             # FastAPI main
└── utils/           # logger, helpers, constants, metrics
tests/               # test files per module
scripts/             # train.py, daily_batch.py, historical_data.py
```

## Tech Stack

- **Python 3.8+** (32-bit required for Kiwoom API)
- **PyTorch 1.8+** - 모델 및 학습
- **pandas / numpy** - 데이터 처리
- **mplfinance** - 캔들차트 이미지 생성 (224x224)
- **ta** - 기술지표 라이브러리 (참고용, 직접 구현)
- **FastAPI** - REST API
- **anthropic** - Top 10 종목 Claude 분석
- **loguru** - 로깅
- **pytest** - 테스트

## Coding Conventions

- **Type hints**: 모든 함수에 타입 힌트 필수
- **Docstrings**: Google style
- **Naming**: PascalCase(클래스), snake_case(함수/변수), UPPER_SNAKE_CASE(상수)
- **Testing**: TDD 원칙, 키움 API는 Mock 사용
- **Commits**: conventional commits (feat:, fix:, refactor:, test:, chore:)

## Data Flow

```
Kiwoom OpenAPI
├── OPT10001: 종목기본정보 (업종코드 → sector_id)
├── OPT10081: 일봉 OHLCV → 기술지표 15개 + 시장감성 4개
├── OPT10059: 투자자별 → foreign_inst_flow (감성 5번째)
└── OPT10014: 공매도 (보조)
         ↓
[시계열 (60, 28)] + [차트 이미지 (3, 224, 224)] + [sector_id]
         ↓
SectorAwareFusionModel
├── LinearTransformerEncoder → 128-dim
├── ChartCNN → 128-dim
├── SharedFusion: concat(256) → 128
└── SectorHead[sector_id]: 128 → 64 → 3 (SELL/HOLD/BUY)
```

## 28-Feature Breakdown

| Index | Features | Count |
|-------|----------|-------|
| 0-4   | OHLCV (open, high, low, close, volume) | 5 |
| 5-8   | MA (5, 20, 60, 120) | 4 |
| 9-12  | RSI, MACD, MACD signal, MACD hist | 4 |
| 13-17 | BB upper/middle/lower/width, ATR | 5 |
| 18-19 | OBV, Volume MA20 | 2 |
| 20-24 | Sentiment (volume_ratio, volatility_ratio, gap_direction, rsi_norm, foreign_inst_flow) | 5 |
| 25-27 | Temporal (weekday, month, day) | 3 |

### Temporal Features (v2)

- **weekday**: 요일 정규화 (월=0, 금=0.8) - `weekday() / 4.0`
- **month**: 월 정규화 (1월=0.08, 12월=1.0) - `month / 12.0`
- **day**: 일 정규화 (1일=0.03, 31일=1.0) - `day / 31.0`

## Market Sentiment 5D (Key Innovation)

1. **volume_ratio**: clip(volume / MA20, 0, 3) / 3 → 0~1
2. **volatility_ratio**: clip(TR / ATR20, 0, 3) / 3 → 0~1
3. **gap_direction**: clip(gap%, -0.1, 0.1) * 10 → -1~+1
4. **rsi_norm**: RSI(14) / 100 → 0~1
5. **foreign_inst_flow**: sign * min(1, log10(|flow|+1)/8) → -1~+1

## Training Strategy

- **Phase 1** (30 epochs, lr=1e-4): 전체 데이터로 backbone 학습
- **Phase 2** (10 epochs/sector, lr=5e-4): backbone 동결, 섹터별 head 학습
- **Phase 3** (5 epochs, lr=1e-5): 전체 파라미터 미세 조정

## Data Splits

- **Train**: 2015-01 ~ 2022-12 (8년)
- **Val**: 2023-01 ~ 2023-12 (1년)
- **Test**: 2024-01 ~ 2024-12 (1년)
- **중요**: 시간순 분할 필수 (미래 데이터 누출 방지)

## Implementation Plan

구현 계획서 (Kiwoom-First): `docs/plans/2026-02-05-lasps-v7a-kiwoom-first.md`
이전 계획서: `docs/plans/2026-02-05-lasps-v7a-implementation.md`

## PRD Reference

원본 PRD: `VIBE_MASTER_v7a_PRD_FINAL.md`

## Commands

```bash
# 테스트 실행
pytest tests/ -v --tb=short

# v2 학습 데이터 생성 (ETF 제외 + temporal features)
python scripts/generate_dataset_v2.py --output data/processed_v2

# 학습
python scripts/train.py --device cuda --data-dir data/processed_v2

# 일일 배치
python scripts/daily_batch.py

# API 서버
uvicorn lasps.api.main:app --reload --port 8000
```

## Claude 작업 규칙

### 작업 로그 기록 (필수)

모든 새로운 작업은 `docs/plans/donelog.md`에 기록해야 한다:

1. **새 파일/스크립트 생성** 시 기록
2. **학습/배치 작업 실행** 시 기록 (시작, 종료, 결과)
3. **중요한 설정 변경** 시 기록
4. **문제 발견 및 해결** 시 기록

로그 형식:
```markdown
## YYYY-MM-DD

### [시간] 작업 제목
- **작업**: 수행한 내용
- **명령어**: 실행한 명령 (해당 시)
- **결과**: 결과 또는 상태
- **비고**: 추가 참고사항
```
