# LASPS v7a 개발 가이드

> 다른 컴퓨터에서 checkout 후 개발을 이어가기 위한 종합 가이드
> 최종 업데이트: 2026-02-05

---

## 1. 환경 설정

### 1.1 필수 요구사항

| 항목 | 버전 | 비고 |
|------|------|------|
| Python | 3.8+ | Kiwoom API는 Windows 32-bit 전용 |
| PyTorch | 1.8.0+ | **2.0 아님!** `batch_first` 미지원 |
| OS | macOS/Linux (개발), Windows (키움 실API) | macOS는 KiwoomMockAPI 사용 |

### 1.2 초기 설정

```bash
# 1. 저장소 클론
git clone https://github.com/yunghoch/stockquant.git
cd stockquant/v7a_sentiment

# 2. 가상환경 생성 (conda 또는 venv)
conda create -n lasps python=3.8
conda activate lasps

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 환경변수 설정 (.env 파일 생성)
cp .env.example .env
# .env에 API 키 입력:
#   KIWOOM_ACCOUNT=계좌번호 (Windows만)
#   ANTHROPIC_API_KEY=sk-ant-... (LLM 분석용)
#   DART_API_KEY=... (부채비율 조회용)

# 5. 테스트 실행으로 환경 검증
pytest tests/ -v --tb=short
# 기대 결과: 65 passed
```

### 1.3 PyTorch 1.8.1 주의사항

이 프로젝트는 PyTorch 1.8.1 환경에서 개발되었으며, 다음 호환성 이슈가 있습니다:

1. **TransformerEncoderLayer**: `batch_first=True` 파라미터 미지원
   - `linear_transformer.py`에서 수동 transpose 처리 (`x.transpose(0, 1)`)

2. **AdamW 버그**: 모든 파라미터의 gradient가 None일 때 `UnboundLocalError: beta1` 발생
   - `trainer.py`의 `_run_epoch()`에서 `has_grad` 체크로 해결

3. **requirements.txt**: `torch>=1.8.0`으로 설정됨 (2.0 아님)

---

## 2. 프로젝트 구조

```
v7a_sentiment/
├── CLAUDE.md                    # AI 어시스턴트용 프로젝트 가이드
├── VIBE_MASTER_v7a_PRD_FINAL.md # 원본 PRD (설계 기준 문서)
├── requirements.txt             # Python 의존성
├── testsen.md                   # 매뉴얼 테스트 시나리오 (32개)
│
├── docs/
│   ├── DEVELOPMENT_GUIDE.md     # ← 이 문서
│   ├── CURRENT_STATUS.md        # 현재 진행 상태 및 남은 작업
│   └── plans/
│       ├── 2026-02-05-lasps-v7a-kiwoom-first.md  # 구현 계획서 (8 Phase)
│       └── 2026-02-05-lasps-v7a-implementation.md # 이전 계획서 (참고용)
│
├── lasps/                       # 메인 패키지
│   ├── config/                  # 설정 및 상수
│   │   ├── settings.py          # 환경변수 기반 설정 (Settings 클래스)
│   │   ├── model_config.py      # MODEL_CONFIG, TRAINING_CONFIG, THREE_PHASE_CONFIG
│   │   ├── sector_config.py     # 20개 섹터 매핑 (업종코드 → sector_id)
│   │   └── tr_config.py         # 키움 TR 코드 정의
│   │
│   ├── data/
│   │   ├── collectors/          # 데이터 수집
│   │   │   ├── kiwoom_base.py   # KiwoomAPIBase (Abstract)
│   │   │   ├── kiwoom_mock.py   # KiwoomMockAPI (GBM 기반 가격 생성)
│   │   │   ├── kiwoom_collector.py  # KiwoomCollector (실제 수집기)
│   │   │   ├── dart_collector.py    # DART 부채비율 수집
│   │   │   └── integrated_collector.py  # 수집→가공→피처 전체 파이프라인
│   │   ├── processors/          # 데이터 가공
│   │   │   ├── technical_indicators.py  # 15개 기술지표 계산
│   │   │   ├── market_sentiment.py      # 5D 시장감성 계산
│   │   │   └── chart_generator.py       # 캔들차트 이미지 생성 (224x224)
│   │   └── datasets/
│   │       └── stock_dataset.py # PyTorch Dataset (npy 파일 기반)
│   │
│   ├── models/                  # 딥러닝 모델
│   │   ├── linear_transformer.py    # LinearTransformerEncoder (60,25)→128
│   │   ├── chart_cnn.py             # ChartCNN (3,224,224)→128
│   │   ├── sector_aware_model.py    # SectorAwareFusionModel (메인 모델)
│   │   └── qvm_screener.py          # QVM 종목 선별기
│   │
│   ├── training/                # 학습 시스템
│   │   ├── trainer.py           # ThreePhaseTrainer (3-Phase 학습)
│   │   └── loss_functions.py    # FocalLoss (클래스 불균형 대응)
│   │
│   ├── services/                # 서비스 레이어
│   │   ├── predictor.py         # SectorAwarePredictor (추론)
│   │   └── llm_analyst.py       # LLMAnalyst (Claude 분석)
│   │
│   ├── api/                     # REST API
│   │   └── main.py              # FastAPI 서버
│   │
│   └── utils/                   # 유틸리티
│       ├── constants.py         # 전역 상수 (FEATURE 인덱스 등)
│       ├── helpers.py           # compute_label, normalize_time_series
│       ├── logger.py            # loguru 설정 (logs/ 자동 생성)
│       └── metrics.py           # 분류 메트릭 (accuracy, f1 등)
│
├── scripts/                     # 실행 스크립트
│   ├── train.py                 # 3-Phase 학습 실행
│   ├── daily_batch.py           # 일일 배치 예측
│   └── historical_data.py       # 과거 데이터 수집
│
└── tests/                       # 테스트 (65개)
    ├── test_collectors.py       # 키움 Mock/Collector 테스트
    ├── test_collectors_integration.py  # 수집 통합 테스트
    ├── test_indicators.py       # 기술지표 테스트
    ├── test_sentiment.py        # 시장감성 테스트
    ├── test_chart_generator.py  # 차트 생성 테스트
    ├── test_integrated_collector.py  # 통합 수집 테스트
    ├── test_processors_integration.py  # 프로세서 통합 테스트
    ├── test_dataset.py          # Dataset 테스트
    ├── test_models.py           # Transformer/CNN 테스트
    ├── test_sector_model.py     # SectorAwareFusionModel 테스트
    ├── test_training.py         # 3-Phase 학습 테스트
    ├── test_predictor.py        # 추론 서비스 테스트
    ├── test_qvm.py              # QVM 선별기 테스트
    ├── test_config.py           # 설정/상수 테스트
    └── test_utils.py            # 유틸리티 테스트
```

---

## 3. 핵심 아키텍처

### 3.1 데이터 흐름

```
Kiwoom OpenAPI (또는 Mock)
├── OPT10001: 종목기본정보 → sector_id (0~19)
├── OPT10081: 일봉 OHLCV (180일)
├── OPT10059: 투자자별 매매동향 (180일)
└── OPT10014: 공매도 (보조)
         ↓
[TechnicalIndicatorCalculator] → 15개 기술지표
[MarketSentimentCalculator]    → 5개 감성지표
[ChartGenerator]               → (3, 224, 224) 캔들차트
         ↓
[IntegratedCollector] → normalize_time_series() 적용
         ↓
시계열 (60, 25) + 차트 (3, 224, 224) + sector_id
         ↓
[SectorAwareFusionModel]
├── LinearTransformerEncoder → 128-dim
├── ChartCNN               → 128-dim
├── SharedFusion: concat(256) → 128
└── SectorHead[sector_id]: 128 → 64 → 3
         ↓
SELL(0) / HOLD(1) / BUY(2) 확률
```

### 3.2 25-Feature 구성

| 인덱스 | 피처 | 정규화 |
|--------|------|--------|
| 0-4 | OHLCV (open, high, low, close, volume) | min-max per stock [0,1] |
| 5-8 | MA (5, 20, 60, 120) | min-max per stock [0,1] |
| 9-12 | RSI, MACD, MACD signal, MACD hist | min-max per stock [0,1] |
| 13-17 | BB upper/middle/lower/width, ATR | min-max per stock [0,1] |
| 18-19 | OBV, Volume MA20 | min-max per stock [0,1] |
| 20-24 | Sentiment 5D | 이미 범위 고정 (변경 안함) |

### 3.3 3-Phase 학습 전략

| Phase | 목적 | 파라미터 | lr | epochs |
|-------|------|---------|-----|--------|
| 1 | Backbone 학습 | 전체 | 1e-4 | 30 (patience=5) |
| 2 | Sector Head 학습 | Head만 (backbone 동결) | 5e-4 | 10/sector |
| 3 | End-to-End 미세조정 | 전체 | 1e-5 | 5 (patience=5) |

### 3.4 모델 설정 (MODEL_CONFIG)

```python
# lasps/config/model_config.py
MODEL_CONFIG = {
    "num_sectors": 20,
    "linear_transformer": {"input_dim": 25, "hidden_dim": 128, "num_layers": 4, "num_heads": 4, "dropout": 0.2},
    "cnn": {"conv_channels": [32, 64, 128, 256], "output_dim": 128, "dropout": 0.3},
    "fusion": {"shared_dim": 128, "sector_head_hidden": 64, "num_classes": 3, "dropout": 0.3},
}
```

---

## 4. 주요 명령어

```bash
# 테스트
pytest tests/ -v --tb=short              # 전체 (65개)
pytest tests/test_training.py -v         # 학습 관련만
pytest tests/test_sector_model.py -v     # 모델 관련만

# 학습
python scripts/train.py --device cuda --data-dir data/processed
python scripts/train.py --phase 1        # Phase 1만
python scripts/train.py --checkpoint checkpoints/phase1_best.pt --phase 3

# 일일 배치
python scripts/daily_batch.py

# API 서버
uvicorn lasps.api.main:app --reload --port 8000

# API 테스트
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"stock_code": "005930"}'
```

---

## 5. 개발 컨벤션

### 코딩 규칙
- **Type hints**: 모든 함수에 필수
- **Docstrings**: Google style
- **Naming**: PascalCase(클래스), snake_case(함수/변수), UPPER_SNAKE_CASE(상수)
- **Testing**: TDD 원칙, 키움 API는 Mock 사용
- **Commits**: conventional commits (`feat:`, `fix:`, `refactor:`, `test:`, `chore:`)

### 키움 API Mock 사용법

macOS/Linux에서는 실제 키움 API 대신 `KiwoomMockAPI`를 사용합니다:

```python
from lasps.data.collectors.kiwoom_mock import KiwoomMockAPI
from lasps.data.collectors.kiwoom_collector import KiwoomCollector

# Mock API (GBM 기반 현실적 가격 생성, seed로 재현 가능)
mock_api = KiwoomMockAPI(seed=42)
collector = KiwoomCollector(mock_api, rate_limit=False)

# 실제 API와 동일한 인터페이스
info = collector.get_stock_info("005930")
ohlcv = collector.get_daily_ohlcv("005930", days=60)
```

### 데이터 분할 규칙

| 구간 | 기간 | 용도 |
|------|------|------|
| Train | 2015-01 ~ 2022-12 (8년) | 학습 |
| Val | 2023-01 ~ 2023-12 (1년) | 검증 / Early stopping |
| Test | 2024-01 ~ 2024-12 (1년) | 최종 평가 |

**중요**: 시간순 분할 필수. 미래 데이터 누출(look-ahead bias) 방지.

---

## 6. 관련 문서

| 문서 | 위치 | 설명 |
|------|------|------|
| PRD (원본 설계) | `VIBE_MASTER_v7a_PRD_FINAL.md` | 전체 시스템 설계 기준 |
| CLAUDE.md | `CLAUDE.md` | AI 어시스턴트용 프로젝트 가이드 |
| 구현 계획서 | `docs/plans/2026-02-05-lasps-v7a-kiwoom-first.md` | 8 Phase 구현 계획 |
| 코드 리뷰 | `docs/codereview.md` | 코드 리뷰 결과 및 수정 이력 |
| 현재 상태 | `docs/CURRENT_STATUS.md` | 완료/미완료 항목, 남은 작업 |
| 테스트 시나리오 | `docs/testsen.md` | 32개 매뉴얼 테스트 시나리오 |
