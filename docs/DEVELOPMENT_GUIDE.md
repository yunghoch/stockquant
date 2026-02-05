# LASPS v7a 개발 가이드

> 다른 컴퓨터에서 checkout 후 개발을 이어가기 위한 종합 가이드
> 최종 업데이트: 2026-02-05

---

## 0. 빠른 시작 가이드

새 컴퓨터에서 아래 순서대로 진행하면 됩니다.

### Step 1: 코드 받기

```bash
git clone https://github.com/yunghoch/stockquant.git
cd stockquant/v7a_sentiment
```

### Step 2: 문서 읽기

| 순서 | 문서 | 목적 |
|------|------|------|
| 1 | `docs/DEVELOPMENT_GUIDE.md` (이 문서) | 환경 설정, 구조, 아키텍처, 학습 데이터 준비 |
| 2 | `docs/CURRENT_STATUS.md` | 현재 진행 상태, 남은 작업, 성능 기준 |

필요할 때 참고:

| 문서 | 언제 읽나 |
|------|----------|
| `CLAUDE.md` | AI 어시스턴트에게 작업 요청할 때 (자동 로드됨) |
| `docs/codereview.md` | 미수정 이슈(#2, #9~12) 작업할 때 |
| `docs/testsen.md` | 수정한 코드 수동 검증할 때 |
| `VIBE_MASTER_v7a_PRD_FINAL.md` | 설계 의도/요구사항 확인할 때 |
| `docs/plans/2026-02-05-lasps-v7a-kiwoom-first.md` | 원래 구현 계획 확인할 때 |

### Step 3: 환경 설정

```bash
conda create -n lasps python=3.8
conda activate lasps
pip install -r requirements.txt

cp .env.example .env
# .env 편집: ANTHROPIC_API_KEY 입력 (필수)
```

### Step 4: 환경 검증

```bash
pytest tests/ -v --tb=short
# 기대 결과: 65 passed
```

### Step 5: 남은 작업 착수

`docs/CURRENT_STATUS.md` 섹션 3에 나온 우선순위대로:

| 순서 | 작업 | 환경 | 참고 |
|------|------|------|------|
| 1 | `scripts/historical_data.py` 구현 → 과거 데이터 수집 | Windows (키움 API) | 섹션 1.1.2, 4.2 |
| 2 | `scripts/train.py` Phase 2 구현 (섹터별 DataLoader) | 아무 OS | CURRENT_STATUS.md 섹션 3 |
| 3 | Mock 데이터로 Phase 1→3 학습 파이프라인 E2E 검증 | macOS 가능 | 섹션 4.3 |
| 4 | 실 데이터로 학습 실행 및 성능 측정 | GPU 권장 | CURRENT_STATUS.md 성능 기준 |

**macOS에서 바로 시작할 수 있는 작업**: 2번(Phase 2 구현) + 3번(Mock E2E 검증)

**Windows가 필요한 작업**: 1번(키움 API 데이터 수집)

---

## 1. 환경 설정

### 1.1 필수 요구사항

| 항목 | 버전 | 비고 |
|------|------|------|
| Python | 3.8+ | Kiwoom API는 Windows 32-bit 전용 |
| PyTorch | 1.8.0+ | **2.0 아님!** `batch_first` 미지원 |
| OS | macOS/Linux (개발), Windows (키움 실API) | macOS는 KiwoomMockAPI 사용 |

### 1.1.1 검증된 의존성 버전 (macOS 개발환경)

| 패키지 | 버전 | 비고 |
|--------|------|------|
| Python | 3.8.3 (Anaconda) | - |
| torch | 1.8.1 | `batch_first` 미지원 |
| pandas | 2.0.3 | Python 3.8에서 동작하지만, 새 환경은 1.x 권장 |
| numpy | 1.24.4 | - |
| mplfinance | 0.12.x | - |
| fastapi | 0.100+ | `on_event` 사용 중 (lifespan 미전환) |

### 1.1.2 Windows 환경 설정 (키움 API 실행 시)

키움 OpenAPI는 **Windows 32-bit Python**에서만 동작합니다:

```bash
# 1. 32-bit Python 3.8 설치 (x86 installer)
# https://www.python.org/downloads/ → Windows x86 executable installer

# 2. 키움 OpenAPI+ 설치
# https://www.kiwoom.com → 다운로드 → OpenAPI+

# 3. 32-bit 가상환경 생성
python -m venv venv32
venv32\Scripts\activate

# 4. 의존성 설치
pip install -r requirements.txt

# 5. 환경변수
set KIWOOM_ACCOUNT=계좌번호
set DART_API_KEY=dart_api_key

# 6. 데이터 수집
python scripts/historical_data.py --output data/processed
```

**주의**: macOS/Linux에서는 `KiwoomMockAPI`를 사용하므로 키움 설치 불필요.

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
# .env에 API 키 입력 (필수/선택):
#   ANTHROPIC_API_KEY=sk-ant-...  (필수: LLM 분석 기능)
#   DART_API_KEY=...              (선택: 부채비율 조회, 없으면 기본값 사용)
#   KIWOOM_ACCOUNT=계좌번호       (Windows만: 실 키움 API 연동 시)

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
│
├── docs/
│   ├── DEVELOPMENT_GUIDE.md     # ← 이 문서
│   ├── CURRENT_STATUS.md        # 현재 진행 상태 및 남은 작업
│   ├── codereview.md            # 코드 리뷰 결과 및 수정 이력
│   ├── testsen.md               # 매뉴얼 테스트 시나리오 (32개)
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

## 4. 학습 데이터 준비

### 4.1 데이터 파일 형식

`StockDataset`은 아래 4개의 npy 파일을 필요로 합니다:

```
data/processed/
├── train/
│   ├── time_series.npy    # (N, 60, 25) float32 - 시계열 피처
│   ├── chart_images.npy   # (N, 3, 224, 224) float32 - 캔들차트 이미지
│   ├── sector_ids.npy     # (N,) int64 - 섹터 ID (0~19)
│   └── labels.npy         # (N,) int64 - 라벨 (0=SELL, 1=HOLD, 2=BUY)
├── val/
│   └── (동일 구조)
└── test/
    └── (동일 구조)
```

**라벨 기준**: 5일 후 종가 수익률 기준, +3% 이상 = BUY(2), -3% 이하 = SELL(0), 나머지 = HOLD(1)

### 4.2 실 데이터 수집 (Windows 필요)

```bash
# Windows에서 키움 OpenAPI 연결 후 실행
python scripts/historical_data.py --output data/processed
```

`scripts/historical_data.py`는 현재 **스텁(stub)**으로, 실제 수집 로직 구현이 필요합니다. 구현 순서:

1. 키움 API로 전 종목 10년치 일봉 수집 (OPT10081)
2. 투자자별 매매동향 수집 (OPT10059)
3. 기술지표 15개 계산 (`TechnicalIndicatorCalculator`)
4. 시장감성 5차원 계산 (`MarketSentimentCalculator`)
5. 캔들차트 이미지 생성 (`ChartGenerator`, 60일 윈도우 슬라이딩)
6. `normalize_time_series()` 적용 (features 0-19 min-max per stock)
7. `compute_label()` 적용 (5일 후 수익률 ±3%)
8. 시간순 분할 → npy 파일 저장

**중요**: MA120에 119일 워밍업이 필요하므로, 180일 이상의 원시 데이터가 있어야 60일 시계열 윈도우를 생성할 수 있습니다.

### 4.3 Mock 데이터로 E2E 테스트

macOS에서 학습 파이프라인을 검증하려면 Mock 데이터를 생성합니다:

```python
import numpy as np
from pathlib import Path

def generate_mock_data(output_dir: str, n_samples: int = 500):
    """Mock 데이터 생성 (학습 파이프라인 E2E 검증용)"""
    out = Path(output_dir)
    for split in ["train", "val", "test"]:
        split_dir = out / split
        split_dir.mkdir(parents=True, exist_ok=True)
        n = n_samples if split == "train" else n_samples // 5
        np.save(split_dir / "time_series.npy",
                np.random.rand(n, 60, 25).astype(np.float32))
        np.save(split_dir / "chart_images.npy",
                np.random.rand(n, 3, 224, 224).astype(np.float32))
        np.save(split_dir / "sector_ids.npy",
                np.random.randint(0, 20, size=n).astype(np.int64))
        np.save(split_dir / "labels.npy",
                np.random.randint(0, 3, size=n).astype(np.int64))
    print(f"Mock data saved to {out}")

# 사용: generate_mock_data("data/processed")
```

생성 후 학습 실행:
```bash
python scripts/train.py --data-dir data/processed --device cpu --phase 1
```

---

## 5. 주요 명령어

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

## 6. 개발 컨벤션

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

## 7. 관련 문서

| 문서 | 위치 | 설명 |
|------|------|------|
| PRD (원본 설계) | `VIBE_MASTER_v7a_PRD_FINAL.md` | 전체 시스템 설계 기준 |
| CLAUDE.md | `CLAUDE.md` | AI 어시스턴트용 프로젝트 가이드 |
| 구현 계획서 | `docs/plans/2026-02-05-lasps-v7a-kiwoom-first.md` | 8 Phase 구현 계획 |
| 코드 리뷰 | `docs/codereview.md` | 코드 리뷰 결과 및 수정 이력 |
| 현재 상태 | `docs/CURRENT_STATUS.md` | 완료/미완료 항목, 남은 작업 |
| 테스트 시나리오 | `docs/testsen.md` | 32개 매뉴얼 테스트 시나리오 |
