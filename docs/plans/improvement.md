# LASPS v7a 2단계 예측 시스템 구현 계획

## 목표
기본 모델(오프라인) + 조정 레이어(온라인)로 예측 정확도 향상 (55~60% → 62~67%)

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────┐
│  기본 모델 (오프라인 학습)                                │
│  입력: 29개 특성 (기존 25개 + 요일/월 sin/cos 4개)        │
│  출력: base_prob [SELL, HOLD, BUY]                      │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  조정 레이어 (온라인 학습)                                │
│  입력: 미국시장 5개 + 뉴스감성 2개 (선택)                  │
│  공식: adjusted = base × (1 + α·us + β·news)            │
│  제약: ±20% 범위, 확률 합 = 1                            │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
              매일 결과로 α, β 업데이트 (lr=0.05)
```

---

## Phase 1: 요일/월 특성 추가 (25→29)

### 수정 파일

| 파일 | 변경 내용 |
|------|----------|
| `lasps/utils/constants.py` | TEMPORAL_DIM=4, TOTAL_FEATURE_DIM=29 |
| `lasps/config/model_config.py` | input_dim: 25→29 |
| `lasps/utils/helpers.py` | normalize_time_series 범위 수정 |
| `scripts/historical_data.py` | export_npy에 요일/월 sin/cos 추가 |

### 핵심 구현

```python
# constants.py 추가
TEMPORAL_DIM = 4
TEMPORAL_FEATURES = ["day_sin", "day_cos", "month_sin", "month_cos"]
TEMPORAL_INDICES = (25, 29)
TOTAL_FEATURE_DIM = 29

# historical_data.py - compute_temporal_features 함수 추가
def compute_temporal_features(dates):
    day_sin = np.sin(2π × weekday / 5)
    day_cos = np.cos(2π × weekday / 5)
    month_sin = np.sin(2π × (month-1) / 12)
    month_cos = np.cos(2π × (month-1) / 12)
    return [day_sin, day_cos, month_sin, month_cos]
```

---

## Phase 2: 미국 시장 데이터 수집기

### 신규 파일

| 파일 | 설명 |
|------|------|
| `lasps/data/collectors/us_market_collector.py` | yfinance 기반 수집기 |
| `lasps/db/models/us_market.py` | DB 모델 |
| `lasps/db/repositories/us_market_repo.py` | Repository |

### 수집 지표 (무료, yfinance)

| 심볼 | 지표 | 정규화 |
|------|------|--------|
| ^GSPC | S&P500 변화율 | ÷5% → [-1,1] |
| ^IXIC | NASDAQ 변화율 | ÷5% → [-1,1] |
| ^VIX | 변동성지수 | (x-15)/35 → [0,1] |
| USDKRW=X | 환율 | (x-1300)/200 → [-1,1] |
| ^TNX | 10년 국채 | (x-3)/4 → [0,1] |

---

## Phase 3: 뉴스 감성 수집기

### 신규 파일

| 파일 | 설명 |
|------|------|
| `lasps/data/collectors/news_collector.py` | 키워드 기반 감성 분석 |
| `lasps/db/models/news_sentiment.py` | DB 모델 |

### 감성 분석 (초기 버전: 키워드 기반)
- 긍정: 상승, 호재, 급등, 신고가, 성장, 흑자...
- 부정: 하락, 악재, 급락, 적자, 위기, 부진...
- 점수: (긍정 - 부정) / 전체 → [-1, 1]

---

## Phase 4: 조정 레이어 (MarketAdjuster)

### 신규 파일

| 파일 | 설명 |
|------|------|
| `lasps/services/market_adjuster.py` | 확률 조정 로직 |
| `lasps/db/models/adjustment_params.py` | 파라미터 저장 |

### 수정 파일

| 파일 | 변경 내용 |
|------|----------|
| `lasps/services/predictor.py` | MarketAdjuster 통합 |

### 조정 공식

```python
# 클래스별 조정 계수
adjustment[c] = Σ(α[feat][c] × us_feat) + Σ(β[feat][c] × news_feat)
adjustment = clip(adjustment, -0.20, +0.20)

# 확률 조정
adjusted[c] = base[c] × (1 + adjustment[c])
adjusted = adjusted / sum(adjusted)  # 정규화
```

---

## Phase 5: 온라인 학습

### 신규 파일

| 파일 | 설명 |
|------|------|
| `scripts/update_adjuster.py` | 일일 파라미터 업데이트 |

### 수정 파일

| 파일 | 변경 내용 |
|------|----------|
| `scripts/daily_batch.py` | Step 12-13 추가 |

### 업데이트 로직

```python
# 5거래일 전 예측 vs 실제 결과 비교
error[actual_class] = +1
error[predicted_class] -= 1

# 가중치 업데이트
α[feat][c] += lr × error[c] × feat_value  # lr=0.05
```

---

## 구현 순서 및 예상 시간

| 순서 | Phase | 작업 | 시간 |
|------|-------|------|------|
| 1 | Phase 1 | 요일/월 특성 추가 | 4h |
| 2 | Phase 2 | 미국 시장 수집기 | 4h |
| 3 | Phase 3 | 뉴스 감성 수집기 | 5h |
| 4 | Phase 4 | 조정 레이어 | 6h |
| 5 | Phase 5 | 온라인 학습 | 5h |
| 6 | - | 통합 테스트 | 4h |

**총 예상: 28시간**

---

## 검증 방법

### 1. Phase 1 검증
```bash
# NPY shape 확인
python -c "
import numpy as np
ts = np.load('data/processed/train/time_series.npy')
print(f'Shape: {ts.shape}')  # Expected: (N, 60, 29)
"
```

### 2. Phase 2 검증
```bash
python -c "
from lasps.data.collectors.us_market_collector import USMarketCollector
from datetime import date
c = USMarketCollector()
print(c.get_normalized_features(date.today()))
"
```

### 3. Phase 4-5 검증
```bash
# 조정 전후 확률 비교
python -c "
from lasps.services.predictor import SectorAwarePredictor
# ... 모델 로드 ...
result = predictor.predict(ts, img, sid, us_features=us_data)
print(f'Base: {result[\"base_probabilities\"]}')
print(f'Adjusted: {result[\"probabilities\"]}')
"
```

### 4. 전체 파이프라인 테스트
```bash
# 일일 배치 실행 (Mock 모드)
python scripts/daily_batch.py --mode mock --date 2024-02-01
```

---

## 의존성 추가

```txt
yfinance>=0.2.0
```

---

## 주의사항

1. **데이터 재처리 필요**: 25→29 차원 변경으로 기존 NPY 재생성 필요
2. **모델 재학습 필요**: input_dim 변경으로 기존 체크포인트 호환 안됨
3. **온라인 학습 안정성**: 학습률 0.05, 조정 범위 ±20%로 제한하여 급격한 변화 방지
4. **미국 시장 시차**: 한국 월요일 → 미국 금요일 데이터 사용 (주말 고려)
