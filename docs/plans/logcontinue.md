# Alpha101 작업 이어하기 가이드

마지막 작업일: 2026-02-23

---

## 1. 현재 상태

### DB (data/lasps.db)

| 테이블 | 내용 | 현황 |
|:---|:---|:---|
| daily_prices | OHLCV + 거래대금 + 시가총액 | 2015-01-02 ~ 2026-02-20, 7,263,993행, 2,881거래일, 4,226종목 |
| stocks | 종목코드, 이름, sector_id | 4,226종목, 20개 섹터 |

daily_prices 필드:
- `id`, `stock_code`, `date`, `open`, `high`, `low`, `close`, `volume` — 100% 채움
- `trading_value` (거래대금), `market_cap_daily` (시가총액) — 79.5% 채움 (ETF 등 1,487K행 NULL)
- VWAP = trading_value / volume (별도 컬럼 없음, 계산 시 나눠서 사용)

### Alpha101 Parquet (data/alpha101/)

- 101개 parquet 파일 존재 (alpha_001 ~ alpha_101)
- **주의**: 이 파일들은 VWAP 데이터 수집 전에 계산된 것. 28개 VWAP 의존 알파는 NaN
- 재계산 필요: `python scripts/compute_alpha101_from_db.py` (VWAP/Cap/Industry 포함 전체 재계산)
- 재계산 시 약 1~2시간 소요

### 알파 계산 코드 (lasps/data/processors/alpha101/)

- `alpha_base.py`: VWAP fallback 버그 수정 완료 (close 대체 제거, ValueError 발생)
- `simple_alphas.py`: 82개 simple alpha 구현
- `industry_alphas.py`: 19개 industry alpha 구현
- `calculator.py`: Alpha101Calculator (vwap, cap, industry 파라미터 지원)
- 전체 101개 중 실제 계산 가능: 93개 (8개는 데이터 부족)

---

## 2. 완료된 작업

### VWAP 데이터 수집 (2026-02-20)
- pykrx로 10년치(2,874일) 거래대금 + 시가총액 수집 → daily_prices에 저장
- 스크립트: `scripts/collect_pykrx_vwap.py`
- 검증 완료: pykrx 원본과 100% 일치 (8/8 샘플)

### OHLCV 품질 이슈 발견 (2026-02-20)
- 키움 수정주가가 수집 시점마다 달라서 일부 종목 가격 불연속
- 삼성전자: 2024-12-19→12-20 사이 3.6배 급락 (수정주가 기준 혼재)
- 현재 유지 중. 향후 pykrx 원주가로 일괄 교체 가능

### Alpha101 Top 5 선정 (2026-02-21)
- 1년 전체 평가(48회 리밸런싱) 기반 상위 5개 연속값 알파 선정
- 스크립트: `scripts/alpha_top5_evaluate.py`
- 필터: 이진 알파 20개 제외, ETF 제외, signal_date 동점 제외

**결과**:

| 순위 | Alpha | 1년 평균 | 2/13→2/20 실제 | 의미 |
|:---:|:---|:---:|:---:|:---|
| 1 | alpha_044_rev | +2.80% | +3.74% | 고가-거래량 동행 종목 |
| 2 | alpha_083_rev | +2.77% | +4.66% | 변동성 축소 후 저가 종목 |
| 3 | alpha_088_rev | +1.58% | +5.38% | 하락 캔들 + 거래량 변화 |
| 4 | alpha_031_rev | +1.37% | +3.68% | 단기 하락 후 바닥 다지기 |
| 5 | alpha_084_rev | +1.36% | -0.34% | VWAP 대비 과매도 |

5개 모두 역방향(rev) = 역추세(mean-reversion) 전략

---

## 3. 미완료 / 다음 작업

### (1) Alpha101 전체 재계산 (우선순위 높음)
현재 parquet 파일은 VWAP 데이터 없이 계산된 것. VWAP/Cap/Industry 포함 재계산 필요.
```bash
python scripts/compute_alpha101_from_db.py
# 약 1~2시간 소요. data/alpha101/*.parquet 갱신됨
```

### (2) daily_prices 최신 데이터 업데이트
DB가 2026-02-20까지만 있음. 최신 날짜까지 업데이트 필요.
```bash
# pykrx로 수집하는 간단한 스크립트 (이전에 inline으로 실행)
# scripts/collect_pykrx_vwap.py는 trading_value/market_cap만 UPDATE
# OHLCV INSERT는 별도로 해야 함 (이전 세션에서 inline 코드로 실행)
```

### (3) Adaptive Alpha Backtest 재실행
재계산된 알파로 적응형 백테스트 재실행.
```bash
python scripts/backtest_alpha_adaptive.py
# Adaptive vs Fixed alpha_044 비교
# 출력: data/alpha_adaptive_results.csv, data/alpha_adaptive_alpha_log.csv
```

### (4) Top 5 알파 종목 선정 재실행
최신 데이터 + 재계산된 알파로 종목 선정.
```bash
python -X utf8 scripts/alpha_top5_evaluate.py
# 출력: data/alpha_top5_verification.json
```

---

## 4. 핵심 스크립트 목록

| 스크립트 | 용도 | 비고 |
|:---|:---|:---|
| `scripts/compute_alpha101_from_db.py` | DB에서 OHLCV 로드 → 101개 알파 계산 → parquet 저장 | VWAP/Cap/Industry 전달 |
| `scripts/alpha_top5_evaluate.py` | 1년 평가 → Top 5 알파 → 종목 선정 | 이진 알파 필터, ETF 필터 포함 |
| `scripts/collect_pykrx_vwap.py` | pykrx 거래대금/시가총액 수집 → DB UPDATE | --start, --no-resume 옵션 |
| `scripts/backtest_alpha_adaptive.py` | 매 리밸런싱마다 최적 알파 3개 자동 선택 백테스트 | Adaptive vs Fixed 비교 |
| `scripts/backtest_alpha101.py` | 고정 알파 Long-Short 백테스트 | 개별 알파 성과 비교 |

---

## 5. 알아둘 사항

### Alpha101 알파 분류
- **연속값 알파** (73개): 종목 순위 매기기 가능. 종목 선정에 사용
- **이진 알파** (20개): 0/1만 출력. 수백 종목 동점이라 종목 선정 무의미. 랭킹에서 제외
- **역방향(rev)**: 원본 점수를 뒤집어 사용. Top 5가 모두 rev = 역추세 전략이 유효

### VWAP 계산
```python
# DB에 vwap 컬럼 없음. 사용 시:
vwap = trading_value / volume  # volume이 0이면 NaN
```

### 데이터 소스
- OHLCV: 키움 API (수정주가, 품질 이슈 있음) + pykrx (2026-02-09 이후)
- trading_value, market_cap_daily: pykrx (2015~2026, 100% 정확)
- sector_id: 키움 API → stocks 테이블

### 주의사항
- Python 실행 시 한글 출력: `python -X utf8` 플래그 필요
- 알파 계산은 메모리 2GB+ 사용 (1년 데이터 기준)
- 전체 10년 계산 시 메모리 4GB+ 필요
