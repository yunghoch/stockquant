# LASPS v7a 현재 상태

> 최종 업데이트: 2026-02-05
> 최신 커밋: `492ac46` (docs: 개발 연속성 문서 추가)

---

## 1. 구현 완료 현황

### Phase별 진행 상태

| Phase | 설명 | 상태 | 커밋 |
|-------|------|------|------|
| Phase 0 | 프로젝트 초기화 | 완료 | `a7b8734` |
| Phase 1 | Config & Utils | 완료 | `41b77e4` ~ `90046d5` |
| Phase 2 | 키움 데이터 수집 | 완료 | `2364996` ~ `fb5e049` |
| Phase 3 | 데이터 프로세서 | 완료 | `3e3d0c1` ~ `cf9f0a4` |
| Phase 4 | 데이터셋 & 통합 파이프라인 | 완료 | `13ec18f` ~ `6ba02f3` |
| Phase 5 | 딥러닝 모델 | 완료 | `d366df7` ~ `2f10e8d` |
| Phase 6 | 학습 시스템 | 완료 | `e1b793a` ~ `9ac2700` |
| Phase 7 | 서비스 & API | 완료 | `ac84dda` ~ `01e868c` |
| 코드리뷰 수정 | 버그 수정 + 품질 개선 | 완료 | `0ca8c48` |

### 테스트 현황

- **전체 테스트: 65개 PASS**
- 실행 명령: `pytest tests/ -v --tb=short`
- 소요 시간: ~42초 (macOS, CPU)

---

## 2. 코드 리뷰 수정 완료 내역

코드 리뷰(`docs/codereview.md`)에서 발견된 12개 이슈 중 아래 항목이 수정 완료되었습니다:

### 수정 완료 (커밋 0ca8c48)

| # | 이슈 | 심각도 | 수정 내용 |
|---|------|--------|----------|
| 1 | sector_id=-1 라우팅 오류 | Critical | `get_sector_id()`가 미매핑 시 `DEFAULT_SECTOR_ID=17`(제조업 기타) 반환 |
| 3 | Trainer에 best model/early stopping 없음 | Important | `best_val_loss` 추적 + `copy.deepcopy(state_dict)` + patience 기반 early stopping 추가 |
| 4 | Phase 2 gradient clip 범위 | Important | `_run_epoch`에 `clip_params` 파라미터 추가, Phase 2에서 해당 sector head params만 전달 |
| 5 | Mock에서 불필요한 sleep | Important | `KiwoomCollector(api, rate_limit=False)` 파라미터 추가 |
| 6 | feature 수 미검증 | Important | `TOTAL_FEATURE_DIM` 체크 + `ValueError` raise |
| 7 | 문서 PyTorch 버전 불일치 | Important | `CLAUDE.md`: PyTorch 1.8+, `requirements.txt`: torch>=1.8.0 |
| - | Division-by-zero 6건 | Critical (추가) | `market_sentiment.py`, `technical_indicators.py`에 `.replace(0, np.nan)` 방어 |
| - | Per-stock 정규화 누락 | Critical (추가) | `normalize_time_series()` 함수 추가 (PRD 준수) |
| - | MODEL_CONFIG 미연결 | High (추가) | `SectorAwareFusionModel`이 `MODEL_CONFIG` dict 사용하도록 변경 |
| - | 에러 핸들링 부재 | High (추가) | `KiwoomCollector` ConnectionError 래핑, `IntegratedCollector` 데이터 검증 |
| - | FocalLoss weight device | Medium (추가) | `register_buffer` 사용으로 device 자동 이동 |
| - | logger 디렉토리 누락 | Medium (추가) | `logs/` 자동 생성 (`mkdir(parents=True, exist_ok=True)`) |

### 미수정 (향후 작업)

| # | 이슈 | 심각도 | 비고 |
|---|------|--------|------|
| 2 | forward_efficient gradient 단절 | Critical | Phase 2에서 빈 sector mask는 zeros로 남음. AdamW crash는 해결했으나 학습 효과 자체가 0. 실제 학습 데이터로 검증 필요 |
| 8 | kiwoom_collector 미사용 import 제거 | Minor | 수정됨 (타입 힌트로 대체) |
| 9 | QVM Momentum 정의 | Minor | PRD 설계대로이므로 유지. 향후 가격 모멘텀으로 개선 가능 |
| 10 | FastAPI on_event deprecation | Minor | 현재 동작함. FastAPI 업그레이드 시 lifespan으로 마이그레이션 |
| 11 | StockDataset mmap OOM 위험 | Minor | 실 데이터 학습 시 모니터링 필요 |
| 12 | forward vs forward_efficient 인터페이스 불일치 | Minor | forward_efficient에 shared_features 추가 고려 |

---

## 3. 남은 개발 작업

### 즉시 필요 (실 데이터 학습 전)

1. **과거 데이터 수집 및 전처리**
   - `scripts/historical_data.py`로 2015~2024 일봉 데이터 수집 (Windows 키움 API 필요)
   - 수집된 데이터를 `data/processed/` 형식으로 변환
   - Train/Val/Test 시간순 분할

2. **scripts/train.py Phase 2 구현 완성**
   - 현재 Phase 2는 스텁(stub)으로만 존재 (`scripts/train.py:118-121`, 로그만 출력)
   - 구현 필요사항:
     ```python
     # 1. 섹터별 DataLoader 분리
     for sector_id in range(NUM_SECTORS):
         sector_mask = train_ds.sector_ids == sector_id
         if sector_mask.sum() < THREE_PHASE_CONFIG["phase2_sector_heads"]["min_samples"]:
             logger.warning(f"Sector {sector_id}: insufficient data, skip")
             continue
         sector_indices = np.where(sector_mask)[0]
         sector_subset = Subset(train_ds, sector_indices)
         sector_loader = DataLoader(sector_subset, batch_size=batch_size,
                                    shuffle=True, collate_fn=stock_collate_fn)
         # 2. backbone 동결, 해당 sector head만 학습
         trainer.train_phase2(sector_loader, val_loader,
                              sector_id=sector_id,
                              epochs=cfg["epochs"], lr=cfg["lr"])
     ```
   - `THREE_PHASE_CONFIG["phase2_sector_heads"]["min_samples"]` (10000) 충족 검증
   - `torch.utils.data.Subset` 사용으로 메모리 효율적 분리

3. **학습 실행 및 검증**
   - Phase 1 → Phase 2 → Phase 3 순차 실행
   - 체크포인트 저장/로드 검증
   - Val/Test 성능 측정 (accuracy, F1, confusion matrix)

### 성능 기준 (목표)

| 지표 | 랜덤 기준 | 최소 목표 | 비고 |
|------|-----------|----------|------|
| Accuracy | 33.3% | 40%+ | 3-class 균등 분포 기준 |
| Macro F1 | 0.33 | 0.38+ | 클래스 불균형 고려 |
| BUY Precision | 33.3% | 45%+ | BUY 시그널 신뢰도 (가장 중요) |
| SELL Recall | 33.3% | 40%+ | 손실 회피 목적 |

**참고**: HOLD 클래스가 다수를 차지하므로 (±3% threshold), Macro F1이 Accuracy보다 중요한 지표임.

### 개선 작업 (선택)

4. **forward_efficient gradient 단절 해결** (코드리뷰 #2)
   - 빈 sector에 대한 logits zeros 처리 개선
   - 또는 Phase 2에서 sector별 DataLoader만 사용하므로 실제 문제 없을 수 있음

5. **StockDataset 메모리 최적화** (코드리뷰 #11)
   - 대량 데이터 학습 시 mmap + DataLoader num_workers 조합 테스트
   - 필요 시 chunk 단위 로딩 구현

6. **FastAPI lifespan 마이그레이션** (코드리뷰 #10)
   - `@app.on_event("startup")` → `lifespan` context manager

7. **QVM Momentum 개선** (코드리뷰 #9)
   - market_cap 대신 3/6/12개월 수익률 기반 모멘텀 스코어

---

## 4. 알려진 제약사항

### PyTorch 1.8.1 호환성
- `TransformerEncoderLayer`에 `batch_first=True` 없음 → 수동 transpose
- `AdamW`가 gradient 없을 때 crash → `has_grad` 체크 필수
- PyTorch 2.0+로 업그레이드 시 위 workaround 제거 가능

### 키움 OpenAPI 제약
- Windows 32-bit Python 전용
- macOS/Linux 개발 시 `KiwoomMockAPI` 사용 (GBM 기반 가격)
- 실 데이터 수집은 반드시 Windows에서 실행

### 정규화 범위
- PRD는 OHLCV(0-4)만 `min-max per stock` 명시
- 현재 구현은 OHLCV + indicators(0-19) 모두 정규화 (ML적으로 더 적합)
- Sentiment(20-24)는 이미 범위 고정되어 정규화 안함

---

## 5. 커밋 히스토리

```
492ac46 docs: 개발 연속성 문서 추가 (다른 환경에서 이어서 개발 가능)
0ca8c48 fix: code review 기반 버그 수정 및 품질 개선 (5개 태스크)
9ac2700 fix: handle PyTorch 1.8 AdamW crash when sector head has no gradients
01e868c feat: add daily batch and historical data collection scripts
501f0c0 feat: add FastAPI server with health, predict, sectors endpoints
d54ea36 feat: add LLMAnalyst service for Claude-powered stock analysis
ac84dda feat: add SectorAwarePredictor service with tests
e1b793a feat: implement 3-phase training system with FocalLoss
2f10e8d feat: implement QVMScreener for stock universe selection
f7d29a4 feat: implement SectorAwareFusionModel with 20 sector heads
dd2f109 feat: implement ChartCNN encoder for candlestick chart images
d366df7 feat: implement LinearTransformerEncoder for time series encoding
6ba02f3 feat: implement IntegratedCollector end-to-end data pipeline
13ec18f feat: implement StockDataset PyTorch dataset for multimodal inputs
cf9f0a4 test: add Phase 3 milestone - collection to features integration test
5d9e295 feat: implement candlestick chart generator (224x224)
f89543c feat: implement 5D market sentiment calculator
3e3d0c1 feat: implement 15 technical indicators calculator
4295167 test: add Phase 2 milestone - end-to-end collection integration tests
fb5e049 feat: implement DART debt ratio collector
5d5938c feat: implement KiwoomCollector with OPT10001/10081/10059/10014
2364996 feat: add Kiwoom API interface and realistic mock implementation
90046d5 feat: add logger, helpers (labeling, normalization), and metrics
6ca9407 feat: add Kiwoom TR config and model hyperparameters
df0c6ce feat: add 20-sector configuration with lookup functions
41b77e4 feat: add core constants and environment settings
a7b8734 chore: scaffold LASPS v7a project structure
```
