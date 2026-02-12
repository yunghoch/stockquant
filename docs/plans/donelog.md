# 작업 로그 (Done Log)

Claude가 수행한 모든 작업을 기록합니다.

---

## 2026-02-12

### [18:30] 섹터 분포 불균형 문제 발견

- **문제**: 학습 데이터의 섹터 분포가 심각하게 불균형
- **현황**:
  | 섹터 ID | 섹터명 | 샘플 수 | 비율 | 상태 |
  |---------|--------|---------|------|------|
  | 15 | 통신업 | 278,240 | 32.57% | 과다 |
  | 17 | 제조업(기타) | 278,170 | 32.56% | 과다 |
  | 19 | 광업 | 1,785 | 0.21% | 심각 부족 |

- **통계**:
  - Top 2 섹터 (15, 17): 65.1% 차지
  - Max/Min 비율: 155.9배

- **의심 원인**:
  - 섹터 17: `DEFAULT_SECTOR_ID = 17`로 미매핑 종목이 모두 여기로
  - 섹터 15 (통신업): 실제 통신업이 32%일 리 없음 → 데이터 수집/매핑 오류 의심

- **조치 필요**: DART, PyKRX, 키움 API의 실제 업종 분포 확인 필요

### [18:45] 실제 업종 분포 조사 결과 (pykrx)

- **실제 시장 현황** (pykrx 기준):
  ```
  KOSPI:  950 종목
  KOSDAQ: 1,821 종목
  Total:  2,771 종목
  ```

- **실제 업종별 종목 수** (KOSPI):
  | 업종 | 실제 종목수 |
  |------|------------|
  | 서비스업 | 494개 |
  | 금융업 | 110개 |
  | 의약품 | 103개 |
  | 전기전자 | 68개 |
  | 기타제조 | 63개 |
  | **통신업** | **5개** |
  | 전기가스 | 10개 |

- **심각한 불일치 발견**:
  | 업종 | 실제 종목수 | 학습데이터 비율 | 문제 |
  |------|------------|----------------|------|
  | **통신업 (15)** | **5개** | **32.57%** | **불가능** |
  | 제조업기타 (17) | 63개 | 32.56% | DEFAULT_SECTOR_ID 문제 |

- **원인 추정**:
  1. 키움 API 업종코드와 sector_config.py 매핑 불일치
  2. 데이터 수집 시 업종코드 파싱 오류
  3. DEFAULT_SECTOR_ID = 17 로 미매핑 종목 집중

- **다음 조치**:
  - 키움 API의 실제 업종코드 체계 재확인 필요
  - generate_dataset_v2.py에서 업종코드 매핑 로직 검토 필요

### [20:28] pykrx 섹터 정보 DB 수집 완료

- **작업**: pykrx에서 종목별 섹터 정보를 수집하여 DB에 저장
- **신규 스크립트**: `scripts/collect_pykrx_sector.py`
- **DB 변경**: `stocks` 테이블에 컬럼 추가
  - `pykrx_sector_idx`: pykrx 업종 인덱스 (예: "1020")
  - `pykrx_sector_name`: pykrx 업종명 (예: "통신")

- **수집 결과**:
  | pykrx Index | Name | Count |
  |-------------|------|-------|
  | 2024 | 제조 (KOSDAQ) | 1,115 |
  | 2012 | 일반서비스 (KOSDAQ) | 140 |
  | 1021 | 금융 | 110 |
  | 1008 | 화학 | 103 |
  | 2027 | 유통 (KOSDAQ) | 101 |
  | 1013 | 전기전자 | 68 |
  | **1020** | **통신** | **5** |
  | Total | - | 2,153 |

- **결과 파일**:
  - `pykrx_sectors.json`: pykrx API 조회 결과
  - `pykrx_db_sectors.json`: DB 저장 후 분포 확인

### [20:36] IT서비스 업종 추가 및 매핑 검증

- **문제 발견**: NAVER, 카카오가 매핑되지 않음
  - 원인: IT 서비스(1046) 인덱스가 수집 대상에 없었음
- **스크립트 수정**: 부동산(1045), IT서비스(1046), 오락문화(1047) 추가
- **재수집 결과**:
  - 추가 업데이트: 65개 종목
  - 최종 매핑: 2,218/4,226 (52.5%)

- **매핑 검증 (대표 종목)**:
  | 종목 | 코드 | pykrx 업종 |
  |------|------|-----------|
  | 삼성전자 | 005930 | 1013:전기전자 |
  | SK하이닉스 | 000660 | 1013:전기전자 |
  | **NAVER** | 035420 | **1046:IT 서비스** |
  | **카카오** | 035720 | **1046:IT 서비스** |
  | KT | 030200 | 1020:통신 |
  | SK텔레콤 | 017670 | 1020:통신 |

- **미매핑 종목 분류** (2,008개):
  - ETF/ETN: 1,286개
  - 우선주: 113개
  - 스팩: 75개
  - 채권: 29개
  - 기타: 505개

- **결론**: 키움 DB와 pykrx 매핑 정상 확인 완료

### [20:45] ETF 필터링 버그 발견 및 학습 중단

- **문제 발견**: 학습 데이터에 ETF가 포함되어 있음
  - 학습 데이터 내 3,024개 종목 중:
    - pykrx 매핑 가능 (일반주식): 1,956개 (64.7%)
    - 우선주: 113개 (3.7%)
    - **ETF 등: 955개 (31.6%)**

- **원인**: `generate_dataset_v2.py`의 ETF 필터링 로직 오류
  ```python
  # 잘못된 로직 (라인 150-153)
  if not code.isdigit():
      etf_excluded += 1
      continue
  ```
  - ETF 종목 코드도 6자리 숫자임 (예: KODEX 200 = 069500)
  - `code.isdigit()` 조건으로는 ETF를 필터링할 수 없음
  - 이 조건은 `0000H0` 같은 신규 형식 코드만 제외함

- **학습 중단**: Phase 1 학습 중지 (shell_id: b068f20)
  - Epoch 5~6 진행 중이었음
  - ETF가 포함된 잘못된 데이터로 학습 중이었음

- **해결 방안**:
  1. pykrx_sector_idx가 있는 종목만 사용 (ETF 자동 제외)
  2. 또는 종목명 키워드로 ETF 필터링 (KODEX, TIGER 등)

### [20:54] v3 학습 데이터 생성 시작 (ETF 제외 + 올바른 섹터 매핑)

- **스크립트 수정**: `generate_dataset_v2.py`
  1. ETF 필터링: `pykrx_sector_idx`가 있는 종목만 사용
  2. sector_id 매핑: `PYKRX_TO_LASPS_SECTOR` 매핑 테이블 추가
  3. `get_valid_stock_codes_v2()` → dict 반환 (code → pykrx_idx)

- **종목 필터링 결과**:
  | 항목 | 개수 |
  |------|------|
  | 전체 DB | 4,226 |
  | 65일 이상 데이터 | 4,139 |
  | pykrx 미매핑 제외 (ETF/우선주) | 1,942 |
  | **최종 유효 종목** | **2,197** |

- **출력 경로**: `data/processed_v3/`
- **실행 명령어**: `python scripts/generate_dataset_v2.py --output data/processed_v3`
- **상태**: ✅ 완료 (23:16)

### [23:16] v3 학습 데이터 생성 완료

- **최종 결과**:
  | Split | 샘플 수 | 종목 수 |
  |-------|---------|---------|
  | Train | 610,424 | 2,116 |
  | Val | 97,832 | - |
  | Test | 101,758 | 2,112 |
  | **Total** | **810,014** | - |

- **레이블 분포** (Test 기준):
  | Label | 개수 | 비율 |
  |-------|------|------|
  | SELL (0) | 29,047 | 28.5% |
  | HOLD (1) | 52,637 | 51.7% |
  | BUY (2) | 20,074 | 19.7% |

- **Feature Shape**: (60, 28)
- **차트 이미지**: 101,758개 생성 (224x224 PNG)
- **소요 시간**: 141.9분 (약 2시간 22분)

- **개선 사항** (v2 대비):
  1. ETF/우선주 완전 제외 (pykrx 매핑 기반)
  2. 올바른 sector_id 매핑 (PYKRX_TO_LASPS_SECTOR 테이블)
  3. v2: 1,183,031 → v3: 810,014 (31.5% 감소, 노이즈 데이터 제거)

### [23:30] v3 데이터 매핑 검증 완료

- **인덱스 매핑 검증**: ✅ 정상
  | 항목 | 결과 |
  |------|------|
  | chart_idx와 파일명 | 일치 (000000.png ~ 610423.png) |
  | ts, labels, sector_ids, metadata | 모두 동일 개수 |
  | 차트 이미지 크기 | 224x224 정상 |
  | 이미지 내용 | 비어있지 않음 (mean: 238~246) |

- **섹터 매핑 검증**: ✅ 정상
  | 종목 | pykrx | sector_id | 결과 |
  |------|-------|-----------|------|
  | 삼성전자 | 1013:전기전자 | 0:전기전자 | ✓ |
  | SK하이닉스 | 1013:전기전자 | 0:전기전자 | ✓ |
  | NAVER | 1046:IT서비스 | 2:서비스업 | ✓ |
  | 카카오 | 1046:IT서비스 | 2:서비스업 | ✓ |
  | KT | 1020:통신 | 15:통신업 | ✓ |
  | SK텔레콤 | 1020:통신 | 15:통신업 | ✓ |
  | 신한지주/KB금융 | 1021:금융 | 1:금융업 | ✓ |
  | 셀트리온 | 1009:의약 | 3:의약품 | ✓ |

- **v2 → v3 섹터 분포 개선**:
  | 지표 | v2 | v3 | 변화 |
  |------|-----|-----|------|
  | 통신업(15) 비율 | 31.72% | 0.28% | **수정됨** |
  | Top 2 섹터 비율 | 67.1% | 56.5% | 개선 |
  | Max/Min 비율 | 184x | 168x | 개선 |

- **섹터 17 (제조업 기타) 47% 설명**:
  - KOSDAQ 제조(pykrx 2024)에 1,115개 종목 존재
  - 한국 주식시장 특성상 제조업 종목이 많음 (정상)

- **결론**: 인덱스 매핑 및 섹터 매핑 모두 정상, v3 학습 준비 완료

## 2026-02-13

### [00:15] 섹터 20개 → 13개 병합

- **이유**: 섹터 불균형 완화 및 모델 단순화
  - 기존: Max/Min 비율 167.8x (섹터 17 vs 섹터 15)
  - 통신업(0.28%), 전기가스(0.55%) 등 소수 섹터 통합

- **새 섹터 구조** (13개):
  | ID | 섹터명 | 샘플 수 | 비율 | 비고 |
  |----|--------|--------:|-----:|------|
  | 0 | 전기전자 | 26,845 | 3.31% | |
  | 1 | 금융업 | 44,858 | 5.54% | |
  | 2 | 서비스업 | 75,886 | 9.37% | |
  | 3 | 의약품 | 20,291 | 2.51% | |
  | 4 | 유통업 | 66,795 | 8.25% | |
  | 5 | 철강금속 | 23,284 | 2.87% | |
  | 6 | 기계 | 26,090 | 3.22% | |
  | 7 | 화학 | 12,636 | 1.56% | |
  | 8 | 건설업 | 43,741 | 5.40% | |
  | 9 | 음식료품 | 16,440 | 2.03% | |
  | 10 | 운수 | 35,328 | 4.36% | 운수장비+운수창고 |
  | 11 | 제조업 | 381,732 | 47.13% | KOSDAQ 제조 |
  | 12 | 기타 | 36,088 | 4.46% | 섬유,비금속,종이,통신,전기가스 등 |

- **수정 파일**:
  - `data/processed_v3/*/sector_ids.npy`: 변환 완료
  - `lasps/config/model_config.py`: num_sectors 20 → 13
  - `lasps/utils/constants.py`: NUM_SECTORS, SECTOR_NAMES, OLD_TO_NEW_SECTOR 추가

- **개선 효과**:
  - Max/Min 비율: 167.8x → 30.2x (381,732 / 12,636)
  - 사용 섹터: 18개 → 13개 (모두 활성)

### [00:45] Phase 1 학습 방식 변경 (방식 B: 단일 공통 헤드)

- **변경 이유**: 섹터 불균형(30.2배)으로 인한 Backbone 편향 방지
  - 기존 (방식 A): Phase 1에서 섹터별 라우팅 → 대형 섹터에 Backbone 편향
  - 변경 (방식 B): Phase 1에서 단일 공통 헤드 → Backbone이 일반 패턴 학습

- **새로운 3-Phase 학습 전략**:
  | Phase | 헤드 | Backbone | 학습 내용 |
  |-------|------|----------|----------|
  | 1 | common_head | 학습 | 일반적인 주식 패턴 (섹터 무관) |
  | 2 | sector_heads | 동결 | 섹터별 특화 패턴 |
  | 3 | sector_heads | 학습 | 전체 미세 조정 |

- **모델 구조 변경** (`sector_aware_model.py`):
  ```python
  # 추가된 구조
  self.common_head = nn.Sequential(...)  # Phase 1용
  self.sector_heads = nn.ModuleList([...])  # Phase 2/3용

  def forward(self, ts, img, sid, phase=1):
      shared_feat = self._encode_backbone(ts, img)
      if phase == 1:
          logits = self.common_head(shared_feat)  # 단일 헤드
      else:
          logits = self._route_to_sector_heads(shared_feat, sid)  # 섹터별 라우팅
  ```

- **새로운 메서드**:
  - `_encode_backbone()`: Backbone만 실행하는 공통 로직
  - `init_sector_heads_from_common()`: Phase 2 시작 시 common_head 가중치로 sector_heads 초기화
  - `get_all_sector_head_params()`: 모든 섹터 헤드 파라미터 반환

- **Trainer 변경** (`trainer.py`):
  - `_current_phase` 상태 변수 추가
  - `_run_epoch()`에서 `model(ts, img, sid, phase=self._current_phase)` 호출
  - `train_phase2(init_from_common=True)`: common_head → sector_heads 가중치 복사
  - `train_phase3(checkpoint_dir)`: Phase 3 체크포인트 저장 지원

- **기대 효과**:
  | 지표 | 방식 A (기존) | 방식 B (변경) |
  |------|--------------|--------------|
  | 소형 섹터 성능 | 나쁨 | 좋음 |
  | Backbone 일반화 | 편향 위험 | 좋음 |
  | 학습 효율성 | 병렬화 어려움 | 완전 병렬화 |

- **수정 파일**:
  - `lasps/models/sector_aware_model.py`: common_head 추가, forward에 phase 파라미터
  - `lasps/training/trainer.py`: _current_phase 관리, phase 전달

### [01:30] 섹터 및 피처 차원 코드 검증 완료

- **검증 범위**: num_sectors=20→13, input_dim=25→28 변경이 모든 코드에 반영되었는지 확인

- **수정된 파일 목록**:
  | 파일 | 변경 내용 |
  |------|----------|
  | `lasps/config/sector_config.py` | 13 섹터 구조 전면 재작성 |
  | `lasps/config/model_config.py` | num_sectors: 20 → 13 |
  | `lasps/utils/constants.py` | NUM_SECTORS=13, SECTOR_NAMES 매핑 |
  | `lasps/models/linear_transformer.py` | 문서: input_dim=28 |
  | `lasps/services/predictor.py` | forward_efficient() → forward(phase=2) |
  | `scripts/historical_data.py` | 문서: (N, 60, 28) |
  | `scripts/daily_batch.py` | 문서: (60, 28) |
  | `tests/test_config.py` | 13 섹터, 28 피처 검증 |
  | `tests/test_models.py` | input_dim=28, tensor shape (60, 28) |
  | `tests/test_sector_model.py` | num_sectors=13, Phase 1/2 테스트 |
  | `tests/test_predictor.py` | num_sectors=13, ts_input_dim=28 |
  | `tests/test_training.py` | ts_input_dim=28, tensor (60, 28) |
  | `tests/test_utils.py` | tensor shape (60, 28) |
  | `tests/test_dataset.py` | (60, 28), sector_ids [0,13) |
  | `tests/test_historical_pipeline.py` | 13 섹터, (60, 28) shape |

- **테스트 결과**: 40 passed, 0 failed ✓
  ```
  tests/test_config.py: 7 tests PASSED
  tests/test_models.py: 4 tests PASSED
  tests/test_sector_model.py: 9 tests PASSED
  tests/test_predictor.py: 2 tests PASSED
  tests/test_training.py: 6 tests PASSED
  tests/test_utils.py: 9 tests PASSED
  tests/test_dataset.py: 3 tests PASSED
  ```

- **예외 처리**: `generate_dataset_v2.py`, `stock_dataset.py`의 `(60, 25)` 주석은 v1→v2 마이그레이션 문서로 유지

### [01:45] 3-Phase 학습 코드 검증 완료

- **검증 범위**: Phase 1/2/3 학습 코드가 새 모델 구조(common_head + sector_heads)를 올바르게 사용하는지 확인

- **Phase별 동작 검증**:
  | Phase | `_current_phase` | Head 사용 | Backbone | 검증 결과 |
  |-------|------------------|-----------|----------|-----------|
  | 1 | 1 | common_head | 학습 | ✓ PASSED |
  | 2 | 2 | sector_heads | 동결 | ✓ PASSED |
  | 3 | 3 | sector_heads | 학습 | ✓ PASSED |

- **Phase 2 학습 흐름** (`trainer.py:284-316`):
  ```python
  def train_phase2(self, sector_loaders, epochs_per_sector=10, lr=5e-4, init_from_common=True):
      self._current_phase = 2
      if init_from_common:
          self._base_model.init_sector_heads_from_common()  # common_head → sector_heads 복사
      self._base_model.freeze_backbone()  # encoder + fusion + common_head 동결
      for sector_id, loader in sector_loaders.items():
          params = list(self._base_model.get_sector_head_params(sector_id))
          optimizer = AdamW(params, lr=lr)
          # 해당 섹터 데이터로만 학습
      self._base_model.unfreeze_backbone()
  ```

- **Phase 3 학습 흐름** (`trainer.py:318-387`):
  ```python
  def train_phase3(self, train_loader, val_loader, epochs=5, lr=1e-5):
      self._current_phase = 3
      self._base_model.unfreeze_backbone()  # encoder + fusion 학습 가능
      # common_head는 동결 유지 (Phase 3에서 사용 안함)
      optimizer = AdamW(self.model.parameters(), lr=lr)
      # 전체 데이터로 fine-tuning (sector_heads 라우팅)
  ```

- **파라미터 상태 검증** (Phase 3 후):
  | 컴포넌트 | requires_grad | 설명 |
  |----------|---------------|------|
  | ts_encoder | True | Backbone 학습 가능 |
  | cnn | True | Backbone 학습 가능 |
  | shared_fusion | True | Backbone 학습 가능 |
  | common_head | False | 동결 (Phase 3에서 미사용) |
  | sector_heads | True | 학습 가능 |

- **추론 코드 반영 필요사항** (`lasps/services/predictor.py`):
  - 추론 시 `phase=2` 또는 `phase=3` 사용 (sector_heads 라우팅)
  - 이미 수정됨: `forward_efficient()` → `forward(phase=2)`
  ```python
  # predictor.py (수정 완료)
  out = self.model(ts, img, sid, phase=2)  # sector_heads 사용
  ```

- **테스트 결과**: 6 training tests PASSED

### [02:00] 추론 코드 phase 파라미터 반영 검증

- **검증 범위**: 모든 추론 관련 코드가 `phase=2` (sector_heads)를 사용하는지 확인

- **수정된 파일**:
  | 파일 | 변경 내용 | 상태 |
  |------|----------|------|
  | `lasps/services/predictor.py` | `forward(phase=2)` 사용 | ✅ 완료 |
  | `lasps/api/main.py` | `SECTOR_CODES` → `SECTOR_NAMES` (13섹터) | ✅ 완료 |
  | `scripts/validate_model.py` | `phase` 파라미터 추가 (기본값=2) | ✅ 완료 |
  | `scripts/daily_batch.py` | `SectorAwarePredictor` 사용 (변경 불필요) | ✅ 확인 |
  | `scripts/train_phase2.py` | `trainer.train_phase2()` 사용 (변경 불필요) | ✅ 확인 |

- **validate_model.py 수정사항**:
  ```python
  # 수정 전
  out = model(ts, img, sid)  # phase=1 기본값 (common_head) ❌

  # 수정 후
  def validate(model, val_loader, criterion, device, phase: int = 2):
      out = model(ts, img, sid, phase=phase)  # sector_heads ✓

  # CLI 옵션 추가
  --phase {1,2,3}  # 기본값: 2 (sector_heads)
  ```

- **api/main.py 수정사항**:
  ```python
  # 수정 전
  from lasps.config.sector_config import SECTOR_CODES  # 존재하지 않음 ❌

  # 수정 후
  from lasps.config.sector_config import SECTOR_NAMES, NUM_SECTORS
  return {"num_sectors": 13, "sectors": {0: "전기전자", ...}}
  ```

- **추론 시 phase 사용 가이드**:
  | 체크포인트 | 권장 phase | 설명 |
  |------------|------------|------|
  | phase1_best.pt | 1 | common_head만 학습됨 |
  | phase2_best.pt | 2 | sector_heads 학습됨 |
  | phase3_final.pt | 2 또는 3 | 전체 fine-tuning 완료 |

- **테스트 결과**: 2 predictor tests PASSED

### [02:15] Git 커밋: 섹터 병합 + 3-Phase 학습 구조 개선

- **커밋 해시**: `63e8e8a`
- **커밋 메시지**: `refactor: 섹터 20→13 병합 + 3-Phase 학습 구조 개선`

- **변경된 파일 (23개)**:
  | 카테고리 | 파일 |
  |----------|------|
  | 설정 | `sector_config.py`, `model_config.py`, `constants.py` |
  | 모델 | `sector_aware_model.py`, `linear_transformer.py` |
  | 학습 | `trainer.py`, `train.py`, `train_phase2.py` (신규) |
  | 추론 | `predictor.py`, `validate_model.py` (신규), `api/main.py`, `daily_batch.py` |
  | 데이터 | `stock_dataset.py`, `historical_data.py` |
  | 테스트 | 7개 테스트 파일 업데이트, `test_historical_pipeline.py` (신규) |
  | 문서 | `donelog.md` (신규) |

- **주요 변경 요약**:
  1. 섹터 20개 → 13개 병합 (Max/Min 비율: 167.8x → 30.2x)
  2. Phase 1: common_head 사용 (일반 패턴 학습)
  3. Phase 2/3: sector_heads 사용 (섹터별 라우팅)
  4. 모든 추론 코드에 `phase=2` 적용
  5. 피처 차원 25 → 28 정리

---

### [20:10] pykrx 섹터 매핑 문제 근본 원인 분석

- **pykrx 섹터 정보 저장 위치**: 별도 저장 없음
  - `update_sector_ids.py` 실행 시 pykrx API 직접 호출
  - 결과를 `data/lasps.db`의 `stocks.sector_id`에 업데이트
  - 결과를 `data/processed/*/sector_ids.npy`에 업데이트

- **pykrx 실제 업종 인덱스** (`pykrx_sectors.json` 저장):
  | Index | Name | Count |
  |-------|------|-------|
  | 1020 | **통신** | 5 |
  | 1024 | **증권** | 18 |
  | 1027 | 제조 | 495 |

- **`update_sector_ids.py` 매핑 오류 발견**:
  ```python
  # 잘못된 매핑 (현재 코드)
  PYKRX_TO_LASPS = {
      "1024": 15,  # 증권(18종목) → 통신업 ❌
      "1020": 19,  # 통신(5종목) → 광업 ❌
  }

  # 올바른 매핑 (수정 필요)
  PYKRX_TO_LASPS = {
      "1020": 15,  # 통신(5종목) → 통신업 ✓
      "1024": 1,   # 증권(18종목) → 금융업 ✓ (또는 별도 증권업 ID)
  }
  ```

- **결론**:
  1. `update_sector_ids.py`가 실행되지 않았거나 실패함
  2. 실행되었더라도 매핑이 완전히 잘못되어 있음
  3. 모든 종목이 `DEFAULT_SECTOR_ID = 17`로 남아있음

- **조치 계획**:
  1. Phase 1 학습은 계속 진행 (Backbone은 sector_id를 입력 피처로 사용하지 않음)
  2. `PYKRX_TO_LASPS` 매핑 전면 수정 필요
  3. Phase 2 시작 전 `update_sector_ids.py` 재실행하여 sector_ids.npy 수정

---

### [17:52] Phase 1 학습 재개 (Epoch 5부터)

- **문제**: 컴퓨터 종료로 학습 중단 (Epoch 4 완료 후)
- **해결**: Resume 기능 구현 및 학습 재개
- **수정 파일**:
  - `scripts/train.py`: `--resume-epoch`, `--no-early-stop` 옵션 추가
  - `lasps/training/trainer.py`: `train_phase1()`에 `start_epoch` 파라미터 추가
- **실행 명령어**:
  ```bash
  python scripts/train.py --data-dir data/processed_v2 --device cuda --multi-gpu --resume-epoch 4 --no-early-stop
  ```
- **설정**:
  - Epoch 4 체크포인트 로드 → Epoch 5부터 재개
  - LR Scheduler를 epoch 4 위치로 advance (lr=0.000097)
  - Early stopping 비활성화 (35 epoch 전체 학습)
  - num_workers=4, batch_size=128 (GPU당 64)
- **예상 종료**: 31 epoch × 30분 ≈ 15.5시간

---

### [작업] 학습 속도 개선 - num_workers=4 적용

- **문제 발견**: PNG 파일 로딩이 병목
  - 85만개 PNG 파일을 매 배치마다 로드
  - `num_workers=0` (Windows 기본값) → 단일 프로세스로 I/O 대기
  - epoch당 수 시간 이상 소요 예상

- **시도 1**: memmap NPY 변환
  - 이전에 시도했으나 **랜덤 액세스 시 더 느려짐** (shuffle=True 때문)
  - 디스크 시크 오버헤드가 더 큼

- **시도 2**: `num_workers=4` (성공)
  - **수정 파일**: `scripts/train.py:143-147`
    ```python
    # 변경 전
    num_workers = 0 if platform.system() == "Windows" else 4

    # 변경 후
    num_workers = 4
    ```
  - **결과**: 100 batch당 27초 → epoch당 ~30분 예상
  - Windows에서도 정상 작동 확인

- **추가 수정**: 배치 진행률 로그 추가
  - **파일**: `lasps/training/trainer.py:164-166`
  - 100 batch마다 `Batch [N/6675] loss=X.XXXX` 출력

- **학습 시작**: 2026-02-12 15:53:27
  - ID: `bb92897`
  - 설정: Multi-GPU (2x RTX 2080 Ti), batch_size=128, num_workers=4
  - 명령어: `python scripts/train.py --data-dir data/processed_v2 --device cuda --multi-gpu`

---

### [작업] 데이터셋 검증 및 차트 파일명 불일치 수정

- **문제 발견**: `StockDataset`의 차트 이미지 로드 코드와 실제 파일명 불일치
  - 코드: `chart_{idx:07d}.png` (예: `chart_0000000.png`)
  - 실제 파일: `{idx:06d}.png` (예: `000000.png`)
  - 결과: 차트 이미지를 찾지 못하고 빈 배열(zeros) 반환

- **수정 파일**: `lasps/data/datasets/stock_dataset.py:106`
  ```python
  # 변경 전
  png_path = self.charts_dir / f"chart_{idx:07d}.png"

  # 변경 후
  png_path = self.charts_dir / f"{idx:06d}.png"
  ```

- **데이터셋 현황 확인** (`data/processed_v2/`):
  | Split | 샘플 수 | 시계열 Shape | 차트 PNG |
  |-------|---------|--------------|----------|
  | Train | 854,326 | (60, 28) | 854,326 |
  | Val | 156,187 | (60, 28) | 156,187 |
  | Test | 172,518 | (60, 28) | 172,518 |
  | **Total** | **1,183,031** | | |

- **인덱스 매핑 확인**: `metadata.csv` 파일로 매핑됨
  - 컬럼: `stock_code`, `date`, `chart_idx`
  - `time_series[i]`, `labels[i]`, `sector_ids[i]`, `{i:06d}.png` 모두 동일 샘플

### [작업] 모델 input_dim 불일치 수정

- **문제 발견**: `SectorAwareFusionModel`이 config의 `input_dim`을 무시
  - config: `linear_transformer.input_dim = 28`
  - 모델: `ts_input_dim: int = 25` (하드코딩 기본값)
  - 결과: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (120x28 and 25x128)`

- **수정 파일**: `lasps/models/sector_aware_model.py:35`
  ```python
  # 변경 전
  ts_input_dim: int = 25,

  # 변경 후
  ts_input_dim: Optional[int] = None,
  # config에서 input_dim 읽기, 명시적 인자가 우선
  actual_input_dim = ts_input_dim if ts_input_dim is not None else lt_cfg["input_dim"]
  ```

- **검증 결과**:
  - 입력: `ts=(1, 60, 28)`, `chart=(1, 3, 224, 224)`, `sector=(1,)`
  - 출력: `logits=(1, 3)`, `probs=[0.32, 0.30, 0.38]`
  - 모델 파라미터: 1,436,348개
  - **학습 준비 완료** ✓

### [참고] 모델 파라미터 저장 경로

- **기본 디렉토리**: `checkpoints/`
- **저장 파일 형식**:
  | Phase | 파일명 패턴 | 예시 |
  |-------|-------------|------|
  | Phase 1 (Backbone) | `phase1_epoch_{NN}.pt` | `phase1_epoch_01.pt` ~ `phase1_epoch_35.pt` |
  | Phase 2 (Sector Heads) | `phase2_sector_{SS}_epoch_{NN}.pt` | `phase2_sector_00_epoch_01.pt` |
  | Phase 3 (Fine-tune) | `phase3_epoch_{NN}.pt` | `phase3_epoch_01.pt` ~ `phase3_epoch_08.pt` |

- **체크포인트 내용**:
  ```python
  {
      "epoch": int,
      "model_state_dict": dict,
      "optimizer_state_dict": dict,
      "train_loss": float,
      "val_loss": float,
  }
  ```

- **학습 명령어**:
  ```bash
  python scripts/train.py --data-dir data/processed_v2 --device cuda --output-dir checkpoints
  ```

---

### [작업] v2 학습 데이터 생성 시스템 구현
- **작업**: ETF 제외 + 65일 미만 종목 제외 + temporal features 추가
- **변경 파일**:
  - `scripts/generate_dataset_v2.py` - 신규 생성 (v2 데이터셋 생성기)
  - `lasps/config/model_config.py` - input_dim 29→28
  - `lasps/utils/constants.py` - TEMPORAL_DIM 4→3, feature 이름 변경
  - `lasps/data/datasets/stock_dataset.py` - v1/v2 데이터 모두 지원
  - `CLAUDE.md` - 28-feature 문서화
- **Feature 구조 변경**:
  - 기존 (60, 25) + 4 temporal (sin/cos) = (60, 29)
  - 신규 (60, 25) + 3 temporal (weekday/month/day) = (60, 28)
- **Temporal Features (v2)**:
  - weekday: 요일 정규화 (월=0, 금=0.8)
  - month: 월 정규화 (1월=0.08, 12월=1.0)
  - day: 일 정규화 (1일=0.03, 31일=1.0)
- **실행 명령어**: `python scripts/generate_dataset_v2.py --output data/processed_v2`

---

## 2026-02-11

### [21:30] E: memmap 차트 데이터 문제 발견
- **작업**: E:/stockquant_data/train_chart_images.npy 파일 분석
- **결과**: 465GB 파일이 **전부 0**으로 채워져 있음 발견
  - C: PNG 파일: 정상 (min=0, max=1, mean≈0.94)
  - E: memmap: 비정상 (모든 값이 0)
- **영향**: 현재 학습이 빈 차트 이미지(검은 화면)로 진행 중이었음

### [21:35] Phase 1 학습 중단
- **작업**: 잘못된 데이터로 진행 중이던 학습 중단
- **명령어**: `KillShell b0f08ab`
- **결과**: 13/35 epoch까지 진행된 학습 중단
- **비고**:
  - 약 13시간 소요 (59분/epoch)
  - 차트 이미지가 0이었으므로 시계열만으로 학습된 불완전한 모델
  - checkpoints/phase1_epoch_01.pt ~ phase1_epoch_13.pt 생성됨 (의미 없음)

### [21:40] 작업 로그 규칙 추가
- **작업**: CLAUDE.md에 작업 로그 기록 규칙 추가
- **파일**:
  - CLAUDE.md - "Claude 작업 규칙" 섹션 추가
  - docs/plans/donelog.md - 로그 파일 생성

---

## 대기 중인 작업

- [ ] `--chart-dir` 옵션 없이 C: PNG 파일로 Phase 1 재학습
  - 예상 명령어: `python scripts/train.py --data-dir data/processed --device cuda --multi-gpu`
  - 예상 속도: ~30분/epoch
