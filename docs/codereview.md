# LASPS v7a 전체 코드 리뷰

## 전체 평가

**전반적으로 잘 구성된 프로젝트입니다.** 59개 테스트 통과, 일관된 코딩 컨벤션, 명확한 모듈 분리가 잘 되어 있습니다. 아래는 심각도별로 분류한 이슈들입니다.

---

## Critical (즉시 수정 필요)

### 1. `sector_id = -1` 처리 누락
`sector_config.py:30` - `get_sector_id()`가 매핑되지 않는 업종코드에 대해 `-1`을 반환합니다. 그러나 `SectorAwareFusionModel`에서 `sector_heads[-1]`은 **마지막 sector head(19번)**를 참조하게 되어 잘못된 예측이 발생합니다. 런타임 에러가 아닌 **조용한 오류**라 위험합니다.

```python
# sector_config.py:30
def get_sector_id(sector_code: str) -> int:
    if sector_code in SECTOR_CODES:
        return SECTOR_CODES[sector_code][0]
    return -1  # <- sector_heads[-1] = sector 19로 라우팅됨
```

**수정 방안:** 기본 섹터를 별도 정의하거나, IntegratedCollector/Predictor에서 `-1` 검증 추가

### 2. `forward_efficient` gradient 단절 가능성
`sector_aware_model.py:113` - `logits = torch.zeros(...)` 로 초기화된 텐서에 인덱싱으로 값을 할당합니다. 이 방식은 동작하지만, 배치 내에 **특정 sector 샘플이 없으면** 해당 위치는 `zeros`로 남아 gradient가 0이 됩니다. Phase 2 학습에서 특히 문제가 될 수 있습니다 (이미 AdamW fix로 crash는 해결했지만 학습 효과 자체가 0).

---

## Important (권장 수정)

### 3. Trainer에 best model 저장 / early stopping 없음
`trainer.py` - 현재 Phase 1, 3에서 **마지막 epoch의 loss만 반환**하고, validation loss 기반 best model을 저장하지 않습니다. 실제 학습 시 overfitting을 방지할 수 없습니다.

```python
# 현재: 마지막 epoch 결과만
return {"train_loss": train_loss, "val_loss": val_loss}
```

**수정 방안:** `best_val_loss` 추적 + 체크포인트 저장 콜백 추가

### 4. Phase 2 `_run_epoch`에서 전체 모델의 gradient를 clip
`trainer.py:61` - Phase 2에서 backbone이 frozen인데도 `[p for p in self.model.parameters() if p.requires_grad]`로 전체 trainable params의 gradient를 clip합니다. Phase 2에서는 현재 학습 중인 sector head의 파라미터만 clip하는 것이 정확합니다 (다른 sector head의 gradient까지 영향 받음).

### 5. `time.sleep` in KiwoomCollector
`kiwoom_collector.py:19` - 모든 API 호출마다 `time.sleep(interval)`이 있습니다. Mock 사용 시에도 불필요한 지연이 발생합니다. 테스트에서 `days=180` x 3 API 호출 = 실제 대기 시간이 누적됩니다.

**수정 방안:** Mock API일 때 sleep 건너뛰기 또는, `_request` 대신 직접 호출하도록 `KiwoomCollector` 생성자에 `rate_limit=True/False` 파라미터 추가

### 6. `IntegratedCollector.collect_stock_data` - 25개 feature 미달 시 무검증
`integrated_collector.py:35` - `feature_cols = [c for c in all_feat if c in merged.columns]` 로 존재하는 컬럼만 선택합니다. 만약 일부 feature 계산이 실패하면 25개 미만이 되어 모델 입력 shape이 맞지 않게 됩니다.

```python
# 검증 없이 진행
feature_cols = [c for c in all_feat if c in merged.columns]
# -> len(feature_cols) < 25면 모델에서 shape mismatch crash
```

### 7. `CLAUDE.md`에 `PyTorch 2.0+` 라고 기재되어 있지만 실제 환경은 1.8.1
```
- **PyTorch 2.0+** - 모델 및 학습
```
이미 코드는 1.8.1 호환으로 작성되었으나 문서가 불일치합니다.

---

## Minor (선택적 개선)

### 8. `kiwoom_collector.py` 미사용 import
```python
import numpy as np      # 사용 안 됨
from typing import Optional  # 사용 안 됨
```

### 9. QVMScreener의 "Momentum" 정의 이슈
`qvm_screener.py` - `m_score`가 `volume_avg_20 + market_cap` rank인데, market_cap은 모멘텀보다는 size factor에 가깝습니다. PRD 설계대로이므로 코드 이슈는 아니지만, 금융학적으로는 가격 모멘텀(3/6/12개월 수익률)이 더 적절합니다.

### 10. `api/main.py`의 `@app.on_event("startup")` deprecation
FastAPI 0.95+에서는 `lifespan` context manager가 권장됩니다. 현재 작동하지만 향후 마이그레이션 필요.

### 11. `StockDataset` - `chart_images` mmap이 메모리 문제 유발 가능
`stock_dataset.py:15` - `chart_images`를 `mmap_mode="r"`로 로드하는데, `(N, 3, 224, 224)` float32로 1만 샘플이면 ~5.6GB입니다. mmap 자체는 OK이지만 `.copy()` 호출 시 매번 메모리 할당이 발생합니다. DataLoader `num_workers > 0`과 결합하면 OOM 위험이 있습니다.

### 12. `forward` vs `forward_efficient` 일관성
`forward()`는 `shared_features`를 반환하지만 `forward_efficient()`는 반환하지 않습니다. Predictor는 `forward_efficient()`만 사용하므로 문제는 없으나, 인터페이스 불일치입니다.

---

## 긍정적인 부분

- **모듈 분리가 깔끔합니다.** `collectors -> processors -> datasets -> models -> training -> services` 파이프라인이 명확합니다.
- **Mock API 설계가 우수합니다.** GBM 기반 현실적 가격 생성, Abstract Base Class로 Real/Mock 교체 가능.
- **PyTorch 1.8.1 호환성을 잘 처리했습니다.** `batch_first` 미지원 -> 수동 transpose, AdamW zero-grad 버그 -> `has_grad` 체크.
- **테스트 커버리지가 적절합니다.** 59개 테스트로 핵심 기능 검증. Shape 검증, dtype 검증, 통합 테스트 포함.
- **3-Phase Training 설계가 PRD와 정확히 일치합니다.** freeze/unfreeze, sector별 optimizer, cosine annealing 등.

---

## 요약

| 심각도 | 건수 | 핵심 |
|--------|------|------|
| **Critical** | 2 | sector_id=-1 라우팅 오류, gradient 단절 |
| **Important** | 5 | best model 미저장, grad clip 범위, sleep in mock, feature 수 검증, 문서 불일치 |
| **Minor** | 5 | 미사용 import, QVM 정의, API deprecation, mmap OOM, forward 인터페이스 |
