# Phase 2 구현 및 모델 병합 계획

## 목표
Phase 1→3 학습 완료된 모델에 Phase 2 (섹터별 헤드 학습)를 추가하여 정확도 향상

## 현재 상태

```
✅ Phase 1: Backbone 학습 (30 epochs) - 구현 완료
❌ Phase 2: Sector Heads 학습 - 미구현 (로그만 출력)
✅ Phase 3: Fine-tuning (5 epochs) - 구현 완료
```

## 예상 효과

| 모델 | 예상 정확도 |
|------|------------|
| Phase 1→3 (현재) | 55~60% |
| Phase 1→3→2 (병합 후) | 58~63% (+3%) |

---

## 구현 단계

### Step 1: 섹터별 데이터셋 필터링 기능 추가

**파일**: `lasps/data/datasets/stock_dataset.py`

```python
# 기존 StockDataset 클래스에 메서드 추가

def get_sector_indices(self, sector_id: int) -> np.ndarray:
    """특정 섹터의 샘플 인덱스 반환"""
    return np.where(self.sector_ids == sector_id)[0]

def get_sector_subset(self, sector_id: int) -> "SectorSubset":
    """특정 섹터의 Subset 반환"""
    indices = self.get_sector_indices(sector_id)
    return SectorSubset(self, indices, sector_id)


class SectorSubset(Dataset):
    """특정 섹터만 포함하는 Subset"""

    def __init__(self, dataset: StockDataset, indices: np.ndarray, sector_id: int):
        self.dataset = dataset
        self.indices = indices
        self.sector_id = sector_id

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        return self.dataset[self.indices[idx]]
```

---

### Step 2: train.py Phase 2 로직 구현

**파일**: `scripts/train.py`

```python
# Line 168-171 수정

if args.phase in (0, 2):
    cfg = THREE_PHASE_CONFIG["phase2_sector_heads"]
    logger.info("Phase 2: Building sector-specific DataLoaders")

    # 섹터별 DataLoader 생성
    sector_loaders = {}
    for sector_id in range(MODEL_CONFIG["num_sectors"]):
        sector_subset = train_ds.get_sector_subset(sector_id)

        if len(sector_subset) < cfg.get("min_samples", 100):
            logger.warning(f"Sector {sector_id}: {len(sector_subset)} samples (skipping)")
            continue

        sector_loaders[sector_id] = DataLoader(
            sector_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=stock_collate_fn,
        )
        logger.info(f"Sector {sector_id}: {len(sector_subset)} samples")

    # Phase 2 학습 실행
    trainer.train_phase2(
        sector_loaders,
        epochs_per_sector=cfg["epochs_per_sector"],
        lr=cfg["lr"],
    )
    torch.save(get_state_dict(), output_dir / "phase2_best.pt")
    logger.info("Phase 2 complete")
```

---

### Step 3: Phase 2 전용 실행 스크립트 (선택)

**파일**: `scripts/train_phase2.py` (신규)

```python
#!/usr/bin/env python
"""Phase 2 전용 학습 스크립트

기존 checkpoint를 로드하여 Phase 2만 실행

Usage:
    python scripts/train_phase2.py --checkpoint checkpoints/phase3_final.pt
"""

import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from loguru import logger

from lasps.models.sector_aware_model import SectorAwareFusionModel
from lasps.data.datasets.stock_dataset import StockDataset
from lasps.training.trainer import ThreePhaseTrainer
from lasps.config.model_config import MODEL_CONFIG, THREE_PHASE_CONFIG
from scripts.train import stock_collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    # 모델 로드
    model = SectorAwareFusionModel(
        num_sectors=MODEL_CONFIG["num_sectors"],
        ts_input_dim=MODEL_CONFIG["linear_transformer"]["input_dim"],
    )
    state_dict = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state_dict)
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # Trainer 생성
    trainer = ThreePhaseTrainer(model, device=args.device)

    # 데이터 로드
    data_dir = Path(args.data_dir)
    train_ds = StockDataset(
        time_series_path=str(data_dir / "train" / "time_series.npy"),
        chart_images_path=str(data_dir / "train" / "chart_images.npy"),
        sector_ids_path=str(data_dir / "train" / "sector_ids.npy"),
        labels_path=str(data_dir / "train" / "labels.npy"),
    )

    # 섹터별 DataLoader 생성
    sector_loaders = {}
    for sector_id in range(MODEL_CONFIG["num_sectors"]):
        sector_subset = train_ds.get_sector_subset(sector_id)
        if len(sector_subset) < 100:
            logger.warning(f"Sector {sector_id}: {len(sector_subset)} samples (skipping)")
            continue
        sector_loaders[sector_id] = DataLoader(
            sector_subset, batch_size=32, shuffle=True,
            num_workers=0, pin_memory=True, collate_fn=stock_collate_fn,
        )
        logger.info(f"Sector {sector_id}: {len(sector_subset)} samples")

    # Phase 2 실행
    trainer.train_phase2(sector_loaders, epochs_per_sector=args.epochs, lr=args.lr)

    # 저장
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "phase2_best.pt")
    logger.info(f"Saved: {output_dir / 'phase2_best.pt'}")


if __name__ == "__main__":
    main()
```

---

### Step 4: Validation 및 Loss 계산 스크립트

**파일**: `scripts/validate_model.py` (신규)

```python
#!/usr/bin/env python
"""모델 검증 및 Loss 계산

Usage:
    python scripts/validate_model.py --checkpoint checkpoints/phase2_best.pt
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from loguru import logger

from lasps.models.sector_aware_model import SectorAwareFusionModel
from lasps.data.datasets.stock_dataset import StockDataset
from lasps.training.loss_functions import FocalLoss
from lasps.config.model_config import MODEL_CONFIG
from lasps.config.settings import settings
from scripts.train import stock_collate_fn


def validate(model, val_loader, criterion, device):
    """전체 validation 실행"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]

    with torch.no_grad():
        for batch in val_loader:
            ts, img, sid, labels = [b.to(device) for b in batch]
            out = model(ts, img, sid)
            loss = criterion(out["logits"], labels)

            total_loss += loss.item()
            preds = out["logits"].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # 클래스별 정확도
            for c in range(3):
                mask = labels == c
                class_correct[c] += (preds[mask] == c).sum().item()
                class_total[c] += mask.sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "class_accuracy": {
            "SELL": class_correct[0] / max(class_total[0], 1),
            "HOLD": class_correct[1] / max(class_total[1], 1),
            "BUY": class_correct[2] / max(class_total[2], 1),
        },
        "class_counts": {
            "SELL": class_total[0],
            "HOLD": class_total[1],
            "BUY": class_total[2],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = args.device

    # 모델 로드
    model = SectorAwareFusionModel(
        num_sectors=MODEL_CONFIG["num_sectors"],
        ts_input_dim=MODEL_CONFIG["linear_transformer"]["input_dim"],
    )
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded: {args.checkpoint}")

    # 데이터 로드
    data_dir = Path(args.data_dir)
    val_ds = StockDataset(
        time_series_path=str(data_dir / "val" / "time_series.npy"),
        chart_images_path=str(data_dir / "val" / "chart_images.npy"),
        sector_ids_path=str(data_dir / "val" / "sector_ids.npy"),
        labels_path=str(data_dir / "val" / "labels.npy"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=stock_collate_fn,
    )

    # Loss 함수
    class_weights = torch.tensor(settings.CLASS_WEIGHTS, dtype=torch.float32).to(device)
    criterion = FocalLoss(num_classes=3, gamma=2.0, weight=class_weights)

    # Validation 실행
    results = validate(model, val_loader, criterion, device)

    # 결과 출력
    print("\n" + "="*50)
    print("VALIDATION RESULTS")
    print("="*50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Val Loss:   {results['loss']:.4f}")
    print(f"Accuracy:   {results['accuracy']*100:.2f}%")
    print()
    print("Class-wise Accuracy:")
    for cls, acc in results['class_accuracy'].items():
        cnt = results['class_counts'][cls]
        print(f"  {cls}: {acc*100:.2f}% ({cnt} samples)")
    print("="*50)


if __name__ == "__main__":
    main()
```

---

## 실행 순서

### 1단계: Phase 1→3 학습 완료 대기
```bash
# 현재 진행 중인 학습 완료 대기
# checkpoints/phase3_final.pt 생성 확인
```

### 2단계: Phase 2 구현
```bash
# stock_dataset.py 수정 (섹터 필터링 추가)
# train_phase2.py 생성
# validate_model.py 생성
```

### 3단계: Phase 2 학습 실행
```bash
python scripts/train_phase2.py \
    --checkpoint checkpoints/phase3_final.pt \
    --data-dir data/processed \
    --epochs 10 \
    --lr 5e-4
```

### 4단계: 모델 검증
```bash
# Phase 3 모델 (Phase 2 적용 전)
python scripts/validate_model.py --checkpoint checkpoints/phase3_final.pt

# Phase 2 모델 (Phase 2 적용 후)
python scripts/validate_model.py --checkpoint checkpoints/phase2_best.pt
```

### 5단계: 결과 비교
```
Phase 3 (적용 전): Loss=0.52, Accuracy=58%
Phase 2 (적용 후): Loss=0.48, Accuracy=61%  ← 예상
```

---

## 수정/생성 파일 목록

| 파일 | 작업 | 설명 |
|------|------|------|
| `lasps/data/datasets/stock_dataset.py` | 수정 | 섹터 필터링 메서드 추가 |
| `scripts/train.py` | 수정 | Phase 2 로직 구현 |
| `scripts/train_phase2.py` | 신규 | Phase 2 전용 스크립트 |
| `scripts/validate_model.py` | 신규 | 모델 검증 스크립트 |

---

## 예상 소요 시간

| 작업 | 시간 |
|------|------|
| Step 1: 섹터 필터링 구현 | 1시간 |
| Step 2: train.py 수정 | 1시간 |
| Step 3: train_phase2.py 생성 | 1시간 |
| Step 4: validate_model.py 생성 | 1시간 |
| Phase 2 학습 (20섹터 × 10epochs) | 5~10시간 |
| 검증 및 비교 | 30분 |

**총 예상: 10~15시간**

---

## 검증 방법

### 1. 섹터별 샘플 수 확인
```bash
python -c "
from lasps.data.datasets.stock_dataset import StockDataset
import numpy as np

ds = StockDataset(
    'data/processed/train/time_series.npy',
    'data/processed/train/chart_images.npy',
    'data/processed/train/sector_ids.npy',
    'data/processed/train/labels.npy',
)

for sid in range(20):
    indices = ds.get_sector_indices(sid)
    print(f'Sector {sid:2d}: {len(indices):6d} samples')
"
```

### 2. Phase 2 학습 로그 확인
```
Phase 2: Sector Heads (backbone frozen)
Phase2 Sector 0 [1/10] loss=0.85
Phase2 Sector 0 [2/10] loss=0.78
...
Phase2 Sector 19 [10/10] loss=0.65
```

### 3. 전후 비교
```bash
# 비교 스크립트
python scripts/validate_model.py --checkpoint checkpoints/phase3_final.pt
python scripts/validate_model.py --checkpoint checkpoints/phase2_best.pt
```

---

## 주의사항

1. **최소 샘플 수**: 섹터당 100개 미만이면 스킵 (과적합 방지)
2. **학습률**: Phase 2는 5e-4 (Phase 1보다 높음, 헤드만 학습)
3. **Backbone 동결**: Phase 2에서는 ts_encoder, cnn, shared_fusion 수정 안 함
4. **저장 시점**: 각 섹터 학습 완료 후가 아닌, 전체 Phase 2 완료 후 저장
