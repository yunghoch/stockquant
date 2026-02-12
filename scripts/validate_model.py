#!/usr/bin/env python
# scripts/validate_model.py

"""모델 검증 및 성능 측정 스크립트

Checkpoint 모델의 validation loss, accuracy, 클래스별 성능을 측정합니다.

Usage:
    python scripts/validate_model.py --checkpoint checkpoints/phase1_best.pt
    python scripts/validate_model.py --checkpoint checkpoints/phase2_best.pt
    python scripts/validate_model.py --checkpoint checkpoints/phase3_final.pt --split test
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import platform
import torch
import numpy as np
from torch.utils.data import DataLoader
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix

from lasps.models.sector_aware_model import SectorAwareFusionModel
from lasps.data.datasets.stock_dataset import StockDataset
from lasps.training.loss_functions import FocalLoss
from lasps.config.model_config import MODEL_CONFIG
from lasps.config.settings import settings
from lasps.utils.logger import setup_logger
from scripts.train import stock_collate_fn


def validate(model, val_loader, criterion, device, phase: int = 2) -> dict:
    """전체 validation 실행.

    Args:
        model: 평가할 모델
        val_loader: Validation DataLoader
        criterion: Loss 함수
        device: Device
        phase: Forward pass phase (2=sector_heads for trained model)

    Returns:
        결과 딕셔너리 (loss, accuracy, class별 통계)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            ts, img, sid, labels = [b.to(device) for b in batch]
            out = model(ts, img, sid, phase=phase)
            loss = criterion(out["logits"], labels)

            total_loss += loss.item()
            preds = out["logits"].argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    avg_loss = total_loss / len(val_loader)
    accuracy = (all_preds == all_labels).mean()

    # 클래스별 통계
    class_names = ["SELL", "HOLD", "BUY"]
    class_correct = {}
    class_total = {}

    for i, name in enumerate(class_names):
        mask = all_labels == i
        class_total[name] = mask.sum()
        class_correct[name] = (all_preds[mask] == i).sum() if mask.sum() > 0 else 0

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "predictions": all_preds,
        "labels": all_labels,
        "class_correct": class_correct,
        "class_total": class_total,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="LASPS v7a Model Validation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint to evaluate")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Directory containing processed data")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"],
                        help="Which data split to use (val or test)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Evaluation device (cuda or cpu)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for evaluation")
    parser.add_argument("--chart-dir", type=str, default=None,
                        help="Directory containing chart NPY files")
    parser.add_argument("--phase", type=int, default=2, choices=[1, 2, 3],
                        help="Model forward phase (1=common_head, 2/3=sector_heads)")
    return parser.parse_args()


def main() -> None:
    """Run model validation."""
    args = parse_args()
    setup_logger("INFO")
    device = args.device

    # Load model
    model = SectorAwareFusionModel(
        num_sectors=MODEL_CONFIG["num_sectors"],
        ts_input_dim=MODEL_CONFIG["linear_transformer"]["input_dim"],
    )

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(str(ckpt_path), map_location=device)
    # Handle checkpoint dict format
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded: {ckpt_path}")

    # Load data
    data_dir = Path(args.data_dir)
    chart_dir = Path(args.chart_dir) if args.chart_dir else data_dir

    split_dir = args.split
    ds = StockDataset(
        time_series_path=str(data_dir / split_dir / "time_series.npy"),
        chart_images_path=str(chart_dir / f"{split_dir}_chart_images.npy") if args.chart_dir else str(data_dir / split_dir / "chart_images.npy"),
        sector_ids_path=str(data_dir / split_dir / "sector_ids.npy"),
        labels_path=str(data_dir / split_dir / "labels.npy"),
    )

    num_workers = 0 if platform.system() == "Windows" else 4
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=stock_collate_fn,
    )

    # Loss function
    class_weights = torch.tensor(settings.CLASS_WEIGHTS, dtype=torch.float32).to(device)
    criterion = FocalLoss(num_classes=3, gamma=2.0, weight=class_weights)

    # Run validation
    logger.info(f"Evaluating on {args.split} set ({len(ds)} samples), phase={args.phase}...")
    results = validate(model, loader, criterion, device, phase=args.phase)

    # Print results
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split:      {args.split}")
    print(f"Phase:      {args.phase} ({'common_head' if args.phase == 1 else 'sector_heads'})")
    print(f"Samples:    {len(ds)}")
    print("-" * 60)
    print(f"Loss:       {results['loss']:.4f}")
    print(f"Accuracy:   {results['accuracy'] * 100:.2f}%")
    print("-" * 60)
    print("Class-wise Performance:")
    for cls in ["SELL", "HOLD", "BUY"]:
        total = results["class_total"][cls]
        correct = results["class_correct"][cls]
        acc = correct / total * 100 if total > 0 else 0
        print(f"  {cls:4s}: {acc:5.2f}% ({correct:5d} / {total:5d})")
    print("=" * 60)

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(results["labels"], results["predictions"])
    print("          Predicted")
    print("          SELL  HOLD   BUY")
    for i, row_name in enumerate(["SELL", "HOLD", "BUY "]):
        print(f"Actual {row_name} {cm[i][0]:5d} {cm[i][1]:5d} {cm[i][2]:5d}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        results["labels"],
        results["predictions"],
        target_names=["SELL", "HOLD", "BUY"],
        digits=4
    ))


if __name__ == "__main__":
    main()
