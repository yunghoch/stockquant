#!/usr/bin/env python
# scripts/train.py

"""
LASPS v7a 3-Phase 학습 스크립트

Usage:
    python scripts/train.py --data-dir data/processed --device cuda
    python scripts/train.py --phase 2 --checkpoint checkpoints/phase1_best.pt
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
from lasps.utils.logger import setup_logger


def stock_collate_fn(batch):
    """Convert StockDataset dict batches to tuple format for ThreePhaseTrainer.

    StockDataset returns dicts with keys: time_series, chart_image, sector_id, label.
    ThreePhaseTrainer._run_epoch expects tuples: (ts, img, sid, labels).

    Args:
        batch: List of dicts from StockDataset.__getitem__.

    Returns:
        Tuple of (time_series, chart_image, sector_id, label) tensors.
    """
    time_series = torch.stack([item["time_series"] for item in batch])
    chart_image = torch.stack([item["chart_image"] for item in batch])
    sector_id = torch.stack([item["sector_id"] for item in batch])
    label = torch.stack([item["label"] for item in batch])
    return time_series, chart_image, sector_id, label


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="LASPS v7a Training")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Directory containing processed data")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Training device (cuda or cpu)")
    parser.add_argument("--phase", type=int, default=0,
                        help="0=all phases, 1/2/3=specific phase")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    return parser.parse_args()


def main():
    """Run LASPS v7a 3-Phase training."""
    args = parse_args()
    setup_logger("INFO")
    logger.info(f"Device: {args.device}")

    model = SectorAwareFusionModel(
        num_sectors=MODEL_CONFIG["num_sectors"],
        ts_input_dim=MODEL_CONFIG["linear_transformer"]["input_dim"],
    )
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
        logger.info(f"Loaded checkpoint: {args.checkpoint}")

    trainer = ThreePhaseTrainer(model, device=args.device)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = StockDataset(
        time_series_path=str(data_dir / "train" / "time_series.npy"),
        chart_images_path=str(data_dir / "train" / "chart_images.npy"),
        sector_ids_path=str(data_dir / "train" / "sector_ids.npy"),
        labels_path=str(data_dir / "train" / "labels.npy"),
    )
    val_ds = StockDataset(
        time_series_path=str(data_dir / "val" / "time_series.npy"),
        chart_images_path=str(data_dir / "val" / "chart_images.npy"),
        sector_ids_path=str(data_dir / "val" / "sector_ids.npy"),
        labels_path=str(data_dir / "val" / "labels.npy"),
    )

    batch_size = 128
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, collate_fn=stock_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=stock_collate_fn,
    )

    if args.phase in (0, 1):
        cfg = THREE_PHASE_CONFIG["phase1_backbone"]
        trainer.train_phase1(train_loader, val_loader, epochs=cfg["epochs"], lr=cfg["lr"])
        torch.save(model.state_dict(), output_dir / "phase1_best.pt")
        logger.info("Phase 1 complete")

    if args.phase in (0, 2):
        # 섹터별 DataLoader 구성
        # 실제로는 sector_id별로 데이터를 분리해야 함
        logger.info("Phase 2: sector-specific training (requires sector-split data)")

    if args.phase in (0, 3):
        cfg = THREE_PHASE_CONFIG["phase3_finetune"]
        trainer.train_phase3(train_loader, val_loader, epochs=cfg["epochs"], lr=cfg["lr"])
        torch.save(model.state_dict(), output_dir / "phase3_final.pt")
        logger.info("Phase 3 complete")


if __name__ == "__main__":
    main()
