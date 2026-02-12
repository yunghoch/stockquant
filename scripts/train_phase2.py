#!/usr/bin/env python
# scripts/train_phase2.py

"""Phase 2 전용 학습 스크립트

기존 checkpoint를 로드하여 Phase 2 (섹터별 헤드 학습)만 실행합니다.
Backbone은 동결되고, 각 섹터의 헤드만 개별 학습됩니다.

Usage:
    python scripts/train_phase2.py --checkpoint checkpoints/phase1_best.pt
    python scripts/train_phase2.py --checkpoint checkpoints/phase3_final.pt --epochs 15
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import platform
import torch
from torch.utils.data import DataLoader
from loguru import logger

from lasps.models.sector_aware_model import SectorAwareFusionModel
from lasps.data.datasets.stock_dataset import StockDataset
from lasps.training.trainer import ThreePhaseTrainer
from lasps.config.model_config import MODEL_CONFIG, THREE_PHASE_CONFIG, TRAINING_CONFIG
from lasps.utils.logger import setup_logger
from scripts.train import stock_collate_fn


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="LASPS v7a Phase 2 Training")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint to load (phase1_best.pt or phase3_final.pt)")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Directory containing processed data")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Training device (cuda or cpu)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Epochs per sector (default: from config)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: from config)")
    parser.add_argument("--min-samples", type=int, default=None,
                        help="Minimum samples per sector (default: from config)")
    parser.add_argument("--chart-dir", type=str, default=None,
                        help="Directory containing chart NPY files")
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Use all available GPUs with DataParallel")
    return parser.parse_args()


def main() -> None:
    """Run Phase 2 training."""
    args = parse_args()
    setup_logger("INFO")
    logger.info(f"Device: {args.device}")

    # Load config
    cfg = THREE_PHASE_CONFIG["phase2_sector_heads"]
    epochs_per_sector = args.epochs or cfg["epochs_per_sector"]
    lr = args.lr or cfg["lr"]
    min_samples = args.min_samples or cfg.get("min_samples", 100)

    # Multi-GPU setup
    use_multi_gpu = args.multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1
    if use_multi_gpu:
        logger.info(f"Multi-GPU 활성화: {torch.cuda.device_count()}개 GPU 사용")

    # Load model
    model = SectorAwareFusionModel(
        num_sectors=MODEL_CONFIG["num_sectors"],
        ts_input_dim=MODEL_CONFIG["linear_transformer"]["input_dim"],
    )

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(str(ckpt_path), map_location=args.device)
    # Handle checkpoint dict format
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    logger.info(f"Loaded checkpoint: {ckpt_path}")

    # Multi-GPU: DataParallel wrapping
    if use_multi_gpu:
        model = torch.nn.DataParallel(model)
        logger.info("DataParallel 적용됨")

    # Create trainer
    trainer = ThreePhaseTrainer(
        model,
        device=args.device,
        gradient_accumulation_steps=TRAINING_CONFIG.get("gradient_accumulation_steps", 1),
        use_amp=TRAINING_CONFIG.get("use_amp", False),
    )

    # Load data
    data_dir = Path(args.data_dir)
    chart_dir = Path(args.chart_dir) if args.chart_dir else data_dir

    train_ds = StockDataset(
        time_series_path=str(data_dir / "train" / "time_series.npy"),
        chart_images_path=str(chart_dir / "train_chart_images.npy") if args.chart_dir else str(data_dir / "train" / "chart_images.npy"),
        sector_ids_path=str(data_dir / "train" / "sector_ids.npy"),
        labels_path=str(data_dir / "train" / "labels.npy"),
    )

    # Build sector DataLoaders
    batch_size = TRAINING_CONFIG["batch_size"]
    if use_multi_gpu:
        batch_size *= torch.cuda.device_count()

    num_workers = 0 if platform.system() == "Windows" else 4

    sector_loaders = {}
    total_samples = 0

    logger.info("Building sector-specific DataLoaders...")
    for sector_id in range(MODEL_CONFIG["num_sectors"]):
        sector_subset = train_ds.get_sector_subset(sector_id)
        n_samples = len(sector_subset)

        if n_samples < min_samples:
            logger.warning(f"Sector {sector_id:2d}: {n_samples:6d} samples (skipping, min={min_samples})")
            continue

        sector_loaders[sector_id] = DataLoader(
            sector_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=stock_collate_fn,
        )
        total_samples += n_samples
        logger.info(f"Sector {sector_id:2d}: {n_samples:6d} samples")

    logger.info(f"Total: {len(sector_loaders)} sectors, {total_samples} samples")

    if not sector_loaders:
        logger.error("No sectors with enough samples. Exiting.")
        return

    # Run Phase 2
    logger.info(f"Phase 2: {epochs_per_sector} epochs/sector, lr={lr}")
    trainer.train_phase2(
        sector_loaders,
        epochs_per_sector=epochs_per_sector,
        lr=lr,
    )

    # Save checkpoint
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    output_path = output_dir / "phase2_best.pt"
    torch.save(state_dict, output_path)
    logger.info(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
