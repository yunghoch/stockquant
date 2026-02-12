#!/usr/bin/env python
# scripts/train.py

"""
LASPS v7a 3-Phase 학습 스크립트

Usage:
    python scripts/train.py --data-dir data/processed --device cuda
    python scripts/train.py --phase 2 --checkpoint checkpoints/phase1_best.pt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from loguru import logger

from lasps.models.sector_aware_model import SectorAwareFusionModel
from lasps.data.datasets.stock_dataset import StockDataset
from lasps.training.trainer import ThreePhaseTrainer
from lasps.config.model_config import MODEL_CONFIG, THREE_PHASE_CONFIG, TRAINING_CONFIG
from lasps.utils.logger import setup_logger

from typing import Dict, List, Tuple


def stock_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Use all available GPUs with DataParallel")
    parser.add_argument("--chart-dir", type=str, default=None,
                        help="Directory containing chart NPY files (e.g., E:/stockquant_data)")
    parser.add_argument("--resume-epoch", type=int, default=0,
                        help="Resume training from this epoch (loads checkpoint and continues)")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stopping (train for all epochs)")
    return parser.parse_args()


def main() -> None:
    """Run LASPS v7a 3-Phase training."""
    args = parse_args()
    setup_logger("INFO")
    logger.info(f"Device: {args.device}")

    # Multi-GPU 설정
    use_multi_gpu = args.multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1
    if use_multi_gpu:
        logger.info(f"Multi-GPU 활성화: {torch.cuda.device_count()}개 GPU 사용")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    model = SectorAwareFusionModel(
        num_sectors=MODEL_CONFIG["num_sectors"],
        ts_input_dim=MODEL_CONFIG["linear_transformer"]["input_dim"],
    )
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        state_dict = torch.load(str(ckpt_path), map_location=args.device)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded checkpoint: {ckpt_path}")

    # Multi-GPU: DataParallel 래핑
    if use_multi_gpu:
        model = torch.nn.DataParallel(model)
        logger.info("DataParallel 적용됨")

    # Trainer with memory optimization settings
    trainer = ThreePhaseTrainer(
        model,
        device=args.device,
        gradient_accumulation_steps=TRAINING_CONFIG.get("gradient_accumulation_steps", 1),
        use_amp=TRAINING_CONFIG.get("use_amp", False),
    )

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 차트 NPY 경로 설정 (별도 디렉토리 또는 기본 위치)
    chart_dir = Path(args.chart_dir) if args.chart_dir else data_dir

    train_ds = StockDataset(
        time_series_path=str(data_dir / "train" / "time_series.npy"),
        chart_images_path=str(chart_dir / "train_chart_images.npy") if args.chart_dir else str(data_dir / "train" / "chart_images.npy"),
        sector_ids_path=str(data_dir / "train" / "sector_ids.npy"),
        labels_path=str(data_dir / "train" / "labels.npy"),
    )
    val_ds = StockDataset(
        time_series_path=str(data_dir / "val" / "time_series.npy"),
        chart_images_path=str(chart_dir / "val_chart_images.npy") if args.chart_dir else str(data_dir / "val" / "chart_images.npy"),
        sector_ids_path=str(data_dir / "val" / "sector_ids.npy"),
        labels_path=str(data_dir / "val" / "labels.npy"),
    )

    batch_size = TRAINING_CONFIG["batch_size"]
    accum_steps = TRAINING_CONFIG.get("gradient_accumulation_steps", 1)
    effective_batch_size = batch_size * accum_steps

    # Multi-GPU: increase batch size proportionally
    if use_multi_gpu:
        batch_size *= torch.cuda.device_count()
        effective_batch_size = batch_size * accum_steps
        logger.info(f"Multi-GPU 배치 크기: {batch_size} (GPU당 {TRAINING_CONFIG['batch_size']})")

    logger.info(f"배치 크기: {batch_size}, Accumulation: {accum_steps}x, Effective: {effective_batch_size}")

    # Windows에서도 num_workers=4 시도 (PNG 로딩 병목 해결)
    # 문제 발생 시 num_workers=0으로 복구
    import platform
    num_workers = 4
    logger.info(f"DataLoader num_workers={num_workers} (Windows={platform.system() == 'Windows'})")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=stock_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=stock_collate_fn,
    )

    # Helper to get state dict (handle DataParallel)
    def get_state_dict():
        if isinstance(model, torch.nn.DataParallel):
            return model.module.state_dict()
        return model.state_dict()

    if args.phase in (0, 1):
        cfg = THREE_PHASE_CONFIG["phase1_backbone"]

        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume_epoch > 0:
            ckpt_file = output_dir / f"phase1_epoch_{args.resume_epoch:02d}.pt"
            if ckpt_file.exists():
                ckpt = torch.load(str(ckpt_file), map_location=args.device)
                if use_multi_gpu:
                    model.module.load_state_dict(ckpt["model_state_dict"])
                else:
                    model.load_state_dict(ckpt["model_state_dict"])
                start_epoch = args.resume_epoch
                logger.info(f"Resumed from {ckpt_file} (epoch {start_epoch})")
            else:
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

        # Early stopping patience (0 to disable)
        patience = 0 if args.no_early_stop else cfg.get("patience", 5)
        logger.info(f"Early stopping: {'disabled' if patience == 0 else f'patience={patience}'}")

        trainer.train_phase1(
            train_loader, val_loader,
            epochs=cfg["epochs"], lr=cfg["lr"],
            patience=patience,
            checkpoint_dir=output_dir,
            start_epoch=start_epoch,
        )
        torch.save(get_state_dict(), output_dir / "phase1_best.pt")
        logger.info("Phase 1 complete")

    if args.phase in (0, 2):
        cfg = THREE_PHASE_CONFIG["phase2_sector_heads"]
        logger.info("Phase 2: Building sector-specific DataLoaders")

        # 섹터별 DataLoader 생성
        min_samples = cfg.get("min_samples", 100)
        sector_loaders = {}

        for sector_id in range(MODEL_CONFIG["num_sectors"]):
            sector_subset = train_ds.get_sector_subset(sector_id)

            if len(sector_subset) < min_samples:
                logger.warning(f"Sector {sector_id}: {len(sector_subset)} samples (skipping, min={min_samples})")
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

        if sector_loaders:
            trainer.train_phase2(
                sector_loaders,
                epochs_per_sector=cfg["epochs_per_sector"],
                lr=cfg["lr"],
            )
            torch.save(get_state_dict(), output_dir / "phase2_best.pt")
            logger.info("Phase 2 complete")
        else:
            logger.warning("Phase 2 skipped: no sectors with enough samples")

    if args.phase in (0, 3):
        cfg = THREE_PHASE_CONFIG["phase3_finetune"]
        trainer.train_phase3(
            train_loader, val_loader,
            epochs=cfg["epochs"], lr=cfg["lr"],
            patience=cfg.get("patience", 5),
        )
        torch.save(get_state_dict(), output_dir / "phase3_final.pt")
        logger.info("Phase 3 complete")


if __name__ == "__main__":
    main()
