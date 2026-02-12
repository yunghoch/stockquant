# lasps/training/trainer.py

"""ThreePhaseTrainer for LASPS v7a 3-Phase training strategy.

Phase 1: Backbone training with common_head (all parameters, lr=1e-4, 30 epochs)
Phase 2: Sector heads training (backbone frozen, lr=5e-4, 10 epochs/sector)
Phase 3: End-to-end fine-tuning with sector_heads (all parameters, lr=1e-5, 5 epochs)

Memory Optimization:
- Gradient Accumulation: 작은 배치로 나눠 처리 후 누적 (effective batch size 유지)
- Mixed Precision (AMP): FP16 연산으로 메모리 40-50% 절감
- 주기적 캐시 정리: torch.cuda.empty_cache()
"""

import copy
import gc
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, List, Optional, Union
from loguru import logger
from lasps.config.settings import settings
from lasps.config.model_config import TRAINING_CONFIG
from lasps.training.loss_functions import FocalLoss


class ThreePhaseTrainer:
    """Three-phase trainer for SectorAwareFusionModel.

    Implements the 3-phase training strategy:
    1. Backbone pre-training with common_head (general pattern learning)
    2. Sector-specific head training with backbone frozen
    3. End-to-end fine-tuning with sector_heads

    Args:
        model: SectorAwareFusionModel instance (can be wrapped in DataParallel).
        device: Device string ('cpu' or 'cuda').
        use_class_weights: Whether to apply class weights for imbalanced data.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        use_class_weights: bool = True,
        gradient_accumulation_steps: int = None,
        use_amp: bool = None,
    ):
        self.model = model.to(device)
        self.device = device
        # Handle DataParallel: access underlying module for method calls
        self._base_model = model.module if isinstance(model, nn.DataParallel) else model

        # Current training phase (for forward pass)
        self._current_phase = 1

        # Memory optimization settings (from config or explicit)
        self.gradient_accumulation_steps = (
            gradient_accumulation_steps
            if gradient_accumulation_steps is not None
            else TRAINING_CONFIG.get("gradient_accumulation_steps", 1)
        )
        self.use_amp = (
            use_amp if use_amp is not None else TRAINING_CONFIG.get("use_amp", False)
        )

        # Mixed Precision scaler (only for CUDA)
        self.scaler = GradScaler() if self.use_amp and device != "cpu" else None

        if self.gradient_accumulation_steps > 1:
            logger.info(f"Gradient Accumulation: {self.gradient_accumulation_steps} steps")
        if self.use_amp and self.scaler:
            logger.info("Mixed Precision (AMP) 활성화: FP16 연산으로 메모리 절감")

        # Apply class weights for imbalanced classification (SELL/HOLD/BUY)
        if use_class_weights:
            class_weights = torch.tensor(settings.CLASS_WEIGHTS, dtype=torch.float32).to(device)
            logger.info(f"Using class weights: SELL={settings.CLASS_WEIGHTS[0]:.2f}, "
                       f"HOLD={settings.CLASS_WEIGHTS[1]:.2f}, BUY={settings.CLASS_WEIGHTS[2]:.2f}")
        else:
            class_weights = None

        self.criterion = FocalLoss(num_classes=3, gamma=2.0, weight=class_weights)

    def _run_epoch(
        self,
        loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        clip_params: Optional[List[nn.Parameter]] = None,
    ) -> float:
        """Run a single training or validation epoch.

        Supports Gradient Accumulation and Mixed Precision (AMP) for memory optimization.

        Args:
            loader: DataLoader yielding (ts, img, sid, labels) tuples.
            optimizer: If provided, runs in training mode; otherwise validation mode.
            clip_params: Parameters to clip gradients for. If None, clips all
                trainable model params. Use optimizer param_groups for Phase 2.

        Returns:
            Average loss over all batches.
        """
        is_train = optimizer is not None
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        total_loss, n_batches = 0.0, 0
        accum_steps = self.gradient_accumulation_steps if is_train else 1

        with torch.set_grad_enabled(is_train):
            for batch_idx, batch in enumerate(loader):
                ts, img, sid, labels = [b.to(self.device) for b in batch]

                # Mixed Precision forward pass with phase
                if self.scaler is not None:
                    with autocast():
                        out = self.model(ts, img, sid, phase=self._current_phase)
                        loss = self.criterion(out["logits"], labels)
                        # Scale loss for gradient accumulation
                        loss = loss / accum_steps
                else:
                    out = self.model(ts, img, sid, phase=self._current_phase)
                    loss = self.criterion(out["logits"], labels)
                    loss = loss / accum_steps

                if is_train:
                    # Backward pass with AMP scaling
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # Optimizer step at accumulation boundary
                    if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
                        if clip_params is not None:
                            if self.scaler is not None:
                                self.scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
                        else:
                            trainable = [p for p in self.model.parameters()
                                         if p.requires_grad]
                            if self.scaler is not None:
                                self.scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(trainable, max_norm=1.0)

                        has_grad = any(
                            p.grad is not None
                            for g in optimizer.param_groups for p in g["params"]
                        )
                        if has_grad:
                            if self.scaler is not None:
                                self.scaler.step(optimizer)
                                self.scaler.update()
                            else:
                                optimizer.step()
                        optimizer.zero_grad()

                # Restore unscaled loss for logging
                total_loss += loss.item() * accum_steps
                n_batches += 1

                # 진행률 로그 (100 batch마다)
                if is_train and batch_idx % 100 == 0:
                    logger.info(f"  Batch [{batch_idx}/{len(loader)}] loss={loss.item() * accum_steps:.4f}")

        # Memory cleanup after epoch
        if self.device != "cpu":
            torch.cuda.empty_cache()
            gc.collect()

        return total_loss / max(n_batches, 1)

    def _get_state_dict(self) -> Dict:
        """Get model state dict (handles DataParallel)."""
        if isinstance(self.model, nn.DataParallel):
            return self.model.module.state_dict()
        return self.model.state_dict()

    def _save_checkpoint(self, checkpoint_dir: Path, epoch: int, phase: str,
                         train_loss: float, val_loss: float) -> None:
        """Save checkpoint for current epoch.

        Args:
            checkpoint_dir: Directory to save checkpoints.
            epoch: Current epoch number (1-indexed).
            phase: Training phase name (e.g., "phase1").
            train_loss: Training loss for this epoch.
            val_loss: Validation loss for this epoch.
        """
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{phase}_epoch_{epoch:02d}.pt"

        checkpoint = {
            "epoch": epoch,
            "phase": phase,
            "model_state_dict": self._get_state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def train_phase1(self, train_loader: DataLoader, val_loader: DataLoader,
                     epochs: int = 30, lr: float = 1e-4,
                     patience: int = 5,
                     checkpoint_dir: Optional[Union[str, Path]] = None,
                     start_epoch: int = 0) -> Dict:
        """Phase 1: Backbone training with common_head.

        Trains backbone + common_head using cosine annealing LR schedule.
        All samples use the same common_head (no sector routing).
        This allows backbone to learn general stock patterns without sector bias.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of training epochs.
            lr: Initial learning rate.
            patience: Early stopping patience (0 to disable).
            checkpoint_dir: Directory to save epoch checkpoints (optional).
            start_epoch: Epoch to resume from (0 for fresh start).

        Returns:
            Dict with train_loss, val_loss, best_val_loss, best_epoch.
        """
        logger.info("Phase 1: Backbone + CommonHead training (no sector routing)")
        self._current_phase = 1

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Advance scheduler to start_epoch position
        if start_epoch > 0:
            for _ in range(start_epoch):
                scheduler.step()
            logger.info(f"Scheduler advanced to epoch {start_epoch}, lr={scheduler.get_last_lr()[0]:.6f}")

        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)

        best_val_loss = float("inf")
        best_epoch = 0
        best_state = None
        no_improve = 0

        for epoch in range(start_epoch, epochs):
            train_loss = self._run_epoch(train_loader, optimizer)
            val_loss = self._run_epoch(val_loader)
            scheduler.step()
            logger.info(f"Phase1 [{epoch+1}/{epochs}] train={train_loss:.4f} val={val_loss:.4f}")

            # Save checkpoint every epoch
            if checkpoint_dir is not None:
                self._save_checkpoint(checkpoint_dir, epoch + 1, "phase1", train_loss, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_state = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if patience > 0 and no_improve >= patience:
                logger.info(f"Phase1 early stopping at epoch {epoch+1} (patience={patience})")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info(f"Phase1 restored best model from epoch {best_epoch}")

        return {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
        }

    def train_phase2(self, sector_loaders: Dict[int, DataLoader],
                     epochs_per_sector: int = 10, lr: float = 5e-4,
                     init_from_common: bool = True) -> None:
        """Phase 2: Sector-specific head training with backbone frozen.

        Freezes backbone (encoders + shared fusion + common_head) and trains
        each sector head independently with its own optimizer.

        Args:
            sector_loaders: Dict mapping sector_id to its DataLoader.
            epochs_per_sector: Number of epochs per sector head.
            lr: Learning rate for sector head training.
            init_from_common: If True, initialize sector heads with common_head weights.
        """
        logger.info("Phase 2: Sector Heads (backbone frozen)")
        self._current_phase = 2

        # Initialize sector heads from common_head for better starting point
        if init_from_common:
            self._base_model.init_sector_heads_from_common()
            logger.info("Sector heads initialized from common_head weights")

        # Freeze backbone (including common_head)
        self._base_model.freeze_backbone()

        for sector_id, loader in sector_loaders.items():
            params = list(self._base_model.get_sector_head_params(sector_id))
            optimizer = torch.optim.AdamW(params, lr=lr)
            for epoch in range(epochs_per_sector):
                loss = self._run_epoch(loader, optimizer, clip_params=params)
                logger.info(f"Phase2 Sector {sector_id} [{epoch+1}/{epochs_per_sector}] loss={loss:.4f}")

        self._base_model.unfreeze_backbone()

    def train_phase3(self, train_loader: DataLoader, val_loader: DataLoader,
                     epochs: int = 5, lr: float = 1e-5,
                     patience: int = 5,
                     checkpoint_dir: Optional[Union[str, Path]] = None) -> Dict:
        """Phase 3: End-to-end fine-tuning with sector_heads.

        Unfreezes backbone and fine-tunes all parameters with cosine annealing.
        Uses sector_heads for routing (not common_head).
        Tracks best validation loss and restores best model weights.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of fine-tuning epochs.
            lr: Initial learning rate (should be small).
            patience: Early stopping patience (0 to disable).
            checkpoint_dir: Directory to save epoch checkpoints (optional).

        Returns:
            Dict with train_loss, val_loss, best_val_loss, best_epoch.
        """
        logger.info("Phase 3: End-to-End fine-tuning (with sector routing)")
        self._current_phase = 3

        # Unfreeze backbone for fine-tuning
        self._base_model.unfreeze_backbone()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)

        best_val_loss = float("inf")
        best_epoch = 0
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            train_loss = self._run_epoch(train_loader, optimizer)
            val_loss = self._run_epoch(val_loader)
            scheduler.step()
            logger.info(f"Phase3 [{epoch+1}/{epochs}] train={train_loss:.4f} val={val_loss:.4f}")

            # Save checkpoint every epoch
            if checkpoint_dir is not None:
                self._save_checkpoint(checkpoint_dir, epoch + 1, "phase3", train_loss, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_state = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if patience > 0 and no_improve >= patience:
                logger.info(f"Phase3 early stopping at epoch {epoch+1} (patience={patience})")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info(f"Phase3 restored best model from epoch {best_epoch}")

        return {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
        }
