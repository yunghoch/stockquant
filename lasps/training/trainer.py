# lasps/training/trainer.py

"""ThreePhaseTrainer for LASPS v7a 3-Phase training strategy.

Phase 1: Backbone training (all parameters, lr=1e-4, 30 epochs)
Phase 2: Sector heads training (backbone frozen, lr=5e-4, 10 epochs/sector)
Phase 3: End-to-end fine-tuning (all parameters, lr=1e-5, 5 epochs)
"""

import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from loguru import logger
from lasps.training.loss_functions import FocalLoss


class ThreePhaseTrainer:
    """Three-phase trainer for SectorAwareFusionModel.

    Implements the 3-phase training strategy:
    1. Backbone pre-training with full dataset
    2. Sector-specific head training with backbone frozen
    3. End-to-end fine-tuning with low learning rate

    Args:
        model: SectorAwareFusionModel instance.
        device: Device string ('cpu' or 'cuda').
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.criterion = FocalLoss(num_classes=3, gamma=2.0)

    def _run_epoch(
        self,
        loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        clip_params: Optional[List[nn.Parameter]] = None,
    ) -> float:
        """Run a single training or validation epoch.

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

        with torch.set_grad_enabled(is_train):
            for batch in loader:
                ts, img, sid, labels = [b.to(self.device) for b in batch]
                out = self.model.forward_efficient(ts, img, sid)
                loss = self.criterion(out["logits"], labels)
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    if clip_params is not None:
                        nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
                    else:
                        trainable = [p for p in self.model.parameters()
                                     if p.requires_grad]
                        nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                    has_grad = any(
                        p.grad is not None
                        for g in optimizer.param_groups for p in g["params"]
                    )
                    if has_grad:
                        optimizer.step()
                total_loss += loss.item()
                n_batches += 1
        return total_loss / max(n_batches, 1)

    def train_phase1(self, train_loader: DataLoader, val_loader: DataLoader,
                     epochs: int = 30, lr: float = 1e-4,
                     patience: int = 5) -> Dict:
        """Phase 1: Backbone training with full dataset.

        Trains all model parameters using cosine annealing LR schedule.
        Tracks best validation loss and restores best model weights.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of training epochs.
            lr: Initial learning rate.
            patience: Early stopping patience (0 to disable).

        Returns:
            Dict with train_loss, val_loss, best_val_loss, best_epoch.
        """
        logger.info("Phase 1: Backbone training")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float("inf")
        best_epoch = 0
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            train_loss = self._run_epoch(train_loader, optimizer)
            val_loss = self._run_epoch(val_loader)
            scheduler.step()
            logger.info(f"Phase1 [{epoch+1}/{epochs}] train={train_loss:.4f} val={val_loss:.4f}")

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
                     epochs_per_sector: int = 10, lr: float = 5e-4) -> None:
        """Phase 2: Sector-specific head training with backbone frozen.

        Freezes backbone (encoders + shared fusion) and trains each sector
        head independently with its own optimizer.

        Args:
            sector_loaders: Dict mapping sector_id to its DataLoader.
            epochs_per_sector: Number of epochs per sector head.
            lr: Learning rate for sector head training.
        """
        logger.info("Phase 2: Sector Heads (backbone frozen)")
        self.model.freeze_backbone()
        for sector_id, loader in sector_loaders.items():
            params = list(self.model.get_sector_head_params(sector_id))
            optimizer = torch.optim.AdamW(params, lr=lr)
            for epoch in range(epochs_per_sector):
                loss = self._run_epoch(loader, optimizer, clip_params=params)
                logger.info(f"Phase2 Sector {sector_id} [{epoch+1}/{epochs_per_sector}] loss={loss:.4f}")
        self.model.unfreeze_backbone()

    def train_phase3(self, train_loader: DataLoader, val_loader: DataLoader,
                     epochs: int = 5, lr: float = 1e-5,
                     patience: int = 5) -> Dict:
        """Phase 3: End-to-end fine-tuning with low learning rate.

        Unfreezes all parameters and fine-tunes with cosine annealing.
        Tracks best validation loss and restores best model weights.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of fine-tuning epochs.
            lr: Initial learning rate (should be small).
            patience: Early stopping patience (0 to disable).

        Returns:
            Dict with train_loss, val_loss, best_val_loss, best_epoch.
        """
        logger.info("Phase 3: End-to-End fine-tuning")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float("inf")
        best_epoch = 0
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            train_loss = self._run_epoch(train_loader, optimizer)
            val_loss = self._run_epoch(val_loader)
            scheduler.step()
            logger.info(f"Phase3 [{epoch+1}/{epochs}] train={train_loss:.4f} val={val_loss:.4f}")

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
