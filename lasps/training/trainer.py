# lasps/training/trainer.py

"""ThreePhaseTrainer for LASPS v7a 3-Phase training strategy.

Phase 1: Backbone training (all parameters, lr=1e-4, 30 epochs)
Phase 2: Sector heads training (backbone frozen, lr=5e-4, 10 epochs/sector)
Phase 3: End-to-end fine-tuning (all parameters, lr=1e-5, 5 epochs)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict
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

    def _run_epoch(self, loader: DataLoader, optimizer=None) -> float:
        """Run a single training or validation epoch.

        Args:
            loader: DataLoader yielding (ts, img, sid, labels) tuples.
            optimizer: If provided, runs in training mode; otherwise validation mode.

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
                    nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        max_norm=1.0,
                    )
                    optimizer.step()
                total_loss += loss.item()
                n_batches += 1
        return total_loss / max(n_batches, 1)

    def train_phase1(self, train_loader: DataLoader, val_loader: DataLoader,
                     epochs: int = 30, lr: float = 1e-4) -> Dict[str, float]:
        """Phase 1: Backbone training with full dataset.

        Trains all model parameters using cosine annealing LR schedule.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of training epochs.
            lr: Initial learning rate.

        Returns:
            Dict with final train_loss and val_loss.
        """
        logger.info("Phase 1: Backbone training")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        for epoch in range(epochs):
            train_loss = self._run_epoch(train_loader, optimizer)
            val_loss = self._run_epoch(val_loader)
            scheduler.step()
            logger.info(f"Phase1 [{epoch+1}/{epochs}] train={train_loss:.4f} val={val_loss:.4f}")
        return {"train_loss": train_loss, "val_loss": val_loss}

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
            optimizer = torch.optim.AdamW(
                self.model.get_sector_head_params(sector_id), lr=lr,
            )
            for epoch in range(epochs_per_sector):
                loss = self._run_epoch(loader, optimizer)
                logger.info(f"Phase2 Sector {sector_id} [{epoch+1}/{epochs_per_sector}] loss={loss:.4f}")
        self.model.unfreeze_backbone()

    def train_phase3(self, train_loader: DataLoader, val_loader: DataLoader,
                     epochs: int = 5, lr: float = 1e-5) -> Dict[str, float]:
        """Phase 3: End-to-end fine-tuning with low learning rate.

        Unfreezes all parameters and fine-tunes with cosine annealing.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of fine-tuning epochs.
            lr: Initial learning rate (should be small).

        Returns:
            Dict with final train_loss and val_loss.
        """
        logger.info("Phase 3: End-to-End fine-tuning")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        for epoch in range(epochs):
            train_loss = self._run_epoch(train_loader, optimizer)
            val_loss = self._run_epoch(val_loader)
            scheduler.step()
            logger.info(f"Phase3 [{epoch+1}/{epochs}] train={train_loss:.4f} val={val_loss:.4f}")
        return {"train_loss": train_loss, "val_loss": val_loss}
