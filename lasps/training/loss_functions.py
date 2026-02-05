# lasps/training/loss_functions.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification.

    Addresses class imbalance in stock prediction (SELL/HOLD/BUY)
    by down-weighting well-classified examples and focusing on
    hard, misclassified ones.

    Args:
        num_classes: Number of target classes.
        gamma: Focusing parameter. Higher gamma increases focus on
            hard examples. gamma=0 is equivalent to cross-entropy.
        weight: Per-class weight tensor for additional class balancing.
    """

    def __init__(
        self,
        num_classes: int = 3,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.gamma = gamma
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight: Optional[torch.Tensor] = None

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: (batch, num_classes) raw model output.
            labels: (batch,) integer class labels.

        Returns:
            Scalar focal loss value.
        """
        ce_loss = F.cross_entropy(logits, labels, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
