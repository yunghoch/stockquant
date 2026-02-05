"""Sector-Aware 2-Branch Fusion Model.

Combines LinearTransformerEncoder (time series) and ChartCNN (chart images)
with 20 sector-specific classification heads. Supports 3-phase training
via freeze/unfreeze backbone utilities.

Architecture:
    LinearTransformerEncoder -> 128-dim
    ChartCNN -> 128-dim
    concat(256) -> SharedFusion -> 128-dim
    SectorHead[sector_id]: 128 -> 64 -> 3 (SELL/HOLD/BUY)
"""

import torch
import torch.nn as nn
from typing import Dict, Iterator

from lasps.models.linear_transformer import LinearTransformerEncoder
from lasps.models.chart_cnn import ChartCNN


class SectorAwareFusionModel(nn.Module):
    """Sector-Aware 2-Branch Fusion Model with 20 sector heads.

    Args:
        num_sectors: Number of sector-specific classification heads.
        ts_input_dim: Number of input features per time step.
    """

    def __init__(self, num_sectors: int = 20, ts_input_dim: int = 25):
        super().__init__()
        self.num_sectors = num_sectors

        self.ts_encoder = LinearTransformerEncoder(
            input_dim=ts_input_dim, hidden_dim=128,
            num_layers=4, num_heads=4, dropout=0.2,
        )
        self.cnn = ChartCNN(
            conv_channels=[32, 64, 128, 256], output_dim=128,
        )
        self.shared_fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.sector_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 3),
            )
            for _ in range(num_sectors)
        ])

    def forward(
        self,
        time_series: torch.Tensor,
        chart_image: torch.Tensor,
        sector_id: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass routing each sample to its sector head.

        Args:
            time_series: (batch, 60, 25) time series input.
            chart_image: (batch, 3, 224, 224) chart image input.
            sector_id: (batch,) integer sector IDs in [0, num_sectors).

        Returns:
            Dict with keys: logits, probabilities, shared_features.
        """
        ts_feat = self.ts_encoder(time_series)
        img_feat = self.cnn(chart_image)
        fused = torch.cat([ts_feat, img_feat], dim=1)
        shared_feat = self.shared_fusion(fused)

        batch_size = time_series.size(0)
        logits = torch.zeros(batch_size, 3, device=time_series.device)
        for i in range(batch_size):
            sid = sector_id[i].item()
            logits[i] = self.sector_heads[sid](
                shared_feat[i:i + 1]
            ).squeeze(0)

        return {
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=1),
            "shared_features": shared_feat,
        }

    def forward_efficient(
        self,
        time_series: torch.Tensor,
        chart_image: torch.Tensor,
        sector_id: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Efficient forward pass grouping samples by sector.

        Args:
            time_series: (batch, 60, 25) time series input.
            chart_image: (batch, 3, 224, 224) chart image input.
            sector_id: (batch,) integer sector IDs in [0, num_sectors).

        Returns:
            Dict with keys: logits, probabilities.
        """
        ts_feat = self.ts_encoder(time_series)
        img_feat = self.cnn(chart_image)
        fused = torch.cat([ts_feat, img_feat], dim=1)
        shared_feat = self.shared_fusion(fused)

        batch_size = time_series.size(0)
        logits = torch.zeros(batch_size, 3, device=time_series.device)
        for sid in range(self.num_sectors):
            mask = (sector_id == sid)
            if mask.sum() > 0:
                logits[mask] = self.sector_heads[sid](shared_feat[mask])

        return {
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=1),
        }

    def freeze_backbone(self) -> None:
        """Freeze backbone (encoders + shared fusion) for Phase 2 training."""
        for param in self.ts_encoder.parameters():
            param.requires_grad = False
        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.shared_fusion.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone for Phase 3 fine-tuning."""
        for param in self.ts_encoder.parameters():
            param.requires_grad = True
        for param in self.cnn.parameters():
            param.requires_grad = True
        for param in self.shared_fusion.parameters():
            param.requires_grad = True

    def get_sector_head_params(self, sector_id: int) -> Iterator[nn.Parameter]:
        """Get parameters for a specific sector head.

        Args:
            sector_id: Sector index in [0, num_sectors).

        Returns:
            Iterator over the sector head's parameters.
        """
        return self.sector_heads[sector_id].parameters()
