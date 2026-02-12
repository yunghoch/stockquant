"""Sector-Aware 2-Branch Fusion Model.

Combines LinearTransformerEncoder (time series) and ChartCNN (chart images)
with sector-specific classification heads. Supports 3-phase training
via freeze/unfreeze backbone utilities.

Architecture:
    LinearTransformerEncoder -> 128-dim
    ChartCNN -> 128-dim
    concat(256) -> SharedFusion -> 128-dim
    Phase 1: CommonHead -> 3 (SELL/HOLD/BUY)
    Phase 2/3: SectorHead[sector_id] -> 3 (SELL/HOLD/BUY)
"""

import torch
import torch.nn as nn
from typing import Dict, Iterator, Optional

from lasps.models.linear_transformer import LinearTransformerEncoder
from lasps.models.chart_cnn import ChartCNN
from lasps.config.model_config import MODEL_CONFIG


class SectorAwareFusionModel(nn.Module):
    """Sector-Aware 2-Branch Fusion Model with common head and sector heads.

    Phase 1: Uses common_head for all samples (backbone learns general patterns)
    Phase 2/3: Uses sector_heads for sector-specific classification

    Args:
        num_sectors: Number of sector-specific classification heads.
        ts_input_dim: Number of input features per time step.
        config: Optional model config dict. Defaults to MODEL_CONFIG.
    """

    def __init__(
        self,
        num_sectors: int = 13,
        ts_input_dim: Optional[int] = None,
        config: Optional[Dict] = None,
        use_chart_cnn: bool = True,
    ):
        super().__init__()
        cfg = config or MODEL_CONFIG
        self.num_sectors = num_sectors
        self.use_chart_cnn = use_chart_cnn

        lt_cfg = cfg["linear_transformer"]
        cnn_cfg = cfg["cnn"]
        fusion_cfg = cfg["fusion"]

        # config에서 input_dim 읽기, 명시적 인자가 우선
        actual_input_dim = ts_input_dim if ts_input_dim is not None else lt_cfg["input_dim"]

        # Backbone: Time Series Encoder
        self.ts_encoder = LinearTransformerEncoder(
            input_dim=actual_input_dim,
            hidden_dim=lt_cfg["hidden_dim"],
            num_layers=lt_cfg["num_layers"],
            num_heads=lt_cfg["num_heads"],
            dropout=lt_cfg["dropout"],
        )

        # Backbone: Chart CNN (optional)
        if use_chart_cnn:
            self.cnn = ChartCNN(
                conv_channels=cnn_cfg["conv_channels"],
                output_dim=cnn_cfg["output_dim"],
                dropout=cnn_cfg["dropout"],
            )
            fusion_input_dim = lt_cfg["hidden_dim"] + cnn_cfg["output_dim"]
        else:
            self.cnn = None
            fusion_input_dim = lt_cfg["hidden_dim"]

        # Backbone: Shared Fusion Layer
        shared_dim = fusion_cfg["shared_dim"]
        self.shared_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(fusion_cfg["dropout"]),
        )

        head_hidden = fusion_cfg["sector_head_hidden"]
        num_classes = fusion_cfg["num_classes"]

        # Phase 1: Common Head (모든 샘플에 동일하게 적용)
        self.common_head = nn.Sequential(
            nn.Linear(shared_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(lt_cfg["dropout"]),
            nn.Linear(head_hidden, num_classes),
        )

        # Phase 2/3: Sector-specific Heads
        self.sector_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dim, head_hidden),
                nn.ReLU(),
                nn.Dropout(lt_cfg["dropout"]),
                nn.Linear(head_hidden, num_classes),
            )
            for _ in range(num_sectors)
        ])

    def _encode_backbone(
        self,
        time_series: torch.Tensor,
        chart_image: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode inputs through backbone (encoders + shared fusion).

        Args:
            time_series: (batch, 60, 28) time series input.
            chart_image: (batch, 3, 224, 224) chart image input.

        Returns:
            shared_feat: (batch, 128) shared feature representation.
        """
        ts_feat = self.ts_encoder(time_series)

        if self.use_chart_cnn and chart_image is not None:
            img_feat = self.cnn(chart_image)
            fused = torch.cat([ts_feat, img_feat], dim=1)
        else:
            fused = ts_feat

        shared_feat = self.shared_fusion(fused)
        return shared_feat

    def forward(
        self,
        time_series: torch.Tensor,
        chart_image: Optional[torch.Tensor],
        sector_id: torch.Tensor,
        phase: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with phase-aware head selection.

        Phase 1: Uses common_head for all samples (backbone training)
        Phase 2/3: Routes each sample to its sector-specific head

        Args:
            time_series: (batch, 60, 28) time series input.
            chart_image: (batch, 3, 224, 224) chart image input.
            sector_id: (batch,) integer sector IDs in [0, num_sectors).
            phase: Training phase (1=common head, 2/3=sector heads).

        Returns:
            Dict with keys: logits, probabilities, shared_features.
        """
        shared_feat = self._encode_backbone(time_series, chart_image)

        if phase == 1:
            # Phase 1: 단일 공통 헤드 사용 (Backbone 일반화 학습)
            logits = self.common_head(shared_feat)
        else:
            # Phase 2/3: 섹터별 라우팅
            assert sector_id.min() >= 0 and sector_id.max() < self.num_sectors, \
                f"sector_id out of range [0, {self.num_sectors}): " \
                f"min={sector_id.min().item()}, max={sector_id.max().item()}"

            batch_size = time_series.size(0)
            logits = torch.zeros(batch_size, 3, device=time_series.device)

            # 효율적인 섹터별 배치 처리
            for sid in range(self.num_sectors):
                mask = (sector_id == sid)
                if mask.sum() > 0:
                    logits[mask] = self.sector_heads[sid](shared_feat[mask])

        return {
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=1),
            "shared_features": shared_feat,
        }

    def freeze_backbone(self) -> None:
        """Freeze backbone (encoders + shared fusion) for Phase 2 training."""
        for param in self.ts_encoder.parameters():
            param.requires_grad = False
        if self.cnn is not None:
            for param in self.cnn.parameters():
                param.requires_grad = False
        for param in self.shared_fusion.parameters():
            param.requires_grad = False
        # common_head도 동결 (Phase 2에서 사용 안함)
        for param in self.common_head.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone for Phase 3 fine-tuning."""
        for param in self.ts_encoder.parameters():
            param.requires_grad = True
        if self.cnn is not None:
            for param in self.cnn.parameters():
                param.requires_grad = True
        for param in self.shared_fusion.parameters():
            param.requires_grad = True
        # common_head는 Phase 3에서도 동결 유지 (사용 안함)

    def get_sector_head_params(self, sector_id: int) -> Iterator[nn.Parameter]:
        """Get parameters for a specific sector head.

        Args:
            sector_id: Sector index in [0, num_sectors).

        Returns:
            Iterator over the sector head's parameters.
        """
        return self.sector_heads[sector_id].parameters()

    def get_all_sector_head_params(self) -> Iterator[nn.Parameter]:
        """Get parameters for all sector heads.

        Returns:
            Iterator over all sector heads' parameters.
        """
        for head in self.sector_heads:
            yield from head.parameters()

    def init_sector_heads_from_common(self) -> None:
        """Initialize sector heads with common_head weights.

        Call this at the start of Phase 2 for better initialization.
        """
        common_state = self.common_head.state_dict()
        for head in self.sector_heads:
            head.load_state_dict(common_state)
