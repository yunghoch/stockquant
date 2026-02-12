# lasps/services/predictor.py

"""Sector-Aware Prediction Service.

Wraps SectorAwareFusionModel for inference, providing single-sample
and batch prediction with confidence scores and class labels.
"""

import torch
import torch.nn as nn
from typing import Dict
from lasps.utils.constants import CLASS_NAMES


class SectorAwarePredictor:
    """Prediction service for SectorAwareFusionModel.

    Args:
        model: A SectorAwareFusionModel instance.
        device: Device string for inference ('cpu' or 'cuda').
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def predict(self, time_series: torch.Tensor, chart_image: torch.Tensor,
                sector_id: torch.Tensor) -> Dict:
        """Predict for a single sample (batch_size=1).

        Args:
            time_series: (1, 60, 28) time series tensor.
            chart_image: (1, 3, 224, 224) chart image tensor.
            sector_id: (1,) sector ID tensor.

        Returns:
            Dict with prediction, label, probabilities, confidence.
        """
        ts = time_series.to(self.device)
        img = chart_image.to(self.device)
        sid = sector_id.to(self.device)
        # Use phase=2 for inference (sector-specific heads)
        out = self.model(ts, img, sid, phase=2)
        probs = out["probabilities"][0]
        pred = probs.argmax().item()
        return {
            "prediction": pred,
            "label": CLASS_NAMES[pred],
            "probabilities": probs.cpu().tolist(),
            "confidence": probs.max().item(),
        }

    @torch.no_grad()
    def predict_batch(self, time_series: torch.Tensor, chart_image: torch.Tensor,
                      sector_id: torch.Tensor) -> Dict:
        """Predict for a batch of samples.

        Args:
            time_series: (batch, 60, 28) time series tensor.
            chart_image: (batch, 3, 224, 224) chart image tensor.
            sector_id: (batch,) sector ID tensor.

        Returns:
            Dict with predictions, labels, probabilities.
        """
        ts = time_series.to(self.device)
        img = chart_image.to(self.device)
        sid = sector_id.to(self.device)
        # Use phase=2 for inference (sector-specific heads)
        out = self.model(ts, img, sid, phase=2)
        preds = out["probabilities"].argmax(dim=1)
        return {
            "predictions": preds.cpu().tolist(),
            "labels": [CLASS_NAMES[p] for p in preds.cpu().tolist()],
            "probabilities": out["probabilities"].cpu().tolist(),
        }
