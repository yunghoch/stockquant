"""Chart CNN encoder for candlestick chart images.

Encodes (batch, 3, 224, 224) chart images into (batch, output_dim=128)
feature vectors using a custom CNN with configurable convolutional layers.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class ChartCNN(nn.Module):
    """Chart image encoder: (batch, 3, 224, 224) -> (batch, 128).

    Uses a sequence of Conv2d -> BatchNorm -> ReLU -> MaxPool blocks
    followed by adaptive average pooling and a fully connected layer.

    Args:
        input_channels: Number of input image channels (3 for RGB).
        conv_channels: List of output channels for each conv block.
        output_dim: Dimension of the output feature vector.
        dropout: Dropout rate before the final FC layer.
    """

    def __init__(
        self,
        input_channels: int = 3,
        conv_channels: Optional[List[int]] = None,
        output_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64, 128, 256]

        layers: List[nn.Module] = []
        in_ch = input_channels
        for out_ch in conv_channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(conv_channels[-1], output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, height, width).

        Returns:
            Feature vector of shape (batch, output_dim).
        """
        x = self.features(x)
        x = self.pool(x)
        return self.fc(x)
