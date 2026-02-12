"""Linear Transformer Encoder for time series data.

Encodes (batch, seq_len=60, input_dim=28) time series into
(batch, hidden_dim=128) feature vectors using a standard
Transformer encoder with learnable positional encoding.
"""

import torch
import torch.nn as nn


class LinearTransformerEncoder(nn.Module):
    """Linear Transformer for time series encoding.

    Input: (batch, seq_len=60, input_dim=28)
    Output: (batch, hidden_dim=128)

    Uses CLS-token-style extraction: returns the first position
    of the encoded sequence as the summary representation.

    Args:
        input_dim: Number of input features per time step.
        hidden_dim: Transformer hidden dimension and output dimension.
        num_layers: Number of Transformer encoder layers.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        max_len: Maximum sequence length for positional encoding.
    """

    def __init__(
        self,
        input_dim: int = 28,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.2,
        max_len: int = 120,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_len, hidden_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            Feature vector of shape (batch, hidden_dim).
        """
        seq_len = x.size(1)
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        # TransformerEncoder expects (seq_len, batch, hidden_dim)
        x = x.transpose(0, 1)
        encoded = self.transformer(x)
        # Back to (batch, seq_len, hidden_dim) and take first position
        encoded = encoded.transpose(0, 1)
        return encoded[:, 0, :]
