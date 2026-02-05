"""Tests for deep learning model components."""

import torch
import pytest


class TestLinearTransformerEncoder:
    def test_output_shape(self):
        from lasps.models.linear_transformer import LinearTransformerEncoder
        model = LinearTransformerEncoder(
            input_dim=25, hidden_dim=128, num_layers=4,
            num_heads=4, dropout=0.2,
        )
        x = torch.randn(8, 60, 25)
        out = model(x)
        assert out.shape == (8, 128)

    def test_different_batch_sizes(self):
        from lasps.models.linear_transformer import LinearTransformerEncoder
        model = LinearTransformerEncoder(
            input_dim=25, hidden_dim=128, num_layers=4,
            num_heads=4, dropout=0.0,
        )
        for bs in [1, 4, 16]:
            x = torch.randn(bs, 60, 25)
            out = model(x)
            assert out.shape == (bs, 128)
