"""Tests for deep learning model components."""

import torch
import pytest


class TestLinearTransformerEncoder:
    def test_output_shape(self):
        from lasps.models.linear_transformer import LinearTransformerEncoder
        model = LinearTransformerEncoder(
            input_dim=28, hidden_dim=128, num_layers=4,
            num_heads=4, dropout=0.2,
        )
        x = torch.randn(8, 60, 28)
        out = model(x)
        assert out.shape == (8, 128)

    def test_different_batch_sizes(self):
        from lasps.models.linear_transformer import LinearTransformerEncoder
        model = LinearTransformerEncoder(
            input_dim=28, hidden_dim=128, num_layers=4,
            num_heads=4, dropout=0.0,
        )
        for bs in [1, 4, 16]:
            x = torch.randn(bs, 60, 28)
            out = model(x)
            assert out.shape == (bs, 128)


class TestChartCNN:
    def test_output_shape(self):
        from lasps.models.chart_cnn import ChartCNN
        model = ChartCNN(conv_channels=[32, 64, 128, 256], output_dim=128)
        x = torch.randn(4, 3, 224, 224)
        out = model(x)
        assert out.shape == (4, 128)

    def test_single_sample(self):
        from lasps.models.chart_cnn import ChartCNN
        model = ChartCNN(conv_channels=[32, 64, 128, 256], output_dim=128)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 128)
