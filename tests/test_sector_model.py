"""Tests for SectorAwareFusionModel."""

import torch
import pytest
from lasps.models.sector_aware_model import SectorAwareFusionModel


@pytest.fixture
def model():
    return SectorAwareFusionModel(num_sectors=20, ts_input_dim=25)


def test_forward_shape(model):
    bs = 8
    ts = torch.randn(bs, 60, 25)
    img = torch.randn(bs, 3, 224, 224)
    sid = torch.randint(0, 20, (bs,))
    out = model(ts, img, sid)
    assert out["logits"].shape == (bs, 3)
    assert out["probabilities"].shape == (bs, 3)


def test_probabilities_sum_to_one(model):
    bs = 4
    ts = torch.randn(bs, 60, 25)
    img = torch.randn(bs, 3, 224, 224)
    sid = torch.randint(0, 20, (bs,))
    out = model(ts, img, sid)
    sums = out["probabilities"].sum(dim=1)
    assert torch.allclose(sums, torch.ones(bs), atol=1e-5)


def test_freeze_unfreeze(model):
    model.freeze_backbone()
    for p in model.ts_encoder.parameters():
        assert not p.requires_grad
    for p in model.sector_heads.parameters():
        assert p.requires_grad
    model.unfreeze_backbone()
    for p in model.ts_encoder.parameters():
        assert p.requires_grad


def test_sector_head_params(model):
    params = list(model.get_sector_head_params(0))
    assert len(params) == 4
