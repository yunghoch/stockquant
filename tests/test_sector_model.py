"""Tests for SectorAwareFusionModel."""

import torch
import pytest
from lasps.models.sector_aware_model import SectorAwareFusionModel


@pytest.fixture
def model():
    return SectorAwareFusionModel(num_sectors=13, ts_input_dim=28)


def test_forward_phase1(model):
    """Phase 1: common_head 사용."""
    bs = 8
    ts = torch.randn(bs, 60, 28)
    img = torch.randn(bs, 3, 224, 224)
    sid = torch.randint(0, 13, (bs,))
    out = model(ts, img, sid, phase=1)
    assert out["logits"].shape == (bs, 3)
    assert out["probabilities"].shape == (bs, 3)


def test_forward_phase2(model):
    """Phase 2/3: sector_heads 라우팅."""
    bs = 8
    ts = torch.randn(bs, 60, 28)
    img = torch.randn(bs, 3, 224, 224)
    sid = torch.randint(0, 13, (bs,))
    out = model(ts, img, sid, phase=2)
    assert out["logits"].shape == (bs, 3)
    assert out["probabilities"].shape == (bs, 3)


def test_probabilities_sum_to_one(model):
    bs = 4
    ts = torch.randn(bs, 60, 28)
    img = torch.randn(bs, 3, 224, 224)
    sid = torch.randint(0, 13, (bs,))
    out = model(ts, img, sid, phase=1)
    sums = out["probabilities"].sum(dim=1)
    assert torch.allclose(sums, torch.ones(bs), atol=1e-5)


def test_freeze_unfreeze(model):
    model.freeze_backbone()
    for p in model.ts_encoder.parameters():
        assert not p.requires_grad
    for p in model.common_head.parameters():
        assert not p.requires_grad  # common_head도 동결
    for p in model.sector_heads.parameters():
        assert p.requires_grad  # sector_heads는 학습 가능
    model.unfreeze_backbone()
    for p in model.ts_encoder.parameters():
        assert p.requires_grad


def test_sector_head_params(model):
    params = list(model.get_sector_head_params(0))
    assert len(params) == 4  # Linear(128,64) weight/bias + Linear(64,3) weight/bias


def test_all_sector_head_params(model):
    params = list(model.get_all_sector_head_params())
    # 13 sectors × 4 params each
    assert len(params) == 13 * 4


def test_init_sector_heads_from_common(model):
    """common_head 가중치를 sector_heads에 복사."""
    model.init_sector_heads_from_common()
    common_state = model.common_head.state_dict()
    for head in model.sector_heads:
        head_state = head.state_dict()
        for key in common_state:
            assert torch.allclose(common_state[key], head_state[key])


def test_invalid_sector_id_phase2_raises(model):
    """Phase 2에서 잘못된 sector_id는 에러."""
    bs = 2
    ts = torch.randn(bs, 60, 28)
    img = torch.randn(bs, 3, 224, 224)
    sid = torch.tensor([-1, 0])
    with pytest.raises(AssertionError, match="sector_id out of range"):
        model(ts, img, sid, phase=2)


def test_invalid_sector_id_max_raises(model):
    """sector_id가 num_sectors 이상이면 에러."""
    bs = 2
    ts = torch.randn(bs, 60, 28)
    img = torch.randn(bs, 3, 224, 224)
    sid = torch.tensor([13, 0])  # 13은 범위 초과
    with pytest.raises(AssertionError, match="sector_id out of range"):
        model(ts, img, sid, phase=2)
