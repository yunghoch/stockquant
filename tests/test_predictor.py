# tests/test_predictor.py

import torch
import pytest
from lasps.services.predictor import SectorAwarePredictor
from lasps.models.sector_aware_model import SectorAwareFusionModel


@pytest.fixture
def predictor():
    model = SectorAwareFusionModel(num_sectors=20, ts_input_dim=25)
    return SectorAwarePredictor(model, device="cpu")


def test_predict_single(predictor):
    ts = torch.randn(1, 60, 25)
    img = torch.randn(1, 3, 224, 224)
    sid = torch.tensor([0])
    result = predictor.predict(ts, img, sid)
    assert result["prediction"] in [0, 1, 2]
    assert len(result["probabilities"]) == 3


def test_predict_batch(predictor):
    bs = 8
    ts = torch.randn(bs, 60, 25)
    img = torch.randn(bs, 3, 224, 224)
    sid = torch.randint(0, 20, (bs,))
    result = predictor.predict_batch(ts, img, sid)
    assert len(result["predictions"]) == bs
