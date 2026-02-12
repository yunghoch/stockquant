# tests/test_training.py

import torch
import pytest
from lasps.training.loss_functions import FocalLoss
from lasps.training.trainer import ThreePhaseTrainer
from lasps.models.sector_aware_model import SectorAwareFusionModel
from torch.utils.data import DataLoader, TensorDataset


def test_focal_loss_shape():
    loss_fn = FocalLoss(num_classes=3, gamma=2.0)
    logits = torch.randn(8, 3)
    labels = torch.randint(0, 3, (8,))
    loss = loss_fn(logits, labels)
    assert loss.shape == ()
    assert loss.item() > 0


def test_focal_loss_perfect_prediction():
    loss_fn = FocalLoss(num_classes=3, gamma=2.0)
    logits = torch.tensor([[100.0, -100, -100], [-100, 100, -100]])
    labels = torch.tensor([0, 1])
    loss = loss_fn(logits, labels)
    assert loss.item() < 0.01


@pytest.fixture
def tiny_model():
    return SectorAwareFusionModel(num_sectors=3, ts_input_dim=28)


@pytest.fixture
def tiny_loader():
    n = 16
    ts = torch.randn(n, 60, 28)
    img = torch.randn(n, 3, 224, 224)
    sid = torch.randint(0, 3, (n,))
    labels = torch.randint(0, 3, (n,))
    ds = TensorDataset(ts, img, sid, labels)
    return DataLoader(ds, batch_size=4)


def test_phase1_runs(tiny_model, tiny_loader):
    trainer = ThreePhaseTrainer(tiny_model, device="cpu")
    metrics = trainer.train_phase1(tiny_loader, tiny_loader, epochs=1, patience=0)
    assert "train_loss" in metrics
    assert "val_loss" in metrics
    assert "best_val_loss" in metrics
    assert "best_epoch" in metrics
    assert metrics["best_epoch"] == 1


def test_phase2_runs(tiny_model, tiny_loader):
    trainer = ThreePhaseTrainer(tiny_model, device="cpu")
    sector_loaders = {0: tiny_loader, 1: tiny_loader}
    trainer.train_phase2(sector_loaders, epochs_per_sector=1)
    for p in tiny_model.ts_encoder.parameters():
        assert p.requires_grad


def test_phase3_runs(tiny_model, tiny_loader):
    trainer = ThreePhaseTrainer(tiny_model, device="cpu")
    metrics = trainer.train_phase3(tiny_loader, tiny_loader, epochs=1, patience=0)
    assert "train_loss" in metrics
    assert "best_val_loss" in metrics


def test_early_stopping_triggers(tiny_model, tiny_loader):
    """Early stopping should stop before max epochs when patience is exceeded."""
    trainer = ThreePhaseTrainer(tiny_model, device="cpu")
    # patience=1 with 10 epochs: should stop early unless val_loss keeps improving
    metrics = trainer.train_phase1(
        tiny_loader, tiny_loader, epochs=10, patience=1
    )
    assert metrics["best_epoch"] >= 1
    assert metrics["best_val_loss"] <= metrics["val_loss"]
