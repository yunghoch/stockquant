import torch
import numpy as np
import pytest
from lasps.data.datasets.stock_dataset import StockDataset


@pytest.fixture
def dummy_dataset(tmp_path):
    n_samples = 50
    time_series = np.random.randn(n_samples, 60, 25).astype(np.float32)
    chart_images = np.random.rand(n_samples, 3, 224, 224).astype(np.float32)
    sector_ids = np.random.randint(0, 20, n_samples).astype(np.int64)
    labels = np.random.randint(0, 3, n_samples).astype(np.int64)

    np.save(tmp_path / "time_series.npy", time_series)
    np.save(tmp_path / "chart_images.npy", chart_images)
    np.save(tmp_path / "sector_ids.npy", sector_ids)
    np.save(tmp_path / "labels.npy", labels)

    return StockDataset(
        time_series_path=str(tmp_path / "time_series.npy"),
        chart_images_path=str(tmp_path / "chart_images.npy"),
        sector_ids_path=str(tmp_path / "sector_ids.npy"),
        labels_path=str(tmp_path / "labels.npy"),
    )


def test_dataset_length(dummy_dataset):
    assert len(dummy_dataset) == 50


def test_dataset_item_shapes(dummy_dataset):
    item = dummy_dataset[0]
    assert item["time_series"].shape == (60, 25)
    assert item["chart_image"].shape == (3, 224, 224)
    assert item["sector_id"].shape == ()
    assert item["label"].shape == ()


def test_dataset_dtypes(dummy_dataset):
    item = dummy_dataset[0]
    assert item["time_series"].dtype == torch.float32
    assert item["chart_image"].dtype == torch.float32
    assert item["sector_id"].dtype == torch.int64
    assert item["label"].dtype == torch.int64
