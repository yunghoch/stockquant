import numpy as np
import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
    """LASPS v7a 멀티모달 데이터셋

    각 샘플: time_series(60,25) + chart_image(3,224,224) + sector_id + label
    """

    def __init__(self, time_series_path: str, chart_images_path: str,
                 sector_ids_path: str, labels_path: str):
        self.time_series = np.load(time_series_path, mmap_mode="r")
        self.chart_images = np.load(chart_images_path, mmap_mode="r")
        self.sector_ids = np.load(sector_ids_path)
        self.labels = np.load(labels_path)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "time_series": torch.from_numpy(self.time_series[idx].copy()).float(),
            "chart_image": torch.from_numpy(self.chart_images[idx].copy()).float(),
            "sector_id": torch.tensor(self.sector_ids[idx], dtype=torch.int64),
            "label": torch.tensor(self.labels[idx], dtype=torch.int64),
        }
