import os
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from typing import Self


def compute_temporal_features(date_str: str) -> np.ndarray:
    """날짜 문자열에서 temporal features (weekday, month, day) 계산.

    v2: 선형 정규화 방식 (sin/cos 대신)

    Args:
        date_str: ISO format 날짜 문자열 (예: "2023-01-15")

    Returns:
        (3,) shape의 temporal features 배열 [weekday, month, day]
    """
    dt = datetime.fromisoformat(date_str)
    weekday = dt.weekday() / 4.0  # 월=0, 금=1 (0~1)
    month = dt.month / 12.0  # 1월=0.08, 12월=1.0
    day = dt.day / 31.0  # 1일=0.03, 31일=1.0

    return np.array([weekday, month, day], dtype=np.float32)


class StockDataset(Dataset):
    """LASPS v7a 멀티모달 데이터셋

    각 샘플: time_series(60,28) + chart_image(3,224,224) + sector_id + label

    두 가지 데이터 형식 지원:
    - v1: time_series (60, 25) + 동적 temporal 계산 → (60, 28)
    - v2: time_series (60, 28) 이미 temporal 포함 (processed_v2/)

    temporal 특성 (v2):
    - weekday: 0~1 (월=0, 금=0.8)
    - month: 0~1 (1월=0.08, 12월=1.0)
    - day: 0~1 (1일=0.03, 31일=1.0)

    차트 이미지는 두 가지 방식 지원:
    1. chart_images.npy 파일 (단일 배열)
    2. charts/ 디렉토리 (개별 PNG 파일)
    """

    def __init__(self, time_series_path: str, chart_images_path: str,
                 sector_ids_path: str, labels_path: str,
                 metadata_path: Optional[str] = None):
        self.time_series = np.load(time_series_path, mmap_mode="r")
        self.sector_ids = np.load(sector_ids_path)
        self.labels = np.load(labels_path)

        # Metadata 로드 (temporal features 계산용)
        if metadata_path is None:
            # 기본 경로: time_series.npy와 같은 디렉토리의 metadata.csv
            ts_dir = Path(time_series_path).parent
            metadata_path = str(ts_dir / "metadata.csv")

        if Path(metadata_path).exists():
            self.metadata = pd.read_csv(metadata_path)
            self.dates = self.metadata["date"].tolist()
        else:
            # metadata가 없으면 temporal features를 0으로 설정
            self.metadata = None
            self.dates = None

        # 차트 이미지: .npy 파일 또는 디렉토리 지원
        chart_path = Path(chart_images_path)
        num_samples = len(self.labels)

        if chart_path.suffix == ".npy" and chart_path.exists():
            # memmap 파일 로드 (raw binary 또는 numpy 형식 모두 지원)
            try:
                self.chart_images = np.load(chart_images_path, mmap_mode="r")
            except ValueError:
                # raw memmap 파일인 경우 직접 로드
                shape = (num_samples, 3, 224, 224)
                self.chart_images = np.memmap(
                    chart_images_path, dtype=np.float32, mode='r', shape=shape
                )
            self.charts_dir = None
        else:
            # charts/ 디렉토리 (개별 PNG 파일)
            charts_dir = chart_path.parent / "charts"
            if charts_dir.is_dir():
                self.charts_dir = charts_dir
                self.chart_images = None
            else:
                raise FileNotFoundError(
                    f"차트 데이터를 찾을 수 없음: {chart_images_path} 또는 {charts_dir}"
                )

    def __len__(self) -> int:
        return len(self.labels)

    def _load_chart_image(self, idx: int) -> np.ndarray:
        """개별 PNG 파일에서 차트 이미지 로드."""
        png_path = self.charts_dir / f"{idx:06d}.png"
        if not png_path.exists():
            # 빈 이미지 반환 (fallback)
            return np.zeros((3, 224, 224), dtype=np.float32)
        img = Image.open(png_path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr.transpose(2, 0, 1)  # HWC -> CHW

    def __getitem__(self, idx: int) -> dict:
        # 차트 이미지 로드
        if self.chart_images is not None:
            chart_img = self.chart_images[idx].copy()
        else:
            chart_img = self._load_chart_image(idx)

        # 시계열 데이터
        ts_base = self.time_series[idx].copy()

        # v2 데이터 (60, 28): temporal 이미 포함
        if ts_base.shape[1] >= 28:
            ts_full = ts_base
        # v1 데이터 (60, 25): temporal 추가 필요
        else:
            if self.dates is not None:
                temporal = compute_temporal_features(self.dates[idx])
                # 60일 모두 같은 temporal features (마지막 날짜 기준)
                temporal_expanded = np.tile(temporal, (ts_base.shape[0], 1))  # (60, 3)
            else:
                # metadata가 없으면 0으로 채움
                temporal_expanded = np.zeros((ts_base.shape[0], 3), dtype=np.float32)

            # (60, 25) + (60, 3) = (60, 28)
            ts_full = np.concatenate([ts_base, temporal_expanded], axis=1)

        return {
            "time_series": torch.from_numpy(ts_full).float(),
            "chart_image": torch.from_numpy(chart_img).float(),
            "sector_id": torch.tensor(self.sector_ids[idx], dtype=torch.int64),
            "label": torch.tensor(self.labels[idx], dtype=torch.int64),
        }

    def get_sector_indices(self, sector_id: int) -> np.ndarray:
        """특정 섹터의 샘플 인덱스 반환.

        Args:
            sector_id: 섹터 ID (0-12)

        Returns:
            해당 섹터의 샘플 인덱스 배열
        """
        return np.where(self.sector_ids == sector_id)[0]

    def get_sector_subset(self, sector_id: int) -> "SectorSubset":
        """특정 섹터의 Subset 반환.

        Args:
            sector_id: 섹터 ID (0-12)

        Returns:
            해당 섹터만 포함하는 SectorSubset 객체
        """
        indices = self.get_sector_indices(sector_id)
        return SectorSubset(self, indices, sector_id)


class SectorSubset(Dataset):
    """특정 섹터만 포함하는 Dataset Subset.

    Phase 2 섹터별 헤드 학습에 사용됩니다.

    Args:
        dataset: 원본 StockDataset
        indices: 해당 섹터의 샘플 인덱스 배열
        sector_id: 섹터 ID
    """

    def __init__(self, dataset: StockDataset, indices: np.ndarray, sector_id: int):
        self.dataset = dataset
        self.indices = indices
        self.sector_id = sector_id

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        return self.dataset[self.indices[idx]]
