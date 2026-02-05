import matplotlib
matplotlib.use("Agg")

import io
import numpy as np
import pandas as pd
import torch
import mplfinance as mpf
from PIL import Image
from lasps.utils.constants import CHART_IMAGE_SIZE


class ChartGenerator:
    """캔들차트 이미지 생성기 (224x224)

    mplfinance를 사용하여 OHLCV 데이터를 캔들차트 이미지로 변환한다.
    모델 입력용 (3, 224, 224) 텐서를 생성한다.
    """

    def __init__(self, style: str = "yahoo"):
        self.style = style
        self.dpi = 100
        self.figsize = (2.56, 2.56)

    def _render_to_pil(self, ohlcv_df: pd.DataFrame) -> Image.Image:
        """OHLCV DataFrame을 PIL Image로 렌더링한다.

        Args:
            ohlcv_df: DatetimeIndex를 가진 Open, High, Low, Close, Volume 컬럼 DataFrame

        Returns:
            224x224 RGB PIL Image
        """
        buf = io.BytesIO()
        mpf.plot(
            ohlcv_df,
            type="candle",
            style=self.style,
            mav=(5, 20),
            volume=False,
            axisoff=True,
            figsize=self.figsize,
            savefig=dict(
                fname=buf, dpi=self.dpi, pad_inches=0, bbox_inches="tight"
            ),
        )
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        img = img.resize((CHART_IMAGE_SIZE, CHART_IMAGE_SIZE), Image.LANCZOS)
        return img

    def generate_tensor(self, ohlcv_df: pd.DataFrame) -> torch.Tensor:
        """OHLCV DataFrame을 (3, 224, 224) 텐서로 변환한다.

        Args:
            ohlcv_df: DatetimeIndex를 가진 OHLCV DataFrame

        Returns:
            (3, 224, 224) float32 텐서, 값 범위 [0, 1]
        """
        img = self._render_to_pil(ohlcv_df)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def save_chart(self, ohlcv_df: pd.DataFrame, path: str) -> None:
        """캔들차트를 이미지 파일로 저장한다.

        Args:
            ohlcv_df: DatetimeIndex를 가진 OHLCV DataFrame
            path: 저장할 파일 경로
        """
        img = self._render_to_pil(ohlcv_df)
        img.save(path)
