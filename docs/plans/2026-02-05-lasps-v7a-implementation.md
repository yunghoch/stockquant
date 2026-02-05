# LASPS v7a Sector-Aware Stock Prediction System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** OHLCV + 기술지표 15개 + 시장감성 5차원을 입력으로, 20개 섹터별 전용 Head를 가진 2-Branch(Transformer+CNN) Fusion 모델로 SELL/HOLD/BUY를 예측하는 시스템을 구축한다.

**Architecture:** Linear Transformer가 (60, 25) 시계열을 인코딩하고, CNN이 224x224 캔들차트 이미지를 인코딩한 뒤, Shared Fusion Layer에서 결합하여 섹터별 Head(20개)로 3-class 분류한다. 3-Phase 학습(Backbone → Sector Heads → End-to-End Fine-tune)으로 범용 패턴과 섹터 특성을 분리 학습한다.

**Tech Stack:** Python 3.8+ (32bit for Kiwoom), PyTorch 2.0+, pandas, numpy, mplfinance, ta, FastAPI, anthropic, loguru, pytest

---

## Phase 0: 프로젝트 초기화

### Task 0.1: 프로젝트 스캐폴딩

**Files:**
- Create: `lasps/__init__.py`
- Create: `lasps/config/__init__.py`
- Create: `lasps/data/__init__.py`, `lasps/data/collectors/__init__.py`, `lasps/data/processors/__init__.py`, `lasps/data/datasets/__init__.py`
- Create: `lasps/models/__init__.py`
- Create: `lasps/training/__init__.py`
- Create: `lasps/services/__init__.py`
- Create: `lasps/api/__init__.py`
- Create: `lasps/utils/__init__.py`
- Create: `tests/__init__.py`
- Create: `scripts/` (directory)
- Create: `requirements.txt`
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`

**Step 1: 디렉토리 구조 생성**

```bash
mkdir -p lasps/{config,data/{collectors,processors,datasets},models,training,services,api,utils}
mkdir -p tests scripts
touch lasps/__init__.py
touch lasps/config/__init__.py
touch lasps/data/__init__.py lasps/data/collectors/__init__.py
touch lasps/data/processors/__init__.py lasps/data/datasets/__init__.py
touch lasps/models/__init__.py lasps/training/__init__.py
touch lasps/services/__init__.py lasps/api/__init__.py
touch lasps/utils/__init__.py tests/__init__.py
```

**Step 2: requirements.txt 작성**

```txt
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
pyqt5>=5.15.0
anthropic>=0.18.0
ta>=0.10.0
mplfinance>=0.12.0
pillow>=9.0.0
fastapi>=0.100.0
uvicorn>=0.22.0
python-dotenv>=1.0.0
loguru>=0.7.0
tqdm>=4.65.0
pytest>=7.0.0
pytest-cov>=4.0.0
```

**Step 3: pyproject.toml 작성**

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "lasps"
version = "7.0.0a1"
description = "LASPS v7a: Sector-Aware Stock Prediction System"
requires-python = ">=3.8"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short"
```

**Step 4: .env.example 작성**

```env
KIWOOM_ACCOUNT=
DART_API_KEY=
ANTHROPIC_API_KEY=
MODEL_PATH=./checkpoints
LOG_LEVEL=INFO
```

**Step 5: .gitignore 작성**

```
__pycache__/
*.pyc
.env
checkpoints/
data/raw/
data/processed/
*.pt
*.pth
.venv/
```

**Step 6: Commit**

```bash
git init
git add -A
git commit -m "chore: scaffold LASPS v7a project structure"
```

---

## Phase 1: Config 모듈 (의존성 없는 순수 설정)

### Task 1.1: 상수 및 환경설정

**Files:**
- Create: `lasps/utils/constants.py`
- Create: `lasps/config/settings.py`
- Test: `tests/test_config.py`

**Step 1: 상수 파일 작성**

```python
# lasps/utils/constants.py

# 시계열 구조
TIME_SERIES_LENGTH = 60
OHLCV_DIM = 5
INDICATOR_DIM = 15
SENTIMENT_DIM = 5
TOTAL_FEATURE_DIM = OHLCV_DIM + INDICATOR_DIM + SENTIMENT_DIM  # 25

# 차트 이미지
CHART_IMAGE_SIZE = 224
CHART_IMAGE_CHANNELS = 3

# 분류
NUM_CLASSES = 3
CLASS_NAMES = {0: "SELL", 1: "HOLD", 2: "BUY"}

# 라벨
PREDICTION_HORIZON = 5
LABEL_THRESHOLD = 0.03

# 섹터
NUM_SECTORS = 20

# 피처 인덱스 범위
OHLCV_INDICES = (0, 5)
INDICATOR_INDICES = (5, 20)
SENTIMENT_INDICES = (20, 25)

OHLCV_FEATURES = ["open", "high", "low", "close", "volume"]
INDICATOR_FEATURES = [
    "ma5", "ma20", "ma60", "ma120",
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_middle", "bb_lower", "bb_width", "atr",
    "obv", "volume_ma20",
]
SENTIMENT_FEATURES = [
    "volume_ratio", "volatility_ratio", "gap_direction",
    "rsi_norm", "foreign_inst_flow",
]
ALL_FEATURES = OHLCV_FEATURES + INDICATOR_FEATURES + SENTIMENT_FEATURES
```

**Step 2: settings.py 작성**

```python
# lasps/config/settings.py

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings:
    KIWOOM_ACCOUNT: str = os.getenv("KIWOOM_ACCOUNT", "")
    DART_API_KEY: str = os.getenv("DART_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    MODEL_PATH: Path = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "checkpoints")))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

settings = Settings()
```

**Step 3: 테스트 작성 및 실행**

```python
# tests/test_config.py

from lasps.utils.constants import (
    TOTAL_FEATURE_DIM, NUM_SECTORS, NUM_CLASSES,
    ALL_FEATURES, TIME_SERIES_LENGTH,
)

def test_feature_dim_consistency():
    assert TOTAL_FEATURE_DIM == 25
    assert len(ALL_FEATURES) == TOTAL_FEATURE_DIM

def test_constants():
    assert NUM_SECTORS == 20
    assert NUM_CLASSES == 3
    assert TIME_SERIES_LENGTH == 60
```

Run: `pytest tests/test_config.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add lasps/utils/constants.py lasps/config/settings.py tests/test_config.py
git commit -m "feat: add core constants and environment settings"
```

---

### Task 1.2: 섹터 설정

**Files:**
- Create: `lasps/config/sector_config.py`
- Test: `tests/test_config.py` (추가)

**Step 1: 테스트 작성**

```python
# tests/test_config.py 에 추가

from lasps.config.sector_config import (
    SECTOR_CODES, get_sector_id, get_sector_name, NUM_SECTORS,
)

def test_sector_codes_count():
    assert len(SECTOR_CODES) == 20

def test_sector_ids_unique():
    ids = [v[0] for v in SECTOR_CODES.values()]
    assert len(set(ids)) == 20
    assert set(ids) == set(range(20))

def test_get_sector_id():
    assert get_sector_id("001") == 0
    assert get_sector_id("020") == 19
    assert get_sector_id("999") == -1

def test_get_sector_name():
    assert get_sector_name(0) == "전기전자"
    assert get_sector_name(19) == "광업"
    assert get_sector_name(99) == "Unknown"
```

Run: `pytest tests/test_config.py::test_sector_codes_count -v`
Expected: FAIL (module not found)

**Step 2: sector_config.py 구현**

```python
# lasps/config/sector_config.py

SECTOR_CODES = {
    "001": (0, "전기전자", 300),
    "002": (1, "금융업", 100),
    "003": (2, "서비스업", 400),
    "004": (3, "의약품", 150),
    "005": (4, "운수창고", 50),
    "006": (5, "유통업", 80),
    "007": (6, "건설업", 70),
    "008": (7, "철강금속", 60),
    "009": (8, "기계", 100),
    "010": (9, "화학", 150),
    "011": (10, "섬유의복", 40),
    "012": (11, "음식료품", 60),
    "013": (12, "비금속광물", 30),
    "014": (13, "종이목재", 20),
    "015": (14, "운수장비", 80),
    "016": (15, "통신업", 20),
    "017": (16, "전기가스업", 15),
    "018": (17, "제조업(기타)", 200),
    "019": (18, "농업임업어업", 10),
    "020": (19, "광업", 10),
}

NUM_SECTORS = 20


def get_sector_id(sector_code: str) -> int:
    if sector_code in SECTOR_CODES:
        return SECTOR_CODES[sector_code][0]
    return -1


def get_sector_name(sector_id: int) -> str:
    for code, (sid, name, _) in SECTOR_CODES.items():
        if sid == sector_id:
            return name
    return "Unknown"
```

**Step 3: 테스트 실행**

Run: `pytest tests/test_config.py -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add lasps/config/sector_config.py tests/test_config.py
git commit -m "feat: add 20-sector configuration with lookup functions"
```

---

### Task 1.3: 모델 하이퍼파라미터 설정

**Files:**
- Create: `lasps/config/model_config.py`
- Create: `lasps/config/tr_config.py`

**Step 1: model_config.py 작성**

```python
# lasps/config/model_config.py

MODEL_CONFIG = {
    "num_sectors": 20,
    "linear_transformer": {
        "input_dim": 25,
        "hidden_dim": 128,
        "num_layers": 4,
        "num_heads": 4,
        "dropout": 0.2,
        "sequence_length": 60,
    },
    "cnn": {
        "input_channels": 3,
        "conv_channels": [32, 64, 128, 256],
        "output_dim": 128,
        "dropout": 0.3,
    },
    "fusion": {
        "shared_dim": 128,
        "sector_head_hidden": 64,
        "num_classes": 3,
        "dropout": 0.3,
    },
}

TRAINING_CONFIG = {
    "batch_size": 128,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "gradient_clip": 1.0,
}

THREE_PHASE_CONFIG = {
    "phase1_backbone": {
        "epochs": 30,
        "lr": 1e-4,
        "scheduler": "cosine",
        "warmup_epochs": 5,
    },
    "phase2_sector_heads": {
        "epochs_per_sector": 10,
        "lr": 5e-4,
        "scheduler": "step",
        "step_size": 5,
        "gamma": 0.5,
        "min_samples": 10000,
    },
    "phase3_finetune": {
        "epochs": 5,
        "lr": 1e-5,
        "scheduler": "cosine",
    },
}

MARKET_SENTIMENT_CONFIG = {
    "lookback_period": 20,
    "default_values": {
        "volume_ratio": 0.33,
        "volatility_ratio": 0.33,
        "gap_direction": 0.0,
        "rsi_norm": 0.5,
        "foreign_inst_flow": 0.0,
    },
}
```

**Step 2: tr_config.py 작성**

```python
# lasps/config/tr_config.py

TR_CODES = {
    "OPT10001": {
        "name": "주식기본정보요청",
        "output": ["종목코드", "종목명", "시가총액", "PER", "PBR", "ROE",
                   "업종코드", "업종명"],
        "interval": 0.2,
    },
    "OPT10081": {
        "name": "주식일봉차트조회요청",
        "output": ["일자", "시가", "고가", "저가", "현재가", "거래량"],
        "interval": 0.2,
    },
    "OPT10059": {
        "name": "종목별투자자기관별요청",
        "output": ["일자", "외국인순매수", "기관계순매수", "개인순매수"],
        "interval": 0.5,
    },
    "OPT10014": {
        "name": "공매도추이요청",
        "output": ["일자", "공매도량", "공매도비중"],
        "interval": 0.5,
    },
}
```

**Step 3: Commit**

```bash
git add lasps/config/model_config.py lasps/config/tr_config.py
git commit -m "feat: add model hyperparameters and Kiwoom TR config"
```

---

## Phase 2: 유틸리티 모듈

### Task 2.1: 로거 및 헬퍼

**Files:**
- Create: `lasps/utils/logger.py`
- Create: `lasps/utils/helpers.py`
- Create: `lasps/utils/metrics.py`

**Step 1: logger.py 작성**

```python
# lasps/utils/logger.py

import sys
from loguru import logger

def setup_logger(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "{message}",
    )
    logger.add(
        "logs/lasps_{time:YYYY-MM-DD}.log",
        level=level,
        rotation="1 day",
        retention="30 days",
    )
```

**Step 2: helpers.py 작성**

```python
# lasps/utils/helpers.py

import pandas as pd
import numpy as np
from lasps.utils.constants import PREDICTION_HORIZON, LABEL_THRESHOLD


def compute_label(close_prices: pd.Series, index: int) -> int:
    """5일 후 수익률 기반 라벨 생성 (0=SELL, 1=HOLD, 2=BUY)"""
    if index + PREDICTION_HORIZON >= len(close_prices):
        return -1  # 라벨 불가
    current = close_prices.iloc[index]
    future = close_prices.iloc[index + PREDICTION_HORIZON]
    ret = (future - current) / current
    if ret >= LABEL_THRESHOLD:
        return 2  # BUY
    elif ret <= -LABEL_THRESHOLD:
        return 0  # SELL
    return 1  # HOLD


def normalize_minmax(series: pd.Series) -> pd.Series:
    """Min-Max 정규화 (0~1)"""
    mn, mx = series.min(), series.max()
    if mx - mn < 1e-10:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)
```

**Step 3: metrics.py 작성**

```python
# lasps/utils/metrics.py

import numpy as np
from typing import Dict


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """3-class 분류 메트릭 계산"""
    accuracy = (y_true == y_pred).mean()

    per_class = {}
    for cls in range(3):
        mask_true = y_true == cls
        mask_pred = y_pred == cls
        tp = (mask_true & mask_pred).sum()
        fp = (~mask_true & mask_pred).sum()
        fn = (mask_true & ~mask_pred).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1}

    macro_f1 = np.mean([v["f1"] for v in per_class.values()])
    return {"accuracy": accuracy, "macro_f1": macro_f1, "per_class": per_class}
```

**Step 4: 테스트**

```python
# tests/test_utils.py

import pandas as pd
import numpy as np
from lasps.utils.helpers import compute_label, normalize_minmax
from lasps.utils.metrics import classification_metrics


def test_compute_label_buy():
    prices = pd.Series([100.0, 101, 102, 103, 104, 110])  # +10%
    assert compute_label(prices, 0) == 2

def test_compute_label_sell():
    prices = pd.Series([100.0, 99, 98, 97, 96, 90])  # -10%
    assert compute_label(prices, 0) == 0

def test_compute_label_hold():
    prices = pd.Series([100.0, 100, 100, 100, 100, 101])  # +1%
    assert compute_label(prices, 0) == 1

def test_compute_label_insufficient_data():
    prices = pd.Series([100.0, 101, 102])
    assert compute_label(prices, 0) == -1

def test_normalize_minmax():
    s = pd.Series([0, 50, 100])
    result = normalize_minmax(s)
    assert result.iloc[0] == 0.0
    assert result.iloc[1] == 0.5
    assert result.iloc[2] == 1.0

def test_classification_metrics():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    m = classification_metrics(y_true, y_pred)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
```

Run: `pytest tests/test_utils.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add lasps/utils/ tests/test_utils.py
git commit -m "feat: add logger, helpers (labeling, normalization), and metrics"
```

---

## Phase 3: 데이터 프로세서

### Task 3.1: 시장 감성 5차원 계산기

**Files:**
- Create: `lasps/data/processors/market_sentiment.py`
- Test: `tests/test_sentiment.py`

**Step 1: 테스트 작성**

```python
# tests/test_sentiment.py

import pandas as pd
import numpy as np
import pytest
from lasps.data.processors.market_sentiment import MarketSentimentCalculator


@pytest.fixture
def sample_ohlcv():
    """40일치 OHLCV 샘플 (lookback 20 + 테스트 데이터)"""
    np.random.seed(42)
    n = 40
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    return pd.DataFrame({
        "date": dates,
        "open": close + np.random.randn(n) * 0.5,
        "high": close + abs(np.random.randn(n)) * 2,
        "low": close - abs(np.random.randn(n)) * 2,
        "close": close,
        "volume": np.random.randint(100_000, 1_000_000, n),
    })


def test_output_shape(sample_ohlcv):
    calc = MarketSentimentCalculator()
    result = calc.calculate(sample_ohlcv)
    assert "volume_ratio" in result.columns
    assert "volatility_ratio" in result.columns
    assert "gap_direction" in result.columns
    assert "rsi_norm" in result.columns
    assert "foreign_inst_flow" in result.columns
    assert len(result) == len(sample_ohlcv)


def test_value_ranges(sample_ohlcv):
    calc = MarketSentimentCalculator()
    result = calc.calculate(sample_ohlcv)
    # NaN 제외 후 범위 검증
    valid = result.dropna()
    assert (valid["volume_ratio"] >= 0).all()
    assert (valid["volume_ratio"] <= 1).all()
    assert (valid["volatility_ratio"] >= 0).all()
    assert (valid["volatility_ratio"] <= 1).all()
    assert (valid["gap_direction"] >= -1).all()
    assert (valid["gap_direction"] <= 1).all()
    assert (valid["rsi_norm"] >= 0).all()
    assert (valid["rsi_norm"] <= 1).all()
    assert (valid["foreign_inst_flow"] >= -1).all()
    assert (valid["foreign_inst_flow"] <= 1).all()


def test_no_nan_after_fillna(sample_ohlcv):
    calc = MarketSentimentCalculator()
    result = calc.calculate(sample_ohlcv)
    for col in calc.get_feature_names():
        assert not result[col].isna().any(), f"{col} has NaN"


def test_with_investor_data(sample_ohlcv):
    calc = MarketSentimentCalculator()
    investor_df = pd.DataFrame({
        "date": sample_ohlcv["date"],
        "foreign_net": np.random.randint(-1_000_000, 1_000_000, len(sample_ohlcv)),
        "inst_net": np.random.randint(-500_000, 500_000, len(sample_ohlcv)),
    })
    result = calc.calculate(sample_ohlcv, investor_df)
    assert not result["foreign_inst_flow"].isna().any()
    # 투자자 데이터가 있으면 0이 아닌 값이 존재해야 함
    assert (result["foreign_inst_flow"] != 0).any()
```

Run: `pytest tests/test_sentiment.py -v`
Expected: FAIL (module not found)

**Step 2: market_sentiment.py 구현 (PRD 코드 그대로)**

PRD Section 4의 `MarketSentimentCalculator` 클래스를 그대로 구현한다. (이 문서의 PRD 참조)

**Step 3: 테스트 실행**

Run: `pytest tests/test_sentiment.py -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add lasps/data/processors/market_sentiment.py tests/test_sentiment.py
git commit -m "feat: implement 5D market sentiment calculator"
```

---

### Task 3.2: 기술지표 15개 계산기

**Files:**
- Create: `lasps/data/processors/technical_indicators.py`
- Test: `tests/test_indicators.py`

**Step 1: 테스트 작성**

```python
# tests/test_indicators.py

import pandas as pd
import numpy as np
import pytest
from lasps.data.processors.technical_indicators import TechnicalIndicatorCalculator
from lasps.utils.constants import INDICATOR_FEATURES


@pytest.fixture
def sample_ohlcv():
    np.random.seed(42)
    n = 150  # MA120 + 여유분
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    return pd.DataFrame({
        "date": dates,
        "open": close + np.random.randn(n) * 0.5,
        "high": close + abs(np.random.randn(n)) * 2,
        "low": close - abs(np.random.randn(n)) * 2,
        "close": close,
        "volume": np.random.randint(100_000, 1_000_000, n).astype(float),
    })


def test_all_15_indicators_present(sample_ohlcv):
    calc = TechnicalIndicatorCalculator()
    result = calc.calculate(sample_ohlcv)
    for feat in INDICATOR_FEATURES:
        assert feat in result.columns, f"Missing: {feat}"


def test_output_length(sample_ohlcv):
    calc = TechnicalIndicatorCalculator()
    result = calc.calculate(sample_ohlcv)
    assert len(result) == len(sample_ohlcv)


def test_ma_ordering(sample_ohlcv):
    """MA 기간이 길수록 더 smooth해야 함"""
    calc = TechnicalIndicatorCalculator()
    result = calc.calculate(sample_ohlcv)
    valid = result.dropna()
    assert valid["ma5"].std() >= valid["ma20"].std()
```

Run: `pytest tests/test_indicators.py -v`
Expected: FAIL

**Step 2: technical_indicators.py 구현**

```python
# lasps/data/processors/technical_indicators.py

import pandas as pd
import numpy as np


class TechnicalIndicatorCalculator:
    """15개 기술지표 계산기

    추세(4) + 모멘텀(4) + 변동성(5) + 거래량(2) = 15개
    """

    def calculate(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        df = ohlcv_df.copy()

        # 추세 (4개)
        df["ma5"] = df["close"].rolling(5).mean()
        df["ma20"] = df["close"].rolling(20).mean()
        df["ma60"] = df["close"].rolling(60).mean()
        df["ma120"] = df["close"].rolling(120).mean()

        # 모멘텀: RSI(14)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # 모멘텀: MACD(12,26,9)
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # 변동성: Bollinger Bands(20, 2)
        df["bb_middle"] = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_middle"] + 2 * bb_std
        df["bb_lower"] = df["bb_middle"] - 2 * bb_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

        # 변동성: ATR(14)
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close = abs(df["low"] - df["close"].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = true_range.rolling(14).mean()

        # 거래량: OBV
        obv = [0.0]
        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i - 1]:
                obv.append(obv[-1] + df["volume"].iloc[i])
            elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
                obv.append(obv[-1] - df["volume"].iloc[i])
            else:
                obv.append(obv[-1])
        df["obv"] = obv

        # 거래량: Volume MA20
        df["volume_ma20"] = df["volume"].rolling(20).mean()

        return df
```

**Step 3: 테스트 실행**

Run: `pytest tests/test_indicators.py -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add lasps/data/processors/technical_indicators.py tests/test_indicators.py
git commit -m "feat: implement 15 technical indicators calculator"
```

---

### Task 3.3: 캔들차트 이미지 생성기

**Files:**
- Create: `lasps/data/processors/chart_generator.py`
- Test: `tests/test_chart_generator.py`

**Step 1: 테스트 작성**

```python
# tests/test_chart_generator.py

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from lasps.data.processors.chart_generator import ChartGenerator


@pytest.fixture
def sample_ohlcv():
    np.random.seed(42)
    n = 60
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    return pd.DataFrame({
        "Open": close + np.random.randn(n) * 0.5,
        "High": close + abs(np.random.randn(n)) * 2,
        "Low": close - abs(np.random.randn(n)) * 2,
        "Close": close,
        "Volume": np.random.randint(100_000, 1_000_000, n),
    }, index=dates)


def test_generate_returns_tensor(sample_ohlcv):
    gen = ChartGenerator()
    tensor = gen.generate_tensor(sample_ohlcv)
    assert tensor.shape == (3, 224, 224)


def test_generate_saves_file(sample_ohlcv, tmp_path):
    gen = ChartGenerator()
    path = tmp_path / "test_chart.png"
    gen.save_chart(sample_ohlcv, str(path))
    assert path.exists()
```

Run: `pytest tests/test_chart_generator.py -v`
Expected: FAIL

**Step 2: chart_generator.py 구현**

```python
# lasps/data/processors/chart_generator.py

import io
import numpy as np
import pandas as pd
import torch
import mplfinance as mpf
from PIL import Image
from lasps.utils.constants import CHART_IMAGE_SIZE


class ChartGenerator:
    """캔들차트 이미지 생성기 (224x224)

    mplfinance로 캔들차트 + MA + Bollinger Bands를 렌더링한 뒤
    PIL로 정확히 224x224로 리사이즈한다.
    """

    def __init__(self, style: str = "yahoo"):
        self.style = style
        self.dpi = 100
        self.figsize = (2.56, 2.56)  # 약간 크게 생성 후 resize

    def _render_to_pil(self, ohlcv_df: pd.DataFrame) -> Image.Image:
        buf = io.BytesIO()
        mpf.plot(
            ohlcv_df,
            type="candle",
            style=self.style,
            mav=(5, 20),
            volume=False,
            axisoff=True,
            figsize=self.figsize,
            savefig=dict(fname=buf, dpi=self.dpi, pad_inches=0,
                         bbox_inches="tight"),
        )
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        img = img.resize((CHART_IMAGE_SIZE, CHART_IMAGE_SIZE), Image.LANCZOS)
        return img

    def generate_tensor(self, ohlcv_df: pd.DataFrame) -> torch.Tensor:
        """DataFrame -> (3, 224, 224) float32 Tensor (0~1)"""
        img = self._render_to_pil(ohlcv_df)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW

    def save_chart(self, ohlcv_df: pd.DataFrame, path: str) -> None:
        img = self._render_to_pil(ohlcv_df)
        img.save(path)
```

**Step 3: 테스트 실행**

Run: `pytest tests/test_chart_generator.py -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add lasps/data/processors/chart_generator.py tests/test_chart_generator.py
git commit -m "feat: implement candlestick chart generator (224x224)"
```

---

### Task 3.4: PyTorch Dataset

**Files:**
- Create: `lasps/data/datasets/stock_dataset.py`
- Test: `tests/test_dataset.py`

**Step 1: 테스트 작성**

```python
# tests/test_dataset.py

import torch
import numpy as np
import pytest
from lasps.data.datasets.stock_dataset import StockDataset


@pytest.fixture
def dummy_dataset(tmp_path):
    """더미 데이터로 Dataset 생성"""
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
```

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL

**Step 2: stock_dataset.py 구현**

```python
# lasps/data/datasets/stock_dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
    """LASPS v7a 멀티모달 데이터셋

    각 샘플: time_series(60,25) + chart_image(3,224,224) + sector_id + label
    """

    def __init__(
        self,
        time_series_path: str,
        chart_images_path: str,
        sector_ids_path: str,
        labels_path: str,
    ):
        self.time_series = np.load(time_series_path, mmap_mode="r")
        self.chart_images = np.load(chart_images_path, mmap_mode="r")
        self.sector_ids = np.load(sector_ids_path)
        self.labels = np.load(labels_path)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "time_series": torch.from_numpy(
                self.time_series[idx].copy()
            ).float(),
            "chart_image": torch.from_numpy(
                self.chart_images[idx].copy()
            ).float(),
            "sector_id": torch.tensor(self.sector_ids[idx], dtype=torch.int64),
            "label": torch.tensor(self.labels[idx], dtype=torch.int64),
        }
```

**Step 3: 테스트 실행**

Run: `pytest tests/test_dataset.py -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add lasps/data/datasets/stock_dataset.py tests/test_dataset.py
git commit -m "feat: implement multi-modal StockDataset (time series + chart + sector)"
```

---

## Phase 4: 모델 구현

### Task 4.1: Linear Transformer Encoder

**Files:**
- Create: `lasps/models/linear_transformer.py`
- Test: `tests/test_models.py`

**Step 1: 테스트 작성**

```python
# tests/test_models.py

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
```

Run: `pytest tests/test_models.py::TestLinearTransformerEncoder -v`
Expected: FAIL

**Step 2: linear_transformer.py 구현**

```python
# lasps/models/linear_transformer.py

import torch
import torch.nn as nn
import math


class LinearTransformerEncoder(nn.Module):
    """Linear Transformer for time series encoding

    입력: (batch, seq_len=60, input_dim=25)
    출력: (batch, hidden_dim=128)

    PyTorch nn.TransformerEncoder 사용, 입력 projection + positional encoding 포함.
    """

    def __init__(
        self,
        input_dim: int = 25,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.2,
        max_len: int = 120,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, hidden_dim)
        """
        seq_len = x.size(1)
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        encoded = self.transformer(x)
        return encoded[:, 0, :]  # CLS-like: 첫 토큰
```

**Step 3: 테스트 실행**

Run: `pytest tests/test_models.py::TestLinearTransformerEncoder -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add lasps/models/linear_transformer.py tests/test_models.py
git commit -m "feat: implement Linear Transformer encoder (60x25 -> 128)"
```

---

### Task 4.2: Chart CNN

**Files:**
- Create: `lasps/models/chart_cnn.py`
- Test: `tests/test_models.py` (추가)

**Step 1: 테스트 작성 (tests/test_models.py에 추가)**

```python
class TestChartCNN:
    def test_output_shape(self):
        from lasps.models.chart_cnn import ChartCNN
        model = ChartCNN(
            conv_channels=[32, 64, 128, 256], output_dim=128,
        )
        x = torch.randn(4, 3, 224, 224)
        out = model(x)
        assert out.shape == (4, 128)

    def test_single_sample(self):
        from lasps.models.chart_cnn import ChartCNN
        model = ChartCNN(
            conv_channels=[32, 64, 128, 256], output_dim=128,
        )
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 128)
```

**Step 2: chart_cnn.py 구현**

```python
# lasps/models/chart_cnn.py

import torch
import torch.nn as nn
from typing import List


class ChartCNN(nn.Module):
    """캔들차트 이미지 인코더

    입력: (batch, 3, 224, 224)
    출력: (batch, output_dim=128)
    """

    def __init__(
        self,
        input_channels: int = 3,
        conv_channels: List[int] = None,
        output_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64, 128, 256]

        layers = []
        in_ch = input_channels
        for out_ch in conv_channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(conv_channels[-1], output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.fc(x)
```

**Step 3: 테스트 실행**

Run: `pytest tests/test_models.py::TestChartCNN -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add lasps/models/chart_cnn.py tests/test_models.py
git commit -m "feat: implement Chart CNN encoder (3x224x224 -> 128)"
```

---

### Task 4.3: Sector-Aware Fusion Model

**Files:**
- Create: `lasps/models/sector_aware_model.py`
- Test: `tests/test_sector_model.py`

**Step 1: 테스트 작성**

```python
# tests/test_sector_model.py

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


def test_forward_efficient(model):
    bs = 16
    ts = torch.randn(bs, 60, 25)
    img = torch.randn(bs, 3, 224, 224)
    sid = torch.randint(0, 20, (bs,))
    out = model.forward_efficient(ts, img, sid)
    assert out["logits"].shape == (bs, 3)


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
    assert len(params) == 4  # Linear(128,64) w,b + Linear(64,3) w,b
```

Run: `pytest tests/test_sector_model.py -v`
Expected: FAIL

**Step 2: sector_aware_model.py 구현 (PRD Section 7 코드 기반)**

PRD의 `SectorAwareFusionModel` 클래스를 구현한다. `LinearTransformerEncoder`와 `ChartCNN`을 import하여 조합한다.

```python
# lasps/models/sector_aware_model.py

import torch
import torch.nn as nn
from lasps.models.linear_transformer import LinearTransformerEncoder
from lasps.models.chart_cnn import ChartCNN


class SectorAwareFusionModel(nn.Module):
    """섹터 인식 2-Branch Fusion 모델 (PRD 기준)"""

    def __init__(self, num_sectors: int = 20, ts_input_dim: int = 25):
        super().__init__()
        self.num_sectors = num_sectors

        self.ts_encoder = LinearTransformerEncoder(
            input_dim=ts_input_dim, hidden_dim=128,
            num_layers=4, num_heads=4, dropout=0.2,
        )
        self.cnn = ChartCNN(
            conv_channels=[32, 64, 128, 256], output_dim=128,
        )
        self.shared_fusion = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
        )
        self.sector_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 3),
            ) for _ in range(num_sectors)
        ])

    def forward(self, time_series, chart_image, sector_id):
        ts_feat = self.ts_encoder(time_series)
        img_feat = self.cnn(chart_image)
        fused = torch.cat([ts_feat, img_feat], dim=1)
        shared_feat = self.shared_fusion(fused)

        batch_size = time_series.size(0)
        logits = torch.zeros(batch_size, 3, device=time_series.device)
        for i in range(batch_size):
            sid = sector_id[i].item()
            logits[i] = self.sector_heads[sid](shared_feat[i:i+1]).squeeze(0)

        return {
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=1),
            "shared_features": shared_feat,
        }

    def forward_efficient(self, time_series, chart_image, sector_id):
        ts_feat = self.ts_encoder(time_series)
        img_feat = self.cnn(chart_image)
        fused = torch.cat([ts_feat, img_feat], dim=1)
        shared_feat = self.shared_fusion(fused)

        batch_size = time_series.size(0)
        logits = torch.zeros(batch_size, 3, device=time_series.device)
        for sid in range(self.num_sectors):
            mask = (sector_id == sid)
            if mask.sum() > 0:
                logits[mask] = self.sector_heads[sid](shared_feat[mask])

        return {"logits": logits, "probabilities": torch.softmax(logits, dim=1)}

    def freeze_backbone(self):
        for param in self.ts_encoder.parameters():
            param.requires_grad = False
        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.shared_fusion.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.ts_encoder.parameters():
            param.requires_grad = True
        for param in self.cnn.parameters():
            param.requires_grad = True
        for param in self.shared_fusion.parameters():
            param.requires_grad = True

    def get_sector_head_params(self, sector_id: int):
        return self.sector_heads[sector_id].parameters()
```

**Step 3: 테스트 실행**

Run: `pytest tests/test_sector_model.py -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add lasps/models/sector_aware_model.py tests/test_sector_model.py
git commit -m "feat: implement Sector-Aware Fusion Model with 20 heads"
```

---

### Task 4.4: QVM Screener

**Files:**
- Create: `lasps/models/qvm_screener.py`
- Test: `tests/test_qvm.py`

**Step 1: 테스트 작성**

```python
# tests/test_qvm.py

import pandas as pd
import pytest
from lasps.models.qvm_screener import QVMScreener


@pytest.fixture
def sample_stocks():
    return pd.DataFrame({
        "code": [f"{i:06d}" for i in range(100)],
        "market_cap": [i * 1e10 for i in range(1, 101)],
        "per": [10 + i * 0.5 for i in range(100)],
        "pbr": [0.5 + i * 0.05 for i in range(100)],
        "roe": [5 + i * 0.3 for i in range(100)],
        "debt_ratio": [50 + i for i in range(100)],
        "volume_avg_20": [1e6 + i * 1e4 for i in range(100)],
    })


def test_screen_returns_50(sample_stocks):
    screener = QVMScreener()
    result = screener.screen(sample_stocks, top_n=50)
    assert len(result) == 50


def test_screen_returns_dataframe(sample_stocks):
    screener = QVMScreener()
    result = screener.screen(sample_stocks, top_n=50)
    assert isinstance(result, pd.DataFrame)
    assert "qvm_score" in result.columns
```

**Step 2: qvm_screener.py 구현**

```python
# lasps/models/qvm_screener.py

import pandas as pd
import numpy as np


class QVMScreener:
    """Quality-Value-Momentum 스크리너

    재무 품질 + 가치 + 유동성 기반으로 상위 N 종목을 선별한다.
    """

    def screen(self, stocks_df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
        df = stocks_df.copy()

        # Quality: ROE 높고 부채비율 낮을수록 좋음
        df["q_score"] = df["roe"].rank(pct=True) + (1 - df["debt_ratio"].rank(pct=True))

        # Value: PER, PBR 낮을수록 좋음
        df["v_score"] = (1 - df["per"].rank(pct=True)) + (1 - df["pbr"].rank(pct=True))

        # Momentum(유동성): 거래량 + 시가총액
        df["m_score"] = df["volume_avg_20"].rank(pct=True) + df["market_cap"].rank(pct=True)

        df["qvm_score"] = df["q_score"] + df["v_score"] + df["m_score"]

        return df.nlargest(top_n, "qvm_score").reset_index(drop=True)
```

**Step 3: 테스트 실행**

Run: `pytest tests/test_qvm.py -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add lasps/models/qvm_screener.py tests/test_qvm.py
git commit -m "feat: implement QVM screener for stock selection"
```

---

## Phase 5: 학습 시스템

### Task 5.1: Loss Functions

**Files:**
- Create: `lasps/training/loss_functions.py`
- Test: `tests/test_training.py`

**Step 1: 테스트 작성**

```python
# tests/test_training.py

import torch
import pytest
from lasps.training.loss_functions import FocalLoss


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
    assert loss.item() < 0.01  # 거의 0
```

**Step 2: loss_functions.py 구현**

```python
# lasps/training/loss_functions.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification

    gamma > 0 이면 쉬운 샘플의 weight를 줄여 어려운 샘플에 집중한다.
    """

    def __init__(self, num_classes: int = 3, gamma: float = 2.0,
                 weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, labels, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
```

**Step 3: 테스트 실행**

Run: `pytest tests/test_training.py -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add lasps/training/loss_functions.py tests/test_training.py
git commit -m "feat: implement Focal Loss for imbalanced 3-class classification"
```

---

### Task 5.2: ThreePhaseTrainer

**Files:**
- Create: `lasps/training/trainer.py`
- Test: `tests/test_training.py` (추가)

**Step 1: 테스트 작성 (tests/test_training.py에 추가)**

```python
from lasps.training.trainer import ThreePhaseTrainer
from lasps.models.sector_aware_model import SectorAwareFusionModel
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def tiny_model():
    return SectorAwareFusionModel(num_sectors=3, ts_input_dim=25)


@pytest.fixture
def tiny_loader():
    n = 16
    ts = torch.randn(n, 60, 25)
    img = torch.randn(n, 3, 224, 224)
    sid = torch.randint(0, 3, (n,))
    labels = torch.randint(0, 3, (n,))
    ds = TensorDataset(ts, img, sid, labels)
    return DataLoader(ds, batch_size=4)


def test_phase1_runs(tiny_model, tiny_loader):
    trainer = ThreePhaseTrainer(tiny_model, device="cpu")
    metrics = trainer.train_phase1(tiny_loader, tiny_loader, epochs=1)
    assert "train_loss" in metrics
    assert "val_loss" in metrics


def test_phase2_runs(tiny_model, tiny_loader):
    trainer = ThreePhaseTrainer(tiny_model, device="cpu")
    sector_loaders = {0: tiny_loader, 1: tiny_loader}
    trainer.train_phase2(sector_loaders, epochs_per_sector=1)
    # backbone이 frozen되었다가 다시 풀려야 함
    for p in tiny_model.ts_encoder.parameters():
        assert p.requires_grad  # phase2 후 unfreeze


def test_phase3_runs(tiny_model, tiny_loader):
    trainer = ThreePhaseTrainer(tiny_model, device="cpu")
    metrics = trainer.train_phase3(tiny_loader, tiny_loader, epochs=1)
    assert "train_loss" in metrics
```

**Step 2: trainer.py 구현**

```python
# lasps/training/trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from loguru import logger
from lasps.training.loss_functions import FocalLoss


class ThreePhaseTrainer:
    """3-Phase 학습 관리자"""

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.criterion = FocalLoss(num_classes=3, gamma=2.0)

    def _run_epoch(self, loader: DataLoader, optimizer=None) -> float:
        is_train = optimizer is not None
        self.model.train() if is_train else self.model.eval()

        total_loss = 0.0
        n_batches = 0

        with torch.set_grad_enabled(is_train):
            for batch in loader:
                ts, img, sid, labels = [
                    b.to(self.device) for b in batch
                ]
                out = self.model.forward_efficient(ts, img, sid)
                loss = self.criterion(out["logits"], labels)

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.model.parameters()),
                        max_norm=1.0,
                    )
                    optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def train_phase1(
        self, train_loader: DataLoader, val_loader: DataLoader,
        epochs: int = 30, lr: float = 1e-4,
    ) -> Dict[str, float]:
        logger.info("Phase 1: Backbone training")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float("inf")
        for epoch in range(epochs):
            train_loss = self._run_epoch(train_loader, optimizer)
            val_loss = self._run_epoch(val_loader)
            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            logger.info(f"Phase1 Epoch {epoch+1}/{epochs} "
                       f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        return {"train_loss": train_loss, "val_loss": val_loss}

    def train_phase2(
        self, sector_loaders: Dict[int, DataLoader],
        epochs_per_sector: int = 10, lr: float = 5e-4,
    ) -> None:
        logger.info("Phase 2: Sector Heads training (backbone frozen)")
        self.model.freeze_backbone()

        for sector_id, loader in sector_loaders.items():
            optimizer = torch.optim.AdamW(
                self.model.get_sector_head_params(sector_id), lr=lr,
            )
            for epoch in range(epochs_per_sector):
                loss = self._run_epoch(loader, optimizer)
                logger.info(f"Phase2 Sector {sector_id} "
                           f"Epoch {epoch+1}/{epochs_per_sector} loss={loss:.4f}")

        self.model.unfreeze_backbone()

    def train_phase3(
        self, train_loader: DataLoader, val_loader: DataLoader,
        epochs: int = 5, lr: float = 1e-5,
    ) -> Dict[str, float]:
        logger.info("Phase 3: End-to-End fine-tuning")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(epochs):
            train_loss = self._run_epoch(train_loader, optimizer)
            val_loss = self._run_epoch(val_loader)
            scheduler.step()
            logger.info(f"Phase3 Epoch {epoch+1}/{epochs} "
                       f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        return {"train_loss": train_loss, "val_loss": val_loss}
```

**Step 3: 테스트 실행**

Run: `pytest tests/test_training.py -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add lasps/training/trainer.py tests/test_training.py
git commit -m "feat: implement ThreePhaseTrainer (backbone -> sector heads -> finetune)"
```

---

## Phase 6: 데이터 수집 (키움 의존)

### Task 6.1: 키움 데이터 수집기

**Files:**
- Create: `lasps/data/collectors/kiwoom_collector.py`
- Test: `tests/test_collectors.py`

**Step 1: 테스트 작성 (키움 API Mock 사용)**

```python
# tests/test_collectors.py

import pytest
from unittest.mock import MagicMock, patch
from lasps.data.collectors.kiwoom_collector import KiwoomCollector


@pytest.fixture
def mock_kiwoom():
    """키움 API 모의 객체"""
    kiwoom = MagicMock()
    kiwoom.request.return_value = {
        "종목코드": "005930",
        "종목명": "삼성전자",
        "시가총액": "400000000000000",
        "업종코드": "001",
        "업종명": "전기전자",
    }
    return kiwoom


def test_get_stock_info(mock_kiwoom):
    collector = KiwoomCollector(mock_kiwoom)
    info = collector.get_stock_info("005930")
    assert info["code"] == "005930"
    assert info["sector_code"] == "001"
    assert info["sector_id"] == 0


def test_get_daily_ohlcv(mock_kiwoom):
    mock_kiwoom.request.return_value = [
        {"일자": "20240101", "시가": "70000", "고가": "71000",
         "저가": "69000", "현재가": "70500", "거래량": "10000000"},
    ]
    collector = KiwoomCollector(mock_kiwoom)
    df = collector.get_daily_ohlcv("005930", days=1)
    assert len(df) == 1
    assert "close" in df.columns
```

**Step 2: kiwoom_collector.py 구현**

```python
# lasps/data/collectors/kiwoom_collector.py

import time
import pandas as pd
from typing import Optional
from loguru import logger
from lasps.config.sector_config import get_sector_id
from lasps.config.tr_config import TR_CODES


class KiwoomCollector:
    """키움 OpenAPI 데이터 수집기"""

    def __init__(self, kiwoom_api):
        self.api = kiwoom_api

    def _request(self, tr_code: str, **kwargs):
        interval = TR_CODES.get(tr_code, {}).get("interval", 0.2)
        time.sleep(interval)
        return self.api.request(tr_code, **kwargs)

    def get_stock_info(self, stock_code: str) -> dict:
        resp = self._request("OPT10001", 종목코드=stock_code)
        sector_code = resp.get("업종코드", "")
        return {
            "code": stock_code,
            "name": resp.get("종목명", ""),
            "market_cap": resp.get("시가총액", ""),
            "sector_code": sector_code,
            "sector_name": resp.get("업종명", ""),
            "sector_id": get_sector_id(sector_code),
        }

    def get_daily_ohlcv(
        self, stock_code: str, days: int = 60
    ) -> pd.DataFrame:
        resp = self._request("OPT10081", 종목코드=stock_code)
        if not isinstance(resp, list):
            resp = [resp]
        rows = []
        for r in resp[:days]:
            rows.append({
                "date": pd.to_datetime(r["일자"]),
                "open": abs(int(r["시가"])),
                "high": abs(int(r["고가"])),
                "low": abs(int(r["저가"])),
                "close": abs(int(r["현재가"])),
                "volume": abs(int(r["거래량"])),
            })
        df = pd.DataFrame(rows)
        return df.sort_values("date").reset_index(drop=True)

    def get_investor_data(self, stock_code: str) -> pd.DataFrame:
        resp = self._request("OPT10059", 종목코드=stock_code)
        if not isinstance(resp, list):
            resp = [resp]
        rows = []
        for r in resp:
            rows.append({
                "date": pd.to_datetime(r["일자"]),
                "foreign_net": int(r.get("외국인순매수", 0)),
                "inst_net": int(r.get("기관계순매수", 0)),
            })
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
```

**Step 3: 테스트 실행**

Run: `pytest tests/test_collectors.py -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add lasps/data/collectors/kiwoom_collector.py tests/test_collectors.py
git commit -m "feat: implement Kiwoom data collector (OPT10001/10081/10059)"
```

---

### Task 6.2: DART 부채비율 수집기

**Files:**
- Create: `lasps/data/collectors/dart_collector.py`
- Test: `tests/test_collectors.py` (추가)

**Step 1: 테스트 작성**

```python
# tests/test_collectors.py 에 추가

from lasps.data.collectors.dart_collector import DartCollector


def test_dart_collector_returns_ratio():
    mock_response = {"debt_ratio": 45.2}
    with patch("lasps.data.collectors.dart_collector.requests") as mock_requests:
        mock_requests.get.return_value.json.return_value = mock_response
        mock_requests.get.return_value.status_code = 200
        collector = DartCollector(api_key="test_key")
        ratio = collector.get_debt_ratio("00126380")
        assert isinstance(ratio, float)
```

**Step 2: dart_collector.py 구현**

```python
# lasps/data/collectors/dart_collector.py

import requests
from typing import Optional
from loguru import logger


class DartCollector:
    """DART API 부채비율 수집기"""

    BASE_URL = "https://opendart.fss.or.kr/api"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_debt_ratio(self, corp_code: str, year: str = "2024") -> Optional[float]:
        try:
            url = f"{self.BASE_URL}/fnlttSinglAcnt.json"
            params = {
                "crtfc_key": self.api_key,
                "corp_code": corp_code,
                "bsns_year": year,
                "reprt_code": "11011",
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return float(data.get("debt_ratio", 0))
        except Exception as e:
            logger.warning(f"DART API error for {corp_code}: {e}")
        return None
```

**Step 3: 테스트 실행 및 Commit**

```bash
git add lasps/data/collectors/dart_collector.py tests/test_collectors.py
git commit -m "feat: implement DART debt ratio collector"
```

---

### Task 6.3: 통합 수집기

**Files:**
- Create: `lasps/data/collectors/integrated_collector.py`

**Step 1: integrated_collector.py 구현**

```python
# lasps/data/collectors/integrated_collector.py

import pandas as pd
from typing import List, Dict
from loguru import logger
from lasps.data.collectors.kiwoom_collector import KiwoomCollector
from lasps.data.processors.technical_indicators import TechnicalIndicatorCalculator
from lasps.data.processors.market_sentiment import MarketSentimentCalculator
from lasps.data.processors.chart_generator import ChartGenerator
from lasps.utils.constants import ALL_FEATURES


class IntegratedCollector:
    """전체 파이프라인을 하나로 통합하는 수집기"""

    def __init__(self, kiwoom_api):
        self.kiwoom = KiwoomCollector(kiwoom_api)
        self.indicator_calc = TechnicalIndicatorCalculator()
        self.sentiment_calc = MarketSentimentCalculator()
        self.chart_gen = ChartGenerator()

    def collect_stock_data(self, stock_code: str) -> Dict:
        """단일 종목의 전체 데이터를 수집하고 가공한다."""
        # 1. 기본정보 + OHLCV + 투자자
        info = self.kiwoom.get_stock_info(stock_code)
        ohlcv = self.kiwoom.get_daily_ohlcv(stock_code, days=180)
        investor = self.kiwoom.get_investor_data(stock_code)

        # 2. 기술지표 15개
        with_indicators = self.indicator_calc.calculate(ohlcv)

        # 3. 시장감성 5차원
        sentiment = self.sentiment_calc.calculate(ohlcv, investor)

        # 4. 차트 이미지 (최근 60일)
        recent_60 = ohlcv.tail(60).copy()
        recent_60.index = pd.to_datetime(recent_60["date"])
        chart_df = recent_60.rename(columns={
            "open": "Open", "high": "High",
            "low": "Low", "close": "Close", "volume": "Volume",
        })
        chart_tensor = self.chart_gen.generate_tensor(chart_df[["Open", "High", "Low", "Close", "Volume"]])

        return {
            "info": info,
            "ohlcv": ohlcv,
            "indicators": with_indicators,
            "sentiment": sentiment,
            "chart_tensor": chart_tensor,
        }

    def collect_batch(self, stock_codes: List[str]) -> List[Dict]:
        results = []
        for code in stock_codes:
            try:
                data = self.collect_stock_data(code)
                results.append(data)
                logger.info(f"Collected {code}: {data['info']['name']}")
            except Exception as e:
                logger.error(f"Failed to collect {code}: {e}")
        return results
```

**Step 2: Commit**

```bash
git add lasps/data/collectors/integrated_collector.py
git commit -m "feat: implement integrated data collection pipeline"
```

---

## Phase 7: 서비스 계층

### Task 7.1: Sector-Aware Predictor

**Files:**
- Create: `lasps/services/predictor.py`
- Test: `tests/test_predictor.py`

**Step 1: 테스트 작성**

```python
# tests/test_predictor.py

import torch
import pytest
from unittest.mock import MagicMock
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
    assert "prediction" in result
    assert "probabilities" in result
    assert result["prediction"] in [0, 1, 2]
    assert len(result["probabilities"]) == 3


def test_predict_batch(predictor):
    bs = 8
    ts = torch.randn(bs, 60, 25)
    img = torch.randn(bs, 3, 224, 224)
    sid = torch.randint(0, 20, (bs,))
    result = predictor.predict_batch(ts, img, sid)
    assert len(result["predictions"]) == bs
```

**Step 2: predictor.py 구현**

```python
# lasps/services/predictor.py

import torch
import torch.nn as nn
from typing import Dict, List
from lasps.utils.constants import CLASS_NAMES


class SectorAwarePredictor:
    """Sector-Aware 예측 서비스"""

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def predict(
        self, time_series: torch.Tensor, chart_image: torch.Tensor,
        sector_id: torch.Tensor,
    ) -> Dict:
        ts = time_series.to(self.device)
        img = chart_image.to(self.device)
        sid = sector_id.to(self.device)

        out = self.model.forward_efficient(ts, img, sid)
        probs = out["probabilities"][0]
        pred = probs.argmax().item()

        return {
            "prediction": pred,
            "label": CLASS_NAMES[pred],
            "probabilities": probs.cpu().tolist(),
            "confidence": probs.max().item(),
        }

    @torch.no_grad()
    def predict_batch(
        self, time_series: torch.Tensor, chart_image: torch.Tensor,
        sector_id: torch.Tensor,
    ) -> Dict:
        ts = time_series.to(self.device)
        img = chart_image.to(self.device)
        sid = sector_id.to(self.device)

        out = self.model.forward_efficient(ts, img, sid)
        preds = out["probabilities"].argmax(dim=1)

        return {
            "predictions": preds.cpu().tolist(),
            "labels": [CLASS_NAMES[p] for p in preds.cpu().tolist()],
            "probabilities": out["probabilities"].cpu().tolist(),
        }
```

**Step 3: 테스트 실행 및 Commit**

```bash
git add lasps/services/predictor.py tests/test_predictor.py
git commit -m "feat: implement Sector-Aware prediction service"
```

---

### Task 7.2: LLM Analyst (Top 10 Claude 분석)

**Files:**
- Create: `lasps/services/llm_analyst.py`

**Step 1: 구현**

```python
# lasps/services/llm_analyst.py

import anthropic
from typing import Dict, List
from loguru import logger


class LLMAnalyst:
    """Top 10 종목 Claude 분석기

    모델 예측 결과 상위 10종목에 대해 Claude API로 상세 분석을 수행한다.
    월 비용: ~$30 (일 1회, 10종목)
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def analyze_stock(self, stock_info: Dict) -> str:
        prompt = (
            f"다음 종목을 분석하고 투자 의견을 제시해주세요.\n\n"
            f"종목명: {stock_info.get('name', '')}\n"
            f"업종: {stock_info.get('sector_name', '')}\n"
            f"모델 예측: {stock_info.get('prediction_label', '')}\n"
            f"신뢰도: {stock_info.get('confidence', 0):.1%}\n"
            f"주요 지표:\n"
            f"- PER: {stock_info.get('per', 'N/A')}\n"
            f"- PBR: {stock_info.get('pbr', 'N/A')}\n"
            f"- ROE: {stock_info.get('roe', 'N/A')}\n\n"
            f"300자 이내로 핵심 포인트만 요약해주세요."
        )

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return f"분석 실패: {e}"

    def analyze_top_stocks(self, stocks: List[Dict]) -> List[Dict]:
        results = []
        for stock in stocks[:10]:
            analysis = self.analyze_stock(stock)
            results.append({**stock, "llm_analysis": analysis})
            logger.info(f"Analyzed: {stock.get('name', 'Unknown')}")
        return results
```

**Step 2: Commit**

```bash
git add lasps/services/llm_analyst.py
git commit -m "feat: implement LLM analyst for Top 10 stock analysis"
```

---

## Phase 8: API 및 스크립트

### Task 8.1: FastAPI 서버

**Files:**
- Create: `lasps/api/main.py`

**Step 1: 구현**

```python
# lasps/api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch

app = FastAPI(title="LASPS v7a", version="7.0.0a1")


class PredictionRequest(BaseModel):
    stock_codes: List[str]


class PredictionResponse(BaseModel):
    code: str
    name: str
    sector: str
    prediction: str
    confidence: float
    probabilities: List[float]
    llm_analysis: Optional[str] = None


# 전역 모델/서비스 (startup에서 초기화)
_predictor = None
_collector = None


@app.on_event("startup")
async def startup():
    global _predictor
    # 체크포인트 로드 로직
    pass


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.post("/predict", response_model=List[PredictionResponse])
async def predict(request: PredictionRequest):
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    # 실제 예측 로직은 서비스 통합 후 연결
    return []


@app.get("/sectors")
async def list_sectors():
    from lasps.config.sector_config import SECTOR_CODES
    return {
        code: {"id": sid, "name": name}
        for code, (sid, name, _) in SECTOR_CODES.items()
    }
```

**Step 2: Commit**

```bash
git add lasps/api/main.py
git commit -m "feat: add FastAPI server skeleton with health/predict/sectors endpoints"
```

---

### Task 8.2: 학습 스크립트

**Files:**
- Create: `scripts/train.py`

**Step 1: 구현**

```python
# scripts/train.py

"""
LASPS v7a 3-Phase 학습 스크립트

Usage:
    python scripts/train.py --data-dir data/processed --device cuda
    python scripts/train.py --phase 2 --checkpoint checkpoints/phase1_best.pt
"""

import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from loguru import logger

from lasps.models.sector_aware_model import SectorAwareFusionModel
from lasps.data.datasets.stock_dataset import StockDataset
from lasps.training.trainer import ThreePhaseTrainer
from lasps.config.model_config import MODEL_CONFIG, THREE_PHASE_CONFIG
from lasps.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="LASPS v7a Training")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--phase", type=int, default=0, help="0=all, 1/2/3=specific phase")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logger("INFO")
    logger.info(f"Device: {args.device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 모델 생성 또는 로드
    model = SectorAwareFusionModel(
        num_sectors=MODEL_CONFIG["num_sectors"],
        ts_input_dim=MODEL_CONFIG["linear_transformer"]["input_dim"],
    )
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
        logger.info(f"Loaded checkpoint: {args.checkpoint}")

    trainer = ThreePhaseTrainer(model, device=args.device)

    data_dir = Path(args.data_dir)

    # 데이터셋 로드
    train_ds = StockDataset(
        str(data_dir / "train_ts.npy"), str(data_dir / "train_charts.npy"),
        str(data_dir / "train_sectors.npy"), str(data_dir / "train_labels.npy"),
    )
    val_ds = StockDataset(
        str(data_dir / "val_ts.npy"), str(data_dir / "val_charts.npy"),
        str(data_dir / "val_sectors.npy"), str(data_dir / "val_labels.npy"),
    )

    batch_size = 128
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    if args.phase in (0, 1):
        cfg = THREE_PHASE_CONFIG["phase1_backbone"]
        trainer.train_phase1(train_loader, val_loader, epochs=cfg["epochs"], lr=cfg["lr"])
        torch.save(model.state_dict(), output_dir / "phase1_best.pt")
        logger.info("Phase 1 complete")

    if args.phase in (0, 2):
        # 섹터별 DataLoader 구성
        # 실제로는 sector_id별로 데이터를 분리해야 함
        logger.info("Phase 2: sector-specific training (requires sector-split data)")

    if args.phase in (0, 3):
        cfg = THREE_PHASE_CONFIG["phase3_finetune"]
        trainer.train_phase3(train_loader, val_loader, epochs=cfg["epochs"], lr=cfg["lr"])
        torch.save(model.state_dict(), output_dir / "phase3_final.pt")
        logger.info("Phase 3 complete")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/train.py
git commit -m "feat: add 3-phase training script"
```

---

### Task 8.3: 일일 배치 스크립트

**Files:**
- Create: `scripts/daily_batch.py`

**Step 1: 구현**

```python
# scripts/daily_batch.py

"""
LASPS v7a 일일 배치 프로세스 (장 마감 후 50분)

Usage:
    python scripts/daily_batch.py
"""

import torch
from datetime import datetime
from pathlib import Path
from loguru import logger

from lasps.config.settings import settings
from lasps.models.sector_aware_model import SectorAwareFusionModel
from lasps.models.qvm_screener import QVMScreener
from lasps.services.predictor import SectorAwarePredictor
from lasps.services.llm_analyst import LLMAnalyst
from lasps.utils.logger import setup_logger


def main():
    setup_logger("INFO")
    today = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"=== LASPS v7a Daily Batch: {today} ===")

    # 1. 키움 로그인
    logger.info("Step 1: Kiwoom login")
    # kiwoom_api = KiwoomAPI()  # 실제 키움 API 연결

    # 2. 전체 종목 기본정보 수집
    logger.info("Step 2: Collecting stock info")

    # 3. DART 부채비율
    logger.info("Step 3: Collecting DART debt ratios")

    # 4. QVM 스크리닝 → 50종목
    logger.info("Step 4: QVM screening -> 50 stocks")

    # 5. 상세 데이터 수집
    logger.info("Step 5: Collecting detailed data for 50 stocks")

    # 6. 시장 감성 계산
    logger.info("Step 6: Computing market sentiment")

    # 7. 기술지표 + 차트 이미지
    logger.info("Step 7: Computing indicators + generating charts")

    # 8. 시계열 구성
    logger.info("Step 8: Assembling time series (60, 25)")

    # 9. Sector-Aware 예측
    logger.info("Step 9: Running Sector-Aware predictions")
    model = SectorAwareFusionModel()
    checkpoint = settings.MODEL_PATH / "phase3_final.pt"
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))

    predictor = SectorAwarePredictor(model, device="cpu")

    # 10. LLM 상세 분석 (Top 10)
    logger.info("Step 10: LLM analysis for Top 10")
    if settings.ANTHROPIC_API_KEY:
        analyst = LLMAnalyst(api_key=settings.ANTHROPIC_API_KEY)

    # 11. 리포트 생성
    logger.info("Step 11: Generating report")
    logger.info(f"=== Daily Batch Complete: {today} ===")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/daily_batch.py
git commit -m "feat: add daily batch processing script skeleton"
```

---

## Phase 9: 히스토리컬 데이터 수집 스크립트

### Task 9.1: 과거 데이터 수집

**Files:**
- Create: `scripts/historical_data.py`

이 스크립트는 2015-01 ~ 2024-12 (10년)의 과거 데이터를 수집하여 학습용 npy 파일로 변환한다.

```python
# scripts/historical_data.py

"""
과거 10년 데이터 수집 및 학습 데이터셋 생성

Usage:
    python scripts/historical_data.py --output data/processed
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from lasps.utils.logger import setup_logger
from lasps.utils.constants import TIME_SERIES_LENGTH, TOTAL_FEATURE_DIM
from lasps.utils.helpers import compute_label

# 데이터 분할 기준
SPLIT_CONFIG = {
    "train": ("2015-01-01", "2022-12-31"),
    "val": ("2023-01-01", "2023-12-31"),
    "test": ("2024-01-01", "2024-12-31"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/processed")
    args = parser.parse_args()
    setup_logger("INFO")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Historical data collection - requires Kiwoom API connection")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Splits: {SPLIT_CONFIG}")

    # 실제 구현 시:
    # 1. 키움 API로 전 종목 10년치 일봉 수집 (OPT10081)
    # 2. 투자자별 데이터 수집 (OPT10059)
    # 3. 기술지표 15개 계산
    # 4. 시장감성 5차원 계산
    # 5. 캔들차트 이미지 생성 (60일 윈도우 슬라이딩)
    # 6. 라벨 생성 (5일 후 수익률 ±3%)
    # 7. 시간순 분할 (train/val/test)
    # 8. npy 파일 저장
    #    - {split}_ts.npy: (N, 60, 25) float32
    #    - {split}_charts.npy: (N, 3, 224, 224) float32
    #    - {split}_sectors.npy: (N,) int64
    #    - {split}_labels.npy: (N,) int64

    logger.info("Done. Output files saved to: {output_dir}")


if __name__ == "__main__":
    main()
```

**Commit:**

```bash
git add scripts/historical_data.py
git commit -m "feat: add historical data collection script skeleton"
```

---

## 구현 순서 요약

```
Phase 0: 프로젝트 스캐폴딩              ← 의존성 없음
Phase 1: Config 모듈 (constants, sectors, model_config, tr_config)
Phase 2: 유틸리티 (logger, helpers, metrics)
Phase 3: 데이터 프로세서 (sentiment, indicators, chart, dataset)
Phase 4: 모델 (transformer, cnn, fusion, qvm)
Phase 5: 학습 (loss, trainer)
Phase 6: 데이터 수집 (kiwoom, dart, integrated)   ← 키움 API 의존
Phase 7: 서비스 (predictor, llm_analyst)
Phase 8: API + 스크립트 (fastapi, train, daily_batch)
Phase 9: 히스토리컬 데이터 수집
```

### 의존 관계 그래프

```
Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
                                  ↓                    ↓
                              Phase 6 ───────────→ Phase 7
                                                       ↓
                                                   Phase 8
                                                       ↓
                                                   Phase 9
```

**핵심:** Phase 3~5 (프로세서 + 모델 + 학습)는 키움 API 없이 개발/테스트 가능하므로 먼저 완성한 뒤, Phase 6(키움 연동)을 진행한다.

---

## 테스트 전략

모든 테스트는 키움 API Mock을 사용하여 오프라인에서 실행 가능해야 한다.

```bash
# 전체 테스트
pytest tests/ -v --tb=short

# 특정 모듈
pytest tests/test_models.py -v
pytest tests/test_sentiment.py -v
pytest tests/test_sector_model.py -v
pytest tests/test_training.py -v

# 커버리지
pytest tests/ --cov=lasps --cov-report=term-missing
```
