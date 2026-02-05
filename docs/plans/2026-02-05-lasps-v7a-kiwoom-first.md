# LASPS v7a Implementation Plan (Kiwoom-First Approach)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 키움 OpenAPI 데이터 수집부터 시작하여, 각 Phase 완료 시 주요 기능을 검증할 수 있는 Sector-Aware 주식 예측 시스템을 구축한다.

**Architecture:** 키움 OpenAPI로 OHLCV/투자자 데이터를 수집하고, 15개 기술지표 + 5D 시장감성을 계산하여 (60,25) 시계열을 구성한다. Linear Transformer + CNN 2-Branch Fusion 모델이 20개 섹터별 Head로 SELL/HOLD/BUY를 분류한다. 3-Phase 학습(Backbone → Sector Heads → Fine-tune)을 적용한다.

**Tech Stack:** Python 3.8+ (32bit for Kiwoom), PyTorch 2.0+, pandas, numpy, mplfinance, FastAPI, anthropic, loguru, pytest

**Platform Note:** 키움 OpenAPI는 Windows 32bit Python 전용이다. macOS 개발 시 Mock을 사용하고, Windows에서 실제 API를 연결한다.

---

## Phase 순서 및 의존 관계

```
Phase 0: 프로젝트 초기화
  ↓
Phase 1: Config & Utils (Foundation)
  ↓
Phase 2: 키움 OpenAPI 데이터 수집 ← 핵심 우선
  ↓
Phase 3: 데이터 프로세서 (지표 + 감성 + 차트)
  ↓
Phase 4: 데이터셋 & 통합 파이프라인
  ↓
Phase 5: 딥러닝 모델
  ↓
Phase 6: 학습 시스템
  ↓
Phase 7: 서비스 & API
```

각 Phase 끝에 **Milestone Test**가 있어 해당 Phase의 핵심 기능을 검증한다.

---

## Phase 0: 프로젝트 초기화

### Task 0.1: 프로젝트 스캐폴딩

**Files:**
- Create: `lasps/__init__.py` 및 모든 하위 `__init__.py`
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
logs/
.DS_Store
```

**Step 6: Commit**

```bash
git init
git add -A
git commit -m "chore: scaffold LASPS v7a project structure"
```

---

## Phase 1: Config & Utils (Foundation)

### Task 1.1: 상수 및 환경설정

**Files:**
- Create: `lasps/utils/constants.py`
- Create: `lasps/config/settings.py`
- Test: `tests/test_config.py`

**Step 1: constants.py 작성**

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

**Step 3: 테스트 작성**

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

**Step 4: 테스트 실행**

Run: `pytest tests/test_config.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add lasps/utils/constants.py lasps/config/settings.py tests/test_config.py
git commit -m "feat: add core constants and environment settings"
```

---

### Task 1.2: 섹터 설정

**Files:**
- Create: `lasps/config/sector_config.py`
- Modify: `tests/test_config.py`

**Step 1: 테스트 작성 (tests/test_config.py에 추가)**

```python
from lasps.config.sector_config import (
    SECTOR_CODES, get_sector_id, get_sector_name,
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

### Task 1.3: TR 설정 및 모델 Config

**Files:**
- Create: `lasps/config/tr_config.py`
- Create: `lasps/config/model_config.py`

**Step 1: tr_config.py 작성**

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

**Step 2: model_config.py 작성**

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

**Step 3: Commit**

```bash
git add lasps/config/tr_config.py lasps/config/model_config.py
git commit -m "feat: add Kiwoom TR config and model hyperparameters"
```

---

### Task 1.4: 로거 및 헬퍼

**Files:**
- Create: `lasps/utils/logger.py`
- Create: `lasps/utils/helpers.py`
- Create: `lasps/utils/metrics.py`
- Test: `tests/test_utils.py`

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
from lasps.utils.constants import PREDICTION_HORIZON, LABEL_THRESHOLD


def compute_label(close_prices: pd.Series, index: int) -> int:
    """5일 후 수익률 기반 라벨 생성 (0=SELL, 1=HOLD, 2=BUY)"""
    if index + PREDICTION_HORIZON >= len(close_prices):
        return -1
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

**Step 4: 테스트 작성**

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

**Step 5: 테스트 실행**

Run: `pytest tests/test_config.py tests/test_utils.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add lasps/utils/ tests/test_utils.py
git commit -m "feat: add logger, helpers (labeling, normalization), and metrics"
```

---

### Phase 1 Milestone Test

Run: `pytest tests/ -v --tb=short`
Expected: 모든 테스트 PASS (test_config.py + test_utils.py)

검증 항목:
- 상수 25차원 피처 일관성
- 20개 섹터 매핑 정확성
- 라벨링 (BUY/SELL/HOLD) 로직
- Min-Max 정규화
- 분류 메트릭 계산

---

## Phase 2: 키움 OpenAPI 데이터 수집 모듈

> **핵심 Phase** - 전체 시스템의 데이터 소스. Mock 기반으로 개발하되, Windows에서 실제 API로 교체 가능한 구조.

### Task 2.1: 키움 API 인터페이스 및 Mock

**Files:**
- Create: `lasps/data/collectors/kiwoom_base.py`
- Create: `lasps/data/collectors/kiwoom_mock.py`
- Test: `tests/test_collectors.py`

**Step 1: 테스트 작성**

```python
# tests/test_collectors.py

import pytest
import pandas as pd
from lasps.data.collectors.kiwoom_base import KiwoomAPIBase
from lasps.data.collectors.kiwoom_mock import KiwoomMockAPI


def test_mock_api_implements_interface():
    mock = KiwoomMockAPI()
    assert isinstance(mock, KiwoomAPIBase)


def test_mock_get_stock_info():
    mock = KiwoomMockAPI()
    info = mock.request("OPT10001", 종목코드="005930")
    assert "종목코드" in info
    assert "업종코드" in info
    assert "종목명" in info


def test_mock_get_daily_ohlcv():
    mock = KiwoomMockAPI()
    data = mock.request("OPT10081", 종목코드="005930")
    assert isinstance(data, list)
    assert len(data) >= 60
    row = data[0]
    assert "일자" in row
    assert "시가" in row
    assert "현재가" in row
    assert "거래량" in row


def test_mock_get_investor_data():
    mock = KiwoomMockAPI()
    data = mock.request("OPT10059", 종목코드="005930")
    assert isinstance(data, list)
    row = data[0]
    assert "외국인순매수" in row
    assert "기관계순매수" in row
```

Run: `pytest tests/test_collectors.py -v`
Expected: FAIL (module not found)

**Step 2: kiwoom_base.py 구현 (추상 인터페이스)**

```python
# lasps/data/collectors/kiwoom_base.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union


class KiwoomAPIBase(ABC):
    """키움 OpenAPI 추상 인터페이스

    실제 API(Windows)와 Mock(macOS/테스트) 모두 이 인터페이스를 구현한다.
    """

    @abstractmethod
    def request(self, tr_code: str, **kwargs) -> Union[Dict, List[Dict]]:
        """TR 요청을 보내고 결과를 반환한다.

        Args:
            tr_code: TR 코드 (OPT10001, OPT10081, OPT10059, OPT10014)
            **kwargs: TR별 파라미터 (종목코드 등)

        Returns:
            단일 dict 또는 dict의 list
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """API 연결 상태 확인"""
        pass
```

**Step 3: kiwoom_mock.py 구현 (현실적 Mock 데이터)**

```python
# lasps/data/collectors/kiwoom_mock.py

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union
from lasps.data.collectors.kiwoom_base import KiwoomAPIBase
from lasps.config.sector_config import SECTOR_CODES


# 대표 종목 Mock 데이터
MOCK_STOCKS = {
    "005930": {"종목명": "삼성전자", "업종코드": "001", "업종명": "전기전자",
               "시가총액": "400000000000000", "PER": "12.5", "PBR": "1.3", "ROE": "10.2"},
    "000660": {"종목명": "SK하이닉스", "업종코드": "001", "업종명": "전기전자",
               "시가총액": "100000000000000", "PER": "8.0", "PBR": "1.5", "ROE": "15.0"},
    "005380": {"종목명": "현대차", "업종코드": "015", "업종명": "운수장비",
               "시가총액": "50000000000000", "PER": "6.0", "PBR": "0.6", "ROE": "12.0"},
    "035420": {"종목명": "NAVER", "업종코드": "003", "업종명": "서비스업",
               "시가총액": "30000000000000", "PER": "25.0", "PBR": "1.8", "ROE": "8.0"},
    "105560": {"종목명": "KB금융", "업종코드": "002", "업종명": "금융업",
               "시가총액": "25000000000000", "PER": "5.0", "PBR": "0.5", "ROE": "9.0"},
}


class KiwoomMockAPI(KiwoomAPIBase):
    """키움 OpenAPI Mock (개발/테스트용)

    현실적인 가격 패턴을 생성하여 다운스트림 모듈 테스트를 가능하게 한다.
    """

    def __init__(self, seed: int = 42):
        self._rng = np.random.RandomState(seed)

    def request(self, tr_code: str, **kwargs) -> Union[Dict, List[Dict]]:
        stock_code = kwargs.get("종목코드", "005930")

        if tr_code == "OPT10001":
            return self._stock_info(stock_code)
        elif tr_code == "OPT10081":
            return self._daily_ohlcv(stock_code)
        elif tr_code == "OPT10059":
            return self._investor_data(stock_code)
        elif tr_code == "OPT10014":
            return self._short_selling(stock_code)
        else:
            raise ValueError(f"Unknown TR code: {tr_code}")

    def is_connected(self) -> bool:
        return True

    def _stock_info(self, stock_code: str) -> Dict:
        if stock_code in MOCK_STOCKS:
            return {"종목코드": stock_code, **MOCK_STOCKS[stock_code]}
        # 알려지지 않은 종목은 랜덤 섹터 배정
        sector_codes = list(SECTOR_CODES.keys())
        sc = self._rng.choice(sector_codes)
        _, name, _ = SECTOR_CODES[sc]
        return {
            "종목코드": stock_code,
            "종목명": f"종목_{stock_code}",
            "업종코드": sc,
            "업종명": name,
            "시가총액": str(self._rng.randint(1e11, 1e14)),
            "PER": f"{self._rng.uniform(3, 30):.1f}",
            "PBR": f"{self._rng.uniform(0.3, 3.0):.1f}",
            "ROE": f"{self._rng.uniform(2, 20):.1f}",
        }

    def _daily_ohlcv(self, stock_code: str, days: int = 200) -> List[Dict]:
        """현실적 주가 패턴 생성 (Geometric Brownian Motion)"""
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=days)
        base_price = self._rng.uniform(10000, 200000)
        returns = self._rng.normal(0.0005, 0.02, days)
        prices = base_price * np.cumprod(1 + returns)

        rows = []
        for i, date in enumerate(dates):
            close = int(prices[i])
            daily_vol = abs(self._rng.normal(0.015, 0.005))
            high = int(close * (1 + daily_vol))
            low = int(close * (1 - daily_vol))
            open_ = int(close * (1 + self._rng.normal(0, 0.005)))
            volume = int(self._rng.uniform(100000, 5000000))
            rows.append({
                "일자": date.strftime("%Y%m%d"),
                "시가": str(open_),
                "고가": str(high),
                "저가": str(low),
                "현재가": str(close),
                "거래량": str(volume),
            })
        return rows

    def _investor_data(self, stock_code: str, days: int = 200) -> List[Dict]:
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=days)
        rows = []
        for date in dates:
            foreign = int(self._rng.normal(0, 500000))
            inst = int(self._rng.normal(0, 300000))
            personal = -(foreign + inst)
            rows.append({
                "일자": date.strftime("%Y%m%d"),
                "외국인순매수": str(foreign),
                "기관계순매수": str(inst),
                "개인순매수": str(personal),
            })
        return rows

    def _short_selling(self, stock_code: str, days: int = 200) -> List[Dict]:
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=days)
        rows = []
        for date in dates:
            rows.append({
                "일자": date.strftime("%Y%m%d"),
                "공매도량": str(int(self._rng.uniform(1000, 50000))),
                "공매도비중": f"{self._rng.uniform(0.5, 5.0):.2f}",
            })
        return rows
```

**Step 4: 테스트 실행**

Run: `pytest tests/test_collectors.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add lasps/data/collectors/kiwoom_base.py lasps/data/collectors/kiwoom_mock.py tests/test_collectors.py
git commit -m "feat: add Kiwoom API interface and realistic mock implementation"
```

---

### Task 2.2: KiwoomCollector (데이터 수집 핵심)

**Files:**
- Create: `lasps/data/collectors/kiwoom_collector.py`
- Modify: `tests/test_collectors.py`

**Step 1: 테스트 작성 (tests/test_collectors.py에 추가)**

```python
from lasps.data.collectors.kiwoom_collector import KiwoomCollector
from lasps.data.collectors.kiwoom_mock import KiwoomMockAPI


@pytest.fixture
def collector():
    return KiwoomCollector(KiwoomMockAPI(seed=42))


def test_collector_get_stock_info(collector):
    info = collector.get_stock_info("005930")
    assert info["code"] == "005930"
    assert info["name"] == "삼성전자"
    assert info["sector_code"] == "001"
    assert info["sector_id"] == 0  # 전기전자


def test_collector_get_daily_ohlcv(collector):
    df = collector.get_daily_ohlcv("005930", days=60)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 60
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert df["close"].dtype in [float, int, np.int64, np.float64]
    # 날짜순 정렬 확인
    assert df["date"].is_monotonic_increasing


def test_collector_get_investor_data(collector):
    df = collector.get_investor_data("005930", days=60)
    assert isinstance(df, pd.DataFrame)
    assert "foreign_net" in df.columns
    assert "inst_net" in df.columns
    assert len(df) == 60


def test_collector_ohlcv_price_sanity(collector):
    """가격 데이터 정합성: high >= close >= low"""
    df = collector.get_daily_ohlcv("005930", days=60)
    assert (df["high"] >= df["low"]).all()
```

Run: `pytest tests/test_collectors.py::test_collector_get_stock_info -v`
Expected: FAIL

**Step 2: kiwoom_collector.py 구현**

```python
# lasps/data/collectors/kiwoom_collector.py

import time
import pandas as pd
import numpy as np
from typing import Optional
from loguru import logger
from lasps.data.collectors.kiwoom_base import KiwoomAPIBase
from lasps.config.sector_config import get_sector_id
from lasps.config.tr_config import TR_CODES


class KiwoomCollector:
    """키움 OpenAPI 데이터 수집기

    KiwoomAPIBase를 받아 실제 API든 Mock이든 동일하게 동작한다.
    """

    def __init__(self, api: KiwoomAPIBase):
        self.api = api

    def _request(self, tr_code: str, **kwargs):
        interval = TR_CODES.get(tr_code, {}).get("interval", 0.2)
        time.sleep(interval)
        return self.api.request(tr_code, **kwargs)

    def get_stock_info(self, stock_code: str) -> dict:
        """OPT10001: 종목 기본정보 + 섹터 정보"""
        resp = self._request("OPT10001", 종목코드=stock_code)
        sector_code = resp.get("업종코드", "")
        return {
            "code": stock_code,
            "name": resp.get("종목명", ""),
            "market_cap": resp.get("시가총액", ""),
            "per": resp.get("PER", ""),
            "pbr": resp.get("PBR", ""),
            "roe": resp.get("ROE", ""),
            "sector_code": sector_code,
            "sector_name": resp.get("업종명", ""),
            "sector_id": get_sector_id(sector_code),
        }

    def get_daily_ohlcv(self, stock_code: str, days: int = 60) -> pd.DataFrame:
        """OPT10081: 일봉 OHLCV 데이터"""
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

    def get_investor_data(self, stock_code: str, days: int = 60) -> pd.DataFrame:
        """OPT10059: 투자자별 매매 데이터"""
        resp = self._request("OPT10059", 종목코드=stock_code)
        if not isinstance(resp, list):
            resp = [resp]
        rows = []
        for r in resp[:days]:
            rows.append({
                "date": pd.to_datetime(r["일자"]),
                "foreign_net": int(r.get("외국인순매수", 0)),
                "inst_net": int(r.get("기관계순매수", 0)),
            })
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    def get_short_selling(self, stock_code: str) -> pd.DataFrame:
        """OPT10014: 공매도 추이"""
        resp = self._request("OPT10014", 종목코드=stock_code)
        if not isinstance(resp, list):
            resp = [resp]
        rows = []
        for r in resp:
            rows.append({
                "date": pd.to_datetime(r["일자"]),
                "short_volume": int(r.get("공매도량", 0)),
                "short_ratio": float(r.get("공매도비중", 0)),
            })
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
```

**Step 3: 테스트 실행**

Run: `pytest tests/test_collectors.py -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add lasps/data/collectors/kiwoom_collector.py tests/test_collectors.py
git commit -m "feat: implement KiwoomCollector with OPT10001/10081/10059/10014"
```

---

### Task 2.3: DART 부채비율 수집기

**Files:**
- Create: `lasps/data/collectors/dart_collector.py`
- Modify: `tests/test_collectors.py`

**Step 1: 테스트 작성 (tests/test_collectors.py에 추가)**

```python
from unittest.mock import patch
from lasps.data.collectors.dart_collector import DartCollector


def test_dart_collector_returns_ratio():
    mock_response_data = {
        "status": "000",
        "list": [{"account_nm": "부채비율", "thstrm_dt": "45.2"}],
    }
    with patch("lasps.data.collectors.dart_collector.requests") as mock_req:
        mock_req.get.return_value.status_code = 200
        mock_req.get.return_value.json.return_value = mock_response_data
        collector = DartCollector(api_key="test_key")
        ratio = collector.get_debt_ratio("00126380")
        assert isinstance(ratio, float)


def test_dart_collector_handles_error():
    with patch("lasps.data.collectors.dart_collector.requests") as mock_req:
        mock_req.get.side_effect = Exception("Network error")
        collector = DartCollector(api_key="test_key")
        ratio = collector.get_debt_ratio("00126380")
        assert ratio is None
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
                items = data.get("list", [])
                for item in items:
                    if "부채비율" in item.get("account_nm", ""):
                        return float(item.get("thstrm_dt", "0").replace(",", ""))
        except Exception as e:
            logger.warning(f"DART API error for {corp_code}: {e}")
        return None
```

**Step 3: 테스트 실행 및 Commit**

Run: `pytest tests/test_collectors.py -v`
Expected: ALL PASS

```bash
git add lasps/data/collectors/dart_collector.py tests/test_collectors.py
git commit -m "feat: implement DART debt ratio collector"
```

---

### Task 2.4: 수집 모듈 통합 테스트 스크립트

**Files:**
- Create: `tests/test_collectors_integration.py`

이 테스트는 Mock API로 전체 수집 파이프라인을 end-to-end로 검증한다.

**Step 1: 통합 테스트 작성**

```python
# tests/test_collectors_integration.py

"""Phase 2 Milestone: 키움 데이터 수집 모듈 통합 테스트

Mock API를 사용하여 전체 수집 파이프라인이 올바르게 동작하는지 검증한다.
"""

import pytest
import pandas as pd
from lasps.data.collectors.kiwoom_mock import KiwoomMockAPI
from lasps.data.collectors.kiwoom_collector import KiwoomCollector
from lasps.config.sector_config import SECTOR_CODES


@pytest.fixture
def collector():
    return KiwoomCollector(KiwoomMockAPI(seed=42))


class TestEndToEndCollection:
    """전체 데이터 수집 흐름 테스트"""

    def test_full_stock_pipeline(self, collector):
        """단일 종목 전체 데이터 수집"""
        code = "005930"

        # 1. 기본정보
        info = collector.get_stock_info(code)
        assert info["sector_id"] >= 0
        assert info["sector_id"] < 20

        # 2. OHLCV
        ohlcv = collector.get_daily_ohlcv(code, days=60)
        assert len(ohlcv) == 60
        assert ohlcv["close"].notna().all()

        # 3. 투자자 데이터
        investor = collector.get_investor_data(code, days=60)
        assert len(investor) == 60

        # 4. 공매도
        short = collector.get_short_selling(code)
        assert len(short) > 0

    def test_multiple_stocks(self, collector):
        """여러 종목 순차 수집"""
        codes = ["005930", "000660", "005380", "035420", "105560"]
        results = []
        for code in codes:
            info = collector.get_stock_info(code)
            ohlcv = collector.get_daily_ohlcv(code, days=60)
            results.append({"info": info, "ohlcv_len": len(ohlcv)})

        assert len(results) == 5
        # 모든 종목이 올바른 섹터에 매핑되는지
        sector_ids = [r["info"]["sector_id"] for r in results]
        assert all(0 <= sid < 20 for sid in sector_ids)

    def test_ohlcv_data_quality(self, collector):
        """OHLCV 데이터 품질 검증"""
        ohlcv = collector.get_daily_ohlcv("005930", days=100)

        # 기본 정합성
        assert (ohlcv["high"] >= ohlcv["low"]).all()
        assert (ohlcv["volume"] > 0).all()
        assert (ohlcv["close"] > 0).all()

        # 날짜 연속성 (영업일 기준 큰 간격 없어야)
        date_diffs = ohlcv["date"].diff().dropna()
        assert date_diffs.max() <= pd.Timedelta(days=5)  # 주말+공휴일 감안

    def test_investor_data_balance(self, collector):
        """투자자 데이터: 외국인+기관+개인 ≈ 0 (Mock 검증)"""
        investor = collector.get_investor_data("005930", days=60)
        assert "foreign_net" in investor.columns
        assert "inst_net" in investor.columns
```

**Step 2: 테스트 실행**

Run: `pytest tests/test_collectors_integration.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/test_collectors_integration.py
git commit -m "test: add Phase 2 milestone - end-to-end collection integration tests"
```

---

### Phase 2 Milestone Test

Run: `pytest tests/test_collectors.py tests/test_collectors_integration.py -v`
Expected: ALL PASS

검증 항목:
- KiwoomAPIBase 인터페이스 준수 (Mock이 인터페이스 구현)
- OPT10001: 종목정보 + 섹터 매핑 (20개 섹터)
- OPT10081: 일봉 OHLCV (날짜순, 양수, high >= low)
- OPT10059: 투자자별 매매 데이터
- OPT10014: 공매도 데이터
- DART: 부채비율 수집 (에러 핸들링 포함)
- 복수 종목 순차 수집 정상 동작
- 실제 API 교체 시 KiwoomAPIBase만 교체하면 됨

---

## Phase 3: 데이터 프로세서 (지표 + 감성 + 차트)

### Task 3.1: 기술지표 15개 계산기

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

### Task 3.2: 시장 감성 5차원 계산기

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


def test_output_columns(sample_ohlcv):
    calc = MarketSentimentCalculator()
    result = calc.calculate(sample_ohlcv)
    for col in ["volume_ratio", "volatility_ratio", "gap_direction",
                "rsi_norm", "foreign_inst_flow"]:
        assert col in result.columns


def test_value_ranges(sample_ohlcv):
    calc = MarketSentimentCalculator()
    result = calc.calculate(sample_ohlcv)
    assert (result["volume_ratio"] >= 0).all()
    assert (result["volume_ratio"] <= 1).all()
    assert (result["gap_direction"] >= -1).all()
    assert (result["gap_direction"] <= 1).all()
    assert (result["rsi_norm"] >= 0).all()
    assert (result["rsi_norm"] <= 1).all()


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
    assert (result["foreign_inst_flow"] != 0).any()
```

Run: `pytest tests/test_sentiment.py -v`
Expected: FAIL

**Step 2: market_sentiment.py 구현 (PRD Section 4 기준)**

```python
# lasps/data/processors/market_sentiment.py

import numpy as np
import pandas as pd
from typing import Optional


class MarketSentimentCalculator:
    """시장 기반 감성 5차원 계산기 (PRD 기준)

    모든 데이터는 키움 OpenAPI에서 수집.
    Claude API 불필요 -> 비용 50% 절감.
    """

    LOOKBACK = 20

    DEFAULT_VALUES = {
        "volume_ratio": 0.33,
        "volatility_ratio": 0.33,
        "gap_direction": 0.0,
        "rsi_norm": 0.5,
        "foreign_inst_flow": 0.0,
    }

    def calculate(
        self,
        ohlcv_df: pd.DataFrame,
        investor_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        df = ohlcv_df.copy()

        # 1. volume_ratio: clip(0, 3) / 3
        df["volume_ma20"] = df["volume"].rolling(self.LOOKBACK).mean()
        df["volume_ratio"] = (df["volume"] / df["volume_ma20"]).clip(0, 3) / 3

        # 2. volatility_ratio: true_range / atr_20, clip(0, 3) / 3
        df["true_range"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1)),
            ),
        )
        df["atr_20"] = df["true_range"].rolling(self.LOOKBACK).mean()
        df["volatility_ratio"] = (df["true_range"] / df["atr_20"]).clip(0, 3) / 3

        # 3. gap_direction: clip(-0.1, 0.1) * 10
        df["prev_close"] = df["close"].shift(1)
        df["gap_pct"] = (df["open"] - df["prev_close"]) / df["prev_close"]
        df["gap_direction"] = df["gap_pct"].clip(-0.1, 0.1) * 10

        # 4. rsi_norm: RSI(14) / 100
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi_norm"] = df["rsi"] / 100

        # 5. foreign_inst_flow: sign * log10 method
        if investor_df is not None and len(investor_df) > 0:
            merged = df.merge(investor_df, on="date", how="left")
            merged["total_flow"] = merged["foreign_net"].fillna(0) + merged["inst_net"].fillna(0)
            merged["flow_sign"] = np.sign(merged["total_flow"])
            merged["flow_magnitude"] = np.minimum(
                1, np.log10(abs(merged["total_flow"]) + 1) / 8
            )
            merged["foreign_inst_flow"] = merged["flow_sign"] * merged["flow_magnitude"]
            df = merged
        else:
            df["foreign_inst_flow"] = 0.0

        # Fill NaN with defaults
        result = df[["date", "volume_ratio", "volatility_ratio",
                     "gap_direction", "rsi_norm", "foreign_inst_flow"]].copy()
        for col, default in self.DEFAULT_VALUES.items():
            result[col] = result[col].fillna(default)

        return result

    def get_feature_names(self) -> list:
        return ["volume_ratio", "volatility_ratio", "gap_direction",
                "rsi_norm", "foreign_inst_flow"]
```

**Step 3: 테스트 실행**

Run: `pytest tests/test_sentiment.py -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add lasps/data/processors/market_sentiment.py tests/test_sentiment.py
git commit -m "feat: implement 5D market sentiment calculator"
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


def test_tensor_value_range(sample_ohlcv):
    gen = ChartGenerator()
    tensor = gen.generate_tensor(sample_ohlcv)
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_generate_saves_file(sample_ohlcv, tmp_path):
    gen = ChartGenerator()
    path = tmp_path / "test_chart.png"
    gen.save_chart(sample_ohlcv, str(path))
    assert path.exists()
```

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
    """캔들차트 이미지 생성기 (224x224)"""

    def __init__(self, style: str = "yahoo"):
        self.style = style
        self.dpi = 100
        self.figsize = (2.56, 2.56)

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
        return torch.from_numpy(arr).permute(2, 0, 1)

    def save_chart(self, ohlcv_df: pd.DataFrame, path: str) -> None:
        img = self._render_to_pil(ohlcv_df)
        img.save(path)
```

**Step 3: 테스트 실행 및 Commit**

Run: `pytest tests/test_chart_generator.py -v`
Expected: ALL PASS

```bash
git add lasps/data/processors/chart_generator.py tests/test_chart_generator.py
git commit -m "feat: implement candlestick chart generator (224x224)"
```

---

### Task 3.4: Phase 3 통합 테스트

**Files:**
- Create: `tests/test_processors_integration.py`

Mock 수집 데이터로 전체 파이프라인(수집->지표->감성->차트) 검증.

**Step 1: 통합 테스트 작성**

```python
# tests/test_processors_integration.py

"""Phase 3 Milestone: 수집 -> 25차원 피처 + 차트 이미지 통합 테스트"""

import pytest
import pandas as pd
import numpy as np
from lasps.data.collectors.kiwoom_mock import KiwoomMockAPI
from lasps.data.collectors.kiwoom_collector import KiwoomCollector
from lasps.data.processors.technical_indicators import TechnicalIndicatorCalculator
from lasps.data.processors.market_sentiment import MarketSentimentCalculator
from lasps.data.processors.chart_generator import ChartGenerator
from lasps.utils.constants import INDICATOR_FEATURES, SENTIMENT_FEATURES, TOTAL_FEATURE_DIM


@pytest.fixture
def collector():
    return KiwoomCollector(KiwoomMockAPI(seed=42))


def test_full_pipeline_single_stock(collector):
    """수집 -> 지표 -> 감성 -> 25차원 벡터 완성"""
    code = "005930"

    ohlcv = collector.get_daily_ohlcv(code, days=150)
    investor = collector.get_investor_data(code, days=150)

    ind_calc = TechnicalIndicatorCalculator()
    with_indicators = ind_calc.calculate(ohlcv)
    for feat in INDICATOR_FEATURES:
        assert feat in with_indicators.columns

    sent_calc = MarketSentimentCalculator()
    sentiment = sent_calc.calculate(ohlcv, investor)
    for feat in SENTIMENT_FEATURES:
        assert feat in sentiment.columns

    merged = with_indicators.merge(sentiment, on="date", how="left")
    from lasps.utils.constants import ALL_FEATURES
    feature_cols = [c for c in ALL_FEATURES if c in merged.columns]

    valid = merged.dropna(subset=feature_cols)
    assert len(valid) >= 60, f"Only {len(valid)} valid rows"

    recent = valid.tail(60)
    feature_matrix = recent[feature_cols].values
    assert feature_matrix.shape == (60, TOTAL_FEATURE_DIM)


def test_chart_from_collected_data(collector):
    """수집 데이터로 차트 이미지 생성"""
    ohlcv = collector.get_daily_ohlcv("005930", days=60)
    chart_df = ohlcv.copy()
    chart_df.index = pd.to_datetime(chart_df["date"])
    chart_df = chart_df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })

    gen = ChartGenerator()
    tensor = gen.generate_tensor(chart_df[["Open", "High", "Low", "Close", "Volume"]])
    assert tensor.shape == (3, 224, 224)
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0
```

**Step 2: 테스트 실행 및 Commit**

Run: `pytest tests/test_processors_integration.py -v`
Expected: ALL PASS

```bash
git add tests/test_processors_integration.py
git commit -m "test: add Phase 3 milestone - collection to features integration test"
```

---

### Phase 3 Milestone Test

Run: `pytest tests/test_indicators.py tests/test_sentiment.py tests/test_chart_generator.py tests/test_processors_integration.py -v`

검증 항목:
- 기술지표 15개 모두 계산
- 시장감성 5차원 범위 준수
- NaN 없이 25차원 피처 벡터 조합 가능
- Mock 수집 -> 가공 -> 차트까지 전체 파이프라인 동작

---

## Phase 4: 데이터셋 & 통합 파이프라인

### Task 4.1: PyTorch Dataset

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
```

**Step 3: 테스트 실행 및 Commit**

Run: `pytest tests/test_dataset.py -v`
Expected: ALL PASS

```bash
git add lasps/data/datasets/stock_dataset.py tests/test_dataset.py
git commit -m "feat: implement multi-modal StockDataset"
```

---

### Task 4.2: 통합 수집기 (IntegratedCollector)

**Files:**
- Create: `lasps/data/collectors/integrated_collector.py`
- Test: `tests/test_integrated_collector.py`

**Step 1: 테스트 작성**

```python
# tests/test_integrated_collector.py

import pytest
from lasps.data.collectors.kiwoom_mock import KiwoomMockAPI
from lasps.data.collectors.integrated_collector import IntegratedCollector


@pytest.fixture
def int_collector():
    return IntegratedCollector(KiwoomMockAPI(seed=42))


def test_collect_single_stock(int_collector):
    result = int_collector.collect_stock_data("005930")
    assert "info" in result
    assert "time_series_25d" in result
    assert "chart_tensor" in result
    assert result["time_series_25d"].shape[1] == 25
    assert result["chart_tensor"].shape == (3, 224, 224)


def test_collect_batch(int_collector):
    codes = ["005930", "000660", "005380"]
    results = int_collector.collect_batch(codes)
    assert len(results) == 3
```

**Step 2: integrated_collector.py 구현**

```python
# lasps/data/collectors/integrated_collector.py

import numpy as np
import pandas as pd
from typing import List, Dict
from loguru import logger
from lasps.data.collectors.kiwoom_base import KiwoomAPIBase
from lasps.data.collectors.kiwoom_collector import KiwoomCollector
from lasps.data.processors.technical_indicators import TechnicalIndicatorCalculator
from lasps.data.processors.market_sentiment import MarketSentimentCalculator
from lasps.data.processors.chart_generator import ChartGenerator
from lasps.utils.constants import (
    OHLCV_FEATURES, INDICATOR_FEATURES, SENTIMENT_FEATURES,
    TIME_SERIES_LENGTH,
)


class IntegratedCollector:
    """수집 -> 가공 -> 피처 벡터 + 차트 전체 파이프라인"""

    def __init__(self, kiwoom_api: KiwoomAPIBase):
        self.kiwoom = KiwoomCollector(kiwoom_api)
        self.indicator_calc = TechnicalIndicatorCalculator()
        self.sentiment_calc = MarketSentimentCalculator()
        self.chart_gen = ChartGenerator()

    def collect_stock_data(self, stock_code: str) -> Dict:
        info = self.kiwoom.get_stock_info(stock_code)
        ohlcv = self.kiwoom.get_daily_ohlcv(stock_code, days=180)
        investor = self.kiwoom.get_investor_data(stock_code, days=180)

        with_indicators = self.indicator_calc.calculate(ohlcv)
        sentiment = self.sentiment_calc.calculate(ohlcv, investor)

        merged = with_indicators.merge(sentiment, on="date", how="left")
        all_feat = OHLCV_FEATURES + INDICATOR_FEATURES + SENTIMENT_FEATURES
        feature_cols = [c for c in all_feat if c in merged.columns]

        valid = merged.dropna(subset=feature_cols)
        recent = valid.tail(TIME_SERIES_LENGTH)
        time_series_25d = recent[feature_cols].values.astype(np.float32)

        chart_ohlcv = ohlcv.tail(TIME_SERIES_LENGTH).copy()
        chart_ohlcv.index = pd.to_datetime(chart_ohlcv["date"])
        chart_df = chart_ohlcv.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })
        chart_tensor = self.chart_gen.generate_tensor(
            chart_df[["Open", "High", "Low", "Close", "Volume"]]
        )

        return {
            "info": info,
            "time_series_25d": time_series_25d,
            "chart_tensor": chart_tensor,
            "sector_id": info["sector_id"],
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

**Step 3: 테스트 실행 및 Commit**

Run: `pytest tests/test_integrated_collector.py -v`
Expected: ALL PASS

```bash
git add lasps/data/collectors/integrated_collector.py tests/test_integrated_collector.py
git commit -m "feat: implement integrated data collection pipeline"
```

---

### Phase 4 Milestone Test

Run: `pytest tests/test_dataset.py tests/test_integrated_collector.py -v`

검증 항목:
- StockDataset: (60,25) + (3,224,224) + sector_id + label 반환
- IntegratedCollector: Mock 수집 -> 25차원 + 차트 한 번에 처리
- DataLoader 호환 dtype (float32, int64)

---

## Phase 5: 딥러닝 모델

### Task 5.1: Linear Transformer Encoder

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

**Step 2: linear_transformer.py 구현**

```python
# lasps/models/linear_transformer.py

import torch
import torch.nn as nn


class LinearTransformerEncoder(nn.Module):
    """Linear Transformer for time series encoding

    Input: (batch, seq_len=60, input_dim=25)
    Output: (batch, hidden_dim=128)
    """

    def __init__(self, input_dim: int = 25, hidden_dim: int = 128,
                 num_layers: int = 4, num_heads: int = 4,
                 dropout: float = 0.2, max_len: int = 120):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        encoded = self.transformer(x)
        return encoded[:, 0, :]
```

**Step 3: 테스트 실행 및 Commit**

Run: `pytest tests/test_models.py::TestLinearTransformerEncoder -v`
Expected: ALL PASS

```bash
git add lasps/models/linear_transformer.py tests/test_models.py
git commit -m "feat: implement Linear Transformer encoder (60x25 -> 128)"
```

---

### Task 5.2: Chart CNN

**Files:**
- Create: `lasps/models/chart_cnn.py`
- Modify: `tests/test_models.py`

**Step 1: 테스트 (tests/test_models.py에 추가)**

```python
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
```

**Step 2: chart_cnn.py 구현**

```python
# lasps/models/chart_cnn.py

import torch
import torch.nn as nn
from typing import List


class ChartCNN(nn.Module):
    """Chart image encoder: (batch, 3, 224, 224) -> (batch, 128)"""

    def __init__(self, input_channels: int = 3, conv_channels: List[int] = None,
                 output_dim: int = 128, dropout: float = 0.3):
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
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(conv_channels[-1], output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.fc(x)
```

**Step 3: 테스트 실행 및 Commit**

```bash
git add lasps/models/chart_cnn.py tests/test_models.py
git commit -m "feat: implement Chart CNN encoder (3x224x224 -> 128)"
```

---

### Task 5.3: Sector-Aware Fusion Model

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
```

**Step 2: sector_aware_model.py 구현**

```python
# lasps/models/sector_aware_model.py

import torch
import torch.nn as nn
from lasps.models.linear_transformer import LinearTransformerEncoder
from lasps.models.chart_cnn import ChartCNN


class SectorAwareFusionModel(nn.Module):
    """Sector-Aware 2-Branch Fusion Model with 20 sector heads"""

    def __init__(self, num_sectors: int = 20, ts_input_dim: int = 25):
        super().__init__()
        self.num_sectors = num_sectors
        self.ts_encoder = LinearTransformerEncoder(
            input_dim=ts_input_dim, hidden_dim=128,
            num_layers=4, num_heads=4, dropout=0.2,
        )
        self.cnn = ChartCNN(conv_channels=[32, 64, 128, 256], output_dim=128)
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

**Step 3: 테스트 실행 및 Commit**

```bash
git add lasps/models/sector_aware_model.py tests/test_sector_model.py
git commit -m "feat: implement Sector-Aware Fusion Model with 20 heads"
```

---

### Task 5.4: QVM Screener

**Files:**
- Create: `lasps/models/qvm_screener.py`
- Test: `tests/test_qvm.py`

**Step 1: 테스트 + 구현**

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
    assert "qvm_score" in result.columns
```

```python
# lasps/models/qvm_screener.py

import pandas as pd


class QVMScreener:
    """Quality-Value-Momentum screener"""

    def screen(self, stocks_df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
        df = stocks_df.copy()
        df["q_score"] = df["roe"].rank(pct=True) + (1 - df["debt_ratio"].rank(pct=True))
        df["v_score"] = (1 - df["per"].rank(pct=True)) + (1 - df["pbr"].rank(pct=True))
        df["m_score"] = df["volume_avg_20"].rank(pct=True) + df["market_cap"].rank(pct=True)
        df["qvm_score"] = df["q_score"] + df["v_score"] + df["m_score"]
        return df.nlargest(top_n, "qvm_score").reset_index(drop=True)
```

```bash
git add lasps/models/qvm_screener.py tests/test_qvm.py
git commit -m "feat: implement QVM screener"
```

---

### Phase 5 Milestone Test

Run: `pytest tests/test_models.py tests/test_sector_model.py tests/test_qvm.py -v`

검증 항목:
- Transformer: (B,60,25) -> (B,128)
- CNN: (B,3,224,224) -> (B,128)
- Fusion Model: forward + forward_efficient, probabilities sum = 1
- freeze/unfreeze backbone
- QVM 스크리너 상위 N 선별

---

## Phase 6: 학습 시스템

### Task 6.1: Focal Loss + ThreePhaseTrainer

**Files:**
- Create: `lasps/training/loss_functions.py`
- Create: `lasps/training/trainer.py`
- Create: `scripts/train.py`
- Test: `tests/test_training.py`

**Step 1: 테스트 작성**

```python
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
    for p in tiny_model.ts_encoder.parameters():
        assert p.requires_grad


def test_phase3_runs(tiny_model, tiny_loader):
    trainer = ThreePhaseTrainer(tiny_model, device="cpu")
    metrics = trainer.train_phase3(tiny_loader, tiny_loader, epochs=1)
    assert "train_loss" in metrics
```

**Step 2: loss_functions.py 구현**

```python
# lasps/training/loss_functions.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
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

**Step 3: trainer.py 구현**

```python
# lasps/training/trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict
from loguru import logger
from lasps.training.loss_functions import FocalLoss


class ThreePhaseTrainer:
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.criterion = FocalLoss(num_classes=3, gamma=2.0)

    def _run_epoch(self, loader: DataLoader, optimizer=None) -> float:
        is_train = optimizer is not None
        self.model.train() if is_train else self.model.eval()
        total_loss, n_batches = 0.0, 0

        with torch.set_grad_enabled(is_train):
            for batch in loader:
                ts, img, sid, labels = [b.to(self.device) for b in batch]
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

    def train_phase1(self, train_loader, val_loader,
                     epochs: int = 30, lr: float = 1e-4) -> Dict[str, float]:
        logger.info("Phase 1: Backbone training")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        for epoch in range(epochs):
            train_loss = self._run_epoch(train_loader, optimizer)
            val_loss = self._run_epoch(val_loader)
            scheduler.step()
            logger.info(f"Phase1 [{epoch+1}/{epochs}] train={train_loss:.4f} val={val_loss:.4f}")
        return {"train_loss": train_loss, "val_loss": val_loss}

    def train_phase2(self, sector_loaders: Dict[int, DataLoader],
                     epochs_per_sector: int = 10, lr: float = 5e-4) -> None:
        logger.info("Phase 2: Sector Heads (backbone frozen)")
        self.model.freeze_backbone()
        for sector_id, loader in sector_loaders.items():
            optimizer = torch.optim.AdamW(
                self.model.get_sector_head_params(sector_id), lr=lr,
            )
            for epoch in range(epochs_per_sector):
                loss = self._run_epoch(loader, optimizer)
                logger.info(f"Phase2 Sector {sector_id} [{epoch+1}/{epochs_per_sector}] loss={loss:.4f}")
        self.model.unfreeze_backbone()

    def train_phase3(self, train_loader, val_loader,
                     epochs: int = 5, lr: float = 1e-5) -> Dict[str, float]:
        logger.info("Phase 3: End-to-End fine-tuning")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        for epoch in range(epochs):
            train_loss = self._run_epoch(train_loader, optimizer)
            val_loss = self._run_epoch(val_loader)
            scheduler.step()
            logger.info(f"Phase3 [{epoch+1}/{epochs}] train={train_loss:.4f} val={val_loss:.4f}")
        return {"train_loss": train_loss, "val_loss": val_loss}
```

**Step 4: scripts/train.py** (Phase 6.3 참고 - 기존 계획서와 동일)

**Step 5: 테스트 실행 및 Commit**

Run: `pytest tests/test_training.py -v`
Expected: ALL PASS

```bash
git add lasps/training/ tests/test_training.py scripts/train.py
git commit -m "feat: implement Focal Loss, ThreePhaseTrainer, and training script"
```

---

### Phase 6 Milestone Test

Run: `pytest tests/test_training.py -v`

검증 항목:
- FocalLoss: 완벽 예측 시 loss near 0
- Phase 1/2/3 각각 1 epoch 정상 실행
- Phase 2 후 backbone unfreeze 확인

---

## Phase 7: 서비스 & API

### Task 7.1: Predictor + LLM Analyst + API + Scripts

**Files:**
- Create: `lasps/services/predictor.py`
- Create: `lasps/services/llm_analyst.py`
- Create: `lasps/api/main.py`
- Create: `scripts/daily_batch.py`
- Create: `scripts/historical_data.py`
- Test: `tests/test_predictor.py`

**Step 1: 테스트 작성**

```python
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
```

**Step 2: predictor.py 구현**

```python
# lasps/services/predictor.py

import torch
import torch.nn as nn
from typing import Dict
from lasps.utils.constants import CLASS_NAMES


class SectorAwarePredictor:
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def predict(self, time_series, chart_image, sector_id) -> Dict:
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
    def predict_batch(self, time_series, chart_image, sector_id) -> Dict:
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

**Step 3: llm_analyst.py, api/main.py, scripts 구현** (기존 계획서 Phase 7-8 코드와 동일)

**Step 4: 테스트 실행 및 Commit**

Run: `pytest tests/test_predictor.py -v`
Expected: ALL PASS

```bash
git add lasps/services/ lasps/api/ scripts/ tests/test_predictor.py
git commit -m "feat: add predictor, LLM analyst, FastAPI, and batch scripts"
```

---

### Phase 7 Milestone Test (Final)

Run: `pytest tests/ -v --tb=short`
Expected: ALL PASS (전체 테스트 스위트)

검증 항목:
- 전체 모듈 import 성공
- 모든 단위 테스트 통과
- 모든 통합 테스트 통과

---

## 전체 Milestone 요약

| Phase | 핵심 검증 | 테스트 명령어 |
|-------|-----------|---------------|
| **1** | 상수, 섹터, 라벨링 | `pytest tests/test_config.py tests/test_utils.py -v` |
| **2** | 키움 수집 (Mock) | `pytest tests/test_collectors.py tests/test_collectors_integration.py -v` |
| **3** | 지표+감성+차트 | `pytest tests/test_indicators.py tests/test_sentiment.py tests/test_chart_generator.py tests/test_processors_integration.py -v` |
| **4** | Dataset, 통합수집 | `pytest tests/test_dataset.py tests/test_integrated_collector.py -v` |
| **5** | 모델 forward pass | `pytest tests/test_models.py tests/test_sector_model.py tests/test_qvm.py -v` |
| **6** | 3-Phase 학습 | `pytest tests/test_training.py -v` |
| **7** | 예측 서비스, 전체 | `pytest tests/ -v --tb=short` |

---

## 테스트 전략

```bash
# 전체 테스트
pytest tests/ -v --tb=short

# 커버리지
pytest tests/ --cov=lasps --cov-report=term-missing

# 특정 Phase만
pytest tests/test_collectors.py tests/test_collectors_integration.py -v  # Phase 2
```
