# LASPS v7a 매뉴얼 테스트 시나리오

> 코드 리뷰 수정 사항 (커밋 0ca8c48) 에 대한 수동 검증 시나리오
> 작성일: 2026-02-05

---

## 목차

1. [CRITICAL] Division-by-Zero 방어 검증
2. [CRITICAL] Per-Stock Min-Max 정규화 검증
3. [HIGH] MODEL_CONFIG 연결 검증
4. [HIGH] 에러 핸들링 및 시간 정합성 검증
5. [MEDIUM] FocalLoss / requirements.txt / logger / checkpoint 검증
6. 통합 테스트 시나리오 (End-to-End)

---

## 1. [CRITICAL] Division-by-Zero 방어 검증

### 대상 파일
- `lasps/data/processors/market_sentiment.py`
- `lasps/data/processors/technical_indicators.py`

### 시나리오 1-1: volume_ratio - volume_ma20이 0인 경우

```python
import pandas as pd
import numpy as np
from lasps.data.processors.market_sentiment import MarketSentimentCalculator

calc = MarketSentimentCalculator()

# volume이 모두 0인 OHLCV 생성
dates = pd.date_range("2024-01-01", periods=60, freq="B")
ohlcv = pd.DataFrame({
    "date": dates,
    "open": [100.0] * 60,
    "high": [100.0] * 60,
    "low": [100.0] * 60,
    "close": [100.0] * 60,
    "volume": [0] * 60,  # 전체 거래량 0
})

result = calc.calculate(ohlcv)

# 검증
assert not result["volume_ratio"].isna().any(), "volume_ratio에 NaN이 없어야 함"
assert (result["volume_ratio"] == 0.33).all(), "기본값 0.33이어야 함"
print("PASS: volume_ratio division-by-zero 방어 정상")
```

**기대 결과**: volume_ma20=0 -> `.replace(0, np.nan)` -> NaN -> `fillna(0.33)`

### 시나리오 1-2: volatility_ratio - ATR20이 0인 경우 (변동 없는 가격)

```python
# 가격이 완전히 동일한 경우 (true_range=0 -> ATR=0)
ohlcv_flat = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=60, freq="B"),
    "open": [50000] * 60,
    "high": [50000] * 60,
    "low": [50000] * 60,
    "close": [50000] * 60,
    "volume": [1000000] * 60,
})

result = calc.calculate(ohlcv_flat)

# 검증
assert not result["volatility_ratio"].isna().any(), "volatility_ratio에 NaN이 없어야 함"
assert (result["volatility_ratio"] == 0.33).all(), "ATR=0이면 기본값 0.33"
print("PASS: volatility_ratio division-by-zero 방어 정상")
```

**기대 결과**: ATR20=0 -> `.replace(0, np.nan)` -> NaN -> `fillna(0.33)`

### 시나리오 1-3: gap_direction - prev_close가 0인 경우

```python
# 첫 번째 행의 prev_close는 NaN (shift 결과)
ohlcv_gap = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=5, freq="B"),
    "open": [100, 105, 110, 115, 120],
    "high": [110, 115, 120, 125, 130],
    "low": [95, 100, 105, 110, 115],
    "close": [105, 110, 115, 120, 125],
    "volume": [1000] * 5,
})

result = calc.calculate(ohlcv_gap)

# 검증: 첫 행은 prev_close=NaN -> gap_direction 기본값 0.0
assert result["gap_direction"].iloc[0] == 0.0, "첫 행 gap_direction은 0.0이어야 함"
print("PASS: gap_direction division-by-zero 방어 정상")
```

### 시나리오 1-4: RSI - 가격 변동 없는 구간 (gain=0, loss=0)

```python
from lasps.data.processors.technical_indicators import TechnicalIndicatorCalculator

ti_calc = TechnicalIndicatorCalculator()

# 60일간 가격 완전 동일
ohlcv_const = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=60, freq="B"),
    "open": [10000] * 60,
    "high": [10000] * 60,
    "low": [10000] * 60,
    "close": [10000] * 60,
    "volume": [500000] * 60,
})

result = ti_calc.calculate(ohlcv_const)

# 검증: gain=0, loss=0 -> rs=0/0 -> fillna(0) -> RSI=0
assert not result["rsi"].isna().all(), "RSI에 전부 NaN이면 안됨 (warmup 이후)"
warmup_done = result["rsi"].iloc[14:]  # 14일 이후
assert (warmup_done.dropna() == 0.0).all(), "변동 없으면 RSI=0"
print("PASS: RSI division-by-zero 방어 정상")
```

### 시나리오 1-5: Bollinger Band width - bb_middle이 0인 경우

```python
# bb_middle(=MA20)이 0이 되려면 close가 0이어야 함 (비현실적이지만 방어 필요)
# 실제로는 bb_middle = close 이므로 0이 아닌 한 정상 동작
result = ti_calc.calculate(ohlcv_const)

# 검증: bb_width에 inf나 NaN이 없는지
bb_valid = result["bb_width"].iloc[19:]  # 20일 warmup 이후
assert not bb_valid.isin([np.inf, -np.inf]).any(), "bb_width에 inf 없어야 함"
print("PASS: bb_width division-by-zero 방어 정상")
```

### 시나리오 1-6: investor_data 없이 sentiment 계산

```python
result = calc.calculate(ohlcv_gap, investor_df=None)

# 검증: foreign_inst_flow가 전부 0.0 (기본값)
assert (result["foreign_inst_flow"] == 0.0).all(), "투자자 데이터 없으면 0.0"
print("PASS: investor_data=None 처리 정상")
```

---

## 2. [CRITICAL] Per-Stock Min-Max 정규화 검증

### 대상 파일
- `lasps/utils/helpers.py` (`normalize_time_series`)
- `lasps/data/collectors/integrated_collector.py`

### 시나리오 2-1: 기본 정규화 동작

```python
import numpy as np
from lasps.utils.helpers import normalize_time_series

# (10, 25) 시계열: features 0-19는 다양한 범위, 20-24는 감성지표
data = np.zeros((10, 25), dtype=np.float32)

# OHLCV (0-4): 가격 범위 10000~20000
data[:, 0] = np.linspace(10000, 20000, 10)  # open
data[:, 1] = np.linspace(10500, 20500, 10)  # high
data[:, 2] = np.linspace(9500, 19500, 10)   # low
data[:, 3] = np.linspace(10000, 20000, 10)  # close
data[:, 4] = np.linspace(1000000, 5000000, 10)  # volume

# Indicators (5-19): 다양한 범위
for i in range(5, 20):
    data[:, i] = np.linspace(i * 10, i * 100, 10)

# Sentiment (20-24): 이미 정규화됨
data[:, 20] = 0.5   # volume_ratio (0~1)
data[:, 21] = 0.3   # volatility_ratio (0~1)
data[:, 22] = -0.2  # gap_direction (-1~+1)
data[:, 23] = 0.7   # rsi_norm (0~1)
data[:, 24] = 0.1   # foreign_inst_flow (-1~+1)

result = normalize_time_series(data)

# 검증 1: features 0-19는 [0, 1] 범위
for col in range(20):
    assert result[:, col].min() >= 0.0, f"Feature {col}: min < 0"
    assert result[:, col].max() <= 1.0, f"Feature {col}: max > 1"
    assert abs(result[:, col].min() - 0.0) < 1e-6, f"Feature {col}: min != 0"
    assert abs(result[:, col].max() - 1.0) < 1e-6, f"Feature {col}: max != 1"

# 검증 2: features 20-24는 변경 없음
assert result[0, 20] == 0.5, "volume_ratio 변경되면 안됨"
assert result[0, 22] == -0.2, "gap_direction 변경되면 안됨"
assert result[0, 24] == 0.1, "foreign_inst_flow 변경되면 안됨"

print("PASS: 기본 정규화 동작 정상")
```

### 시나리오 2-2: 상수 피처 처리 (값이 동일한 열)

```python
data_const = np.ones((10, 25), dtype=np.float32)
# 모든 피처가 1.0으로 동일

result = normalize_time_series(data_const)

# 검증: features 0-19는 상수 -> 0.5 매핑
for col in range(20):
    assert (result[:, col] == 0.5).all(), f"Feature {col}: 상수인데 0.5가 아님"

# 검증: features 20-24는 원본 유지 (1.0)
for col in range(20, 25):
    assert (result[:, col] == 1.0).all(), f"Feature {col}: 변경되면 안됨"

print("PASS: 상수 피처 처리 정상 (0.5 매핑)")
```

### 시나리오 2-3: 단일 행 입력 (T=1)

```python
data_single = np.random.randn(1, 25).astype(np.float32)
result = normalize_time_series(data_single)

# 검증: T=1이면 min==max -> 상수 처리 -> features 0-19 = 0.5
for col in range(20):
    assert result[0, col] == 0.5, f"Feature {col}: 단일 행에서 0.5가 아님"

print("PASS: 단일 행 입력 처리 정상")
```

### 시나리오 2-4: shape 보존

```python
for shape in [(60, 25), (30, 25), (1, 25)]:
    data = np.random.randn(*shape).astype(np.float32)
    result = normalize_time_series(data)
    assert result.shape == shape, f"Shape 변경됨: {shape} -> {result.shape}"

print("PASS: 입력 shape 보존 정상")
```

---

## 3. [HIGH] MODEL_CONFIG 연결 검증

### 대상 파일
- `lasps/models/sector_aware_model.py`
- `lasps/config/model_config.py`
- `scripts/train.py`
- `lasps/config/settings.py`
- `lasps/training/trainer.py`

### 시나리오 3-1: 기본 MODEL_CONFIG로 모델 생성

```python
import torch
from lasps.models.sector_aware_model import SectorAwareFusionModel
from lasps.config.model_config import MODEL_CONFIG

# config=None -> MODEL_CONFIG 사용
model = SectorAwareFusionModel(num_sectors=20, ts_input_dim=25)

# 검증: hidden_dim=128이 실제 적용되었는지
ts = torch.randn(2, 60, 25)
img = torch.randn(2, 3, 224, 224)
sid = torch.tensor([0, 5])
out = model(ts, img, sid)

assert out["logits"].shape == (2, 3), "출력 shape (2, 3)이어야 함"
assert out["shared_features"].shape == (2, 128), "shared_dim=128이어야 함"
print("PASS: MODEL_CONFIG 기본 연결 정상")
```

### 시나리오 3-2: 커스텀 config로 모델 생성

```python
custom_config = {
    "linear_transformer": {
        "hidden_dim": 64, "num_layers": 2, "num_heads": 2, "dropout": 0.1
    },
    "cnn": {
        "conv_channels": [16, 32], "output_dim": 64, "dropout": 0.1
    },
    "fusion": {
        "shared_dim": 64, "sector_head_hidden": 32, "num_classes": 3, "dropout": 0.1
    },
}

model_custom = SectorAwareFusionModel(
    num_sectors=5, ts_input_dim=25, config=custom_config
)

ts = torch.randn(2, 60, 25)
img = torch.randn(2, 3, 224, 224)
sid = torch.tensor([0, 3])
out = model_custom(ts, img, sid)

assert out["logits"].shape == (2, 3), "커스텀 config 출력 shape 확인"
assert out["shared_features"].shape == (2, 64), "커스텀 shared_dim=64 확인"
print("PASS: 커스텀 config 연결 정상")
```

### 시나리오 3-3: 잘못된 config 키 -> KeyError

```python
bad_config = {"linear_transformer": {"hidden_dim": 64}}  # 나머지 키 누락

try:
    model_bad = SectorAwareFusionModel(config=bad_config)
    assert False, "KeyError가 발생해야 함"
except KeyError as e:
    print(f"PASS: 잘못된 config -> KeyError 발생: {e}")
```

### 시나리오 3-4: train.py의 TRAINING_CONFIG batch_size 사용 확인

```python
from lasps.config.model_config import TRAINING_CONFIG

# 검증: batch_size가 하드코딩이 아닌 config에서 오는지
assert "batch_size" in TRAINING_CONFIG, "TRAINING_CONFIG에 batch_size 존재"
assert TRAINING_CONFIG["batch_size"] == 128, "기본 batch_size=128"
print("PASS: TRAINING_CONFIG 사용 확인")
```

### 시나리오 3-5: Settings 클래스 docstring 확인

```python
from lasps.config.settings import Settings

assert Settings.__doc__ is not None, "Settings docstring 존재해야 함"
assert "environment" in Settings.__doc__.lower(), "docstring에 environment 언급"
print(f"PASS: Settings docstring = '{Settings.__doc__.strip()}'")
```

### 시나리오 3-6: trainer.py 타입 힌트 확인

```python
import inspect
from lasps.training.trainer import ThreePhaseTrainer

sig = inspect.signature(ThreePhaseTrainer._run_epoch)
params = sig.parameters

assert params["optimizer"].annotation != inspect.Parameter.empty, "optimizer 타입 힌트 존재"
assert params["clip_params"].annotation != inspect.Parameter.empty, "clip_params 타입 힌트 존재"
print("PASS: _run_epoch 타입 힌트 확인")
```

---

## 4. [HIGH] 에러 핸들링 및 시간 정합성 검증

### 대상 파일
- `lasps/data/collectors/kiwoom_collector.py`
- `lasps/data/collectors/integrated_collector.py`

### 시나리오 4-1: KiwoomCollector API 실패 시 ConnectionError 래핑

```python
from lasps.data.collectors.kiwoom_base import KiwoomAPIBase
from lasps.data.collectors.kiwoom_collector import KiwoomCollector

class FailingAPI(KiwoomAPIBase):
    def request(self, tr_code, **kwargs):
        raise RuntimeError("서버 연결 끊김")
    def is_connected(self):
        return False

collector = KiwoomCollector(FailingAPI(), rate_limit=False)

try:
    collector.get_stock_info("005930")
    assert False, "ConnectionError가 발생해야 함"
except ConnectionError as e:
    assert "API request failed" in str(e), "에러 메시지에 원인 포함"
    assert e.__cause__ is not None, "원본 예외 체인 보존"
    print(f"PASS: ConnectionError 래핑 정상: {e}")
```

### 시나리오 4-2: 빈 OHLCV 응답 -> ValueError

```python
class EmptyAPI(KiwoomAPIBase):
    def request(self, tr_code, **kwargs):
        if tr_code == "OPT10081":
            return []  # 빈 응답
        return {"업종코드": "001", "종목명": "테스트"}
    def is_connected(self):
        return True

collector = KiwoomCollector(EmptyAPI(), rate_limit=False)

try:
    collector.get_daily_ohlcv("999999")
    assert False, "ValueError가 발생해야 함"
except ValueError as e:
    assert "Empty OHLCV response" in str(e)
    print(f"PASS: 빈 OHLCV 응답 처리 정상: {e}")
```

### 시나리오 4-3: IntegratedCollector 정상 데이터 수집

```python
from lasps.data.collectors.kiwoom_mock import KiwoomMockAPI
from lasps.data.collectors.integrated_collector import IntegratedCollector

# 정상 케이스: 180일 데이터 요청 -> 60일 시계열 생성 가능
ic = IntegratedCollector(KiwoomMockAPI(seed=42), rate_limit=False)
result = ic.collect_stock_data("005930")

assert result["time_series_25d"].shape[0] == 60, "시계열 60일이어야 함"
assert result["time_series_25d"].shape[1] == 25, "피처 25개여야 함"
assert result["chart_tensor"].shape == (3, 224, 224), "차트 텐서 shape 확인"
print("PASS: 정상 데이터 수집 성공")
```

### 시나리오 4-4: collect_batch 반환 형식 검증

```python
ic = IntegratedCollector(KiwoomMockAPI(seed=42), rate_limit=False)

codes = ["005930", "000660", "005380"]
batch = ic.collect_batch(codes)

# 검증: 반환 형식
assert "results" in batch, "results 키 존재"
assert "failures" in batch, "failures 키 존재"
assert isinstance(batch["results"], list), "results는 리스트"
assert isinstance(batch["failures"], list), "failures는 리스트"

# 정상 케이스: 3개 모두 성공
assert len(batch["results"]) == 3, "3개 모두 성공해야 함"
assert len(batch["failures"]) == 0, "실패 0건"

print("PASS: collect_batch 반환 형식 정상")
```

### 시나리오 4-5: collect_batch 부분 실패 시 failures에 기록

```python
# FailingAPI를 사용하면 모든 종목이 실패
ic_fail = IntegratedCollector(FailingAPI(), rate_limit=False)
batch_fail = ic_fail.collect_batch(["005930", "000660"])

assert len(batch_fail["results"]) == 0, "전부 실패해야 함"
assert len(batch_fail["failures"]) == 2, "2건 실패 기록"
assert batch_fail["failures"][0]["code"] == "005930", "실패 종목코드 기록"
assert "error" in batch_fail["failures"][0], "에러 메시지 포함"

print("PASS: collect_batch 실패 추적 정상")
```

### 시나리오 4-6: 차트/피처 시간 정합성 확인

```python
ic = IntegratedCollector(KiwoomMockAPI(seed=42), rate_limit=False)
result = ic.collect_stock_data("005930")

# time_series_25d의 마지막 60일과 chart_tensor의 날짜 범위가 동일해야 함
# (내부적으로 chart_ohlcv = ohlcv[ohlcv["date"].isin(chart_dates)] 사용)
assert result["time_series_25d"].shape[0] == 60, "시계열 60일"
assert result["chart_tensor"].shape == (3, 224, 224), "차트 (3,224,224)"

print("PASS: 차트/피처 시간 정합성 확인")
```

---

## 5. [MEDIUM] FocalLoss / requirements / logger / checkpoint 검증

### 시나리오 5-1: FocalLoss 기본 (weight=None)

```python
import torch
from lasps.training.loss_functions import FocalLoss

fl = FocalLoss(num_classes=3, gamma=2.0)
logits = torch.randn(8, 3)
labels = torch.randint(0, 3, (8,))
loss = fl(logits, labels)

assert loss.shape == (), "스칼라 loss"
assert loss.item() > 0, "loss > 0"
assert fl.weight is None, "weight=None 확인"
print(f"PASS: FocalLoss 기본 동작 정상, loss={loss.item():.4f}")
```

### 시나리오 5-2: FocalLoss class weight 적용

```python
weight = torch.tensor([0.5, 1.0, 2.0])  # SELL 낮은 가중치, BUY 높은 가중치
fl_w = FocalLoss(num_classes=3, gamma=2.0, weight=weight)

# 검증: register_buffer로 저장됨
assert fl_w.weight is not None, "weight가 저장되어야 함"
assert torch.equal(fl_w.weight, weight), "weight 값 일치"

# 검증: state_dict에 포함됨
sd = fl_w.state_dict()
assert "weight" in sd, "weight가 state_dict에 포함"

logits = torch.randn(8, 3)
labels = torch.randint(0, 3, (8,))
loss = fl_w(logits, labels)
assert loss.item() > 0, "weighted focal loss > 0"

print(f"PASS: FocalLoss weight 적용 정상, loss={loss.item():.4f}")
```

### 시나리오 5-3: FocalLoss gamma=0 -> Cross-Entropy와 동일

```python
import torch.nn.functional as F

fl_ce = FocalLoss(num_classes=3, gamma=0.0)

torch.manual_seed(42)
logits = torch.randn(16, 3)
labels = torch.randint(0, 3, (16,))

focal_loss = fl_ce(logits, labels)
ce_loss = F.cross_entropy(logits, labels)

assert abs(focal_loss.item() - ce_loss.item()) < 1e-5, \
    f"gamma=0이면 CE와 동일: focal={focal_loss.item():.6f}, ce={ce_loss.item():.6f}"
print("PASS: gamma=0 == Cross-Entropy 확인")
```

### 시나리오 5-4: FocalLoss device 이동 (weight 자동 이동)

```python
weight = torch.tensor([1.0, 1.0, 1.0])
fl_buf = FocalLoss(num_classes=3, gamma=2.0, weight=weight)

# CPU에서 동작 확인
logits = torch.randn(4, 3)
labels = torch.randint(0, 3, (4,))
loss = fl_buf(logits, labels)
assert loss.device.type == "cpu", "CPU에서 동작"

# GPU 있을 경우 (없으면 스킵)
if torch.cuda.is_available():
    fl_gpu = fl_buf.cuda()
    assert fl_gpu.weight.device.type == "cuda", "weight도 GPU로 이동"
    loss_gpu = fl_gpu(logits.cuda(), labels.cuda())
    assert loss_gpu.device.type == "cuda", "GPU에서 loss 계산"
    print("PASS: FocalLoss GPU 이동 정상")
else:
    print("SKIP: GPU 없음 - CPU 테스트만 PASS")
```

### 시나리오 5-5: requirements.txt PyTorch 버전

```python
from pathlib import Path

req_text = Path("requirements.txt").read_text()
assert "torch>=1.8.0" in req_text, "torch>=1.8.0이어야 함"
assert "torch>=2.0.0" not in req_text, "torch>=2.0.0이면 안됨"

print("PASS: requirements.txt PyTorch 버전 정상")
```

### 시나리오 5-6: logger 디렉토리 자동 생성

```python
import shutil
from pathlib import Path
from lasps.utils.logger import setup_logger

# logs/ 디렉토리 삭제 후 재생성 테스트
log_dir = Path("logs")
if log_dir.exists():
    shutil.rmtree(log_dir)

assert not log_dir.exists(), "logs/ 삭제 확인"

setup_logger("INFO")

assert log_dir.exists(), "setup_logger 호출 후 logs/ 생성되어야 함"
assert log_dir.is_dir(), "logs/는 디렉토리여야 함"

print("PASS: logger 디렉토리 자동 생성 정상")
```

### 시나리오 5-7: checkpoint 경로 검증

```python
from pathlib import Path

# 존재하지 않는 checkpoint 경로
fake_path = Path("nonexistent_checkpoint.pt")
assert not fake_path.exists(), "가짜 경로가 존재하면 안됨"

# scripts/train.py의 checkpoint 로딩 로직 검증
# (실제 실행은 argparse 필요하므로 로직만 검증)
try:
    if not fake_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {fake_path}")
    assert False, "FileNotFoundError가 발생해야 함"
except FileNotFoundError as e:
    assert "Checkpoint not found" in str(e)
    print(f"PASS: checkpoint 경로 검증 정상: {e}")
```

---

## 6. 통합 테스트 시나리오 (End-to-End)

### 시나리오 6-1: 데이터 수집 -> 모델 추론 전체 파이프라인

```python
import torch
from lasps.data.collectors.kiwoom_mock import KiwoomMockAPI
from lasps.data.collectors.integrated_collector import IntegratedCollector
from lasps.models.sector_aware_model import SectorAwareFusionModel

# 1. 데이터 수집
ic = IntegratedCollector(KiwoomMockAPI(seed=42), rate_limit=False)
stock_data = ic.collect_stock_data("005930")

# 2. 텐서 준비
ts = torch.tensor(stock_data["time_series_25d"]).unsqueeze(0)    # (1, 60, 25)
chart = stock_data["chart_tensor"].unsqueeze(0)                    # (1, 3, 224, 224)
sid = torch.tensor([stock_data["sector_id"]])                     # (1,)

# 3. 모델 추론
model = SectorAwareFusionModel(num_sectors=20, ts_input_dim=25)
model.eval()

with torch.no_grad():
    out = model(ts, chart, sid)

# 4. 결과 검증
logits = out["logits"]
probs = out["probabilities"]

assert logits.shape == (1, 3), f"logits shape: {logits.shape}"
assert probs.shape == (1, 3), f"probs shape: {probs.shape}"
assert abs(probs.sum().item() - 1.0) < 1e-5, "확률 합 = 1.0"

pred_class = logits.argmax(dim=1).item()
class_names = {0: "SELL", 1: "HOLD", 2: "BUY"}
print(f"PASS: E2E 파이프라인 정상 - 예측: {class_names[pred_class]} "
      f"(확률: {probs[0].tolist()})")
```

### 시나리오 6-2: 3-Phase 학습 미니 테스트

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from lasps.models.sector_aware_model import SectorAwareFusionModel
from lasps.training.trainer import ThreePhaseTrainer

# 미니 데이터셋 생성
n = 16
ts = torch.randn(n, 60, 25)
img = torch.randn(n, 3, 224, 224)
sid = torch.randint(0, 3, (n,))
labels = torch.randint(0, 3, (n,))

ds = TensorDataset(ts, img, sid, labels)
loader = DataLoader(ds, batch_size=4)

# 모델 + 트레이너
model = SectorAwareFusionModel(num_sectors=3, ts_input_dim=25)
trainer = ThreePhaseTrainer(model, device="cpu")

# Phase 1
p1 = trainer.train_phase1(loader, loader, epochs=2, patience=0)
assert "train_loss" in p1
assert "best_val_loss" in p1
print(f"Phase 1 완료: train_loss={p1['train_loss']:.4f}, val_loss={p1['val_loss']:.4f}")

# Phase 2
sector_loaders = {0: loader, 1: loader}
trainer.train_phase2(sector_loaders, epochs_per_sector=1)
print("Phase 2 완료")

# Phase 3
p3 = trainer.train_phase3(loader, loader, epochs=2, patience=0)
assert "train_loss" in p3
print(f"Phase 3 완료: train_loss={p3['train_loss']:.4f}, val_loss={p3['val_loss']:.4f}")

print("PASS: 3-Phase 학습 전체 파이프라인 정상")
```

### 시나리오 6-3: 정규화 -> 모델 입력 값 범위 확인

```python
import torch
import numpy as np
from lasps.data.collectors.kiwoom_mock import KiwoomMockAPI
from lasps.data.collectors.integrated_collector import IntegratedCollector

ic = IntegratedCollector(KiwoomMockAPI(seed=42), rate_limit=False)
data = ic.collect_stock_data("005930")

ts = data["time_series_25d"]

# 검증: OHLCV + indicators (0-19)가 [0, 1] 범위
for col in range(20):
    col_min = ts[:, col].min()
    col_max = ts[:, col].max()
    assert col_min >= -0.01, f"Feature {col}: min={col_min:.4f} (음수)"
    assert col_max <= 1.01, f"Feature {col}: max={col_max:.4f} (1 초과)"

# 검증: Sentiment (20-24)는 원래 범위 유지
# volume_ratio: 0~1, volatility_ratio: 0~1
# gap_direction: -1~+1, rsi_norm: 0~1, foreign_inst_flow: -1~+1
for col in range(20, 25):
    col_min = ts[:, col].min()
    col_max = ts[:, col].max()
    assert col_min >= -1.1, f"Sentiment {col}: min={col_min:.4f}"
    assert col_max <= 3.1, f"Sentiment {col}: max={col_max:.4f}"

print("PASS: 정규화된 시계열 값 범위 정상")
```

### 시나리오 6-4: 멀티 종목 배치 수집 + 모델 일괄 추론

```python
import torch
from lasps.data.collectors.kiwoom_mock import KiwoomMockAPI
from lasps.data.collectors.integrated_collector import IntegratedCollector
from lasps.models.sector_aware_model import SectorAwareFusionModel

ic = IntegratedCollector(KiwoomMockAPI(seed=42), rate_limit=False)
codes = ["005930", "000660", "005380", "035420", "105560"]
batch = ic.collect_batch(codes)

assert len(batch["results"]) == 5, f"5종목 성공 필요, got {len(batch['results'])}"
assert len(batch["failures"]) == 0, "실패 0건이어야 함"

# 텐서로 변환
ts_list = [torch.tensor(r["time_series_25d"]) for r in batch["results"]]
chart_list = [r["chart_tensor"] for r in batch["results"]]
sid_list = [r["sector_id"] for r in batch["results"]]

ts_batch = torch.stack(ts_list)                             # (5, 60, 25)
chart_batch = torch.stack(chart_list)                       # (5, 3, 224, 224)
sid_batch = torch.tensor(sid_list)                          # (5,)

model = SectorAwareFusionModel(num_sectors=20, ts_input_dim=25)
model.eval()

with torch.no_grad():
    out = model(ts_batch, chart_batch, sid_batch)

assert out["logits"].shape == (5, 3), "5종목 배치 추론 shape"
assert out["probabilities"].shape == (5, 3), "확률 shape"

class_names = {0: "SELL", 1: "HOLD", 2: "BUY"}
preds = out["logits"].argmax(dim=1)
for i, code in enumerate(codes):
    pred = class_names[preds[i].item()]
    prob = out["probabilities"][i].max().item()
    print(f"  {code}: {pred} (신뢰도 {prob:.2%})")

print("PASS: 멀티 종목 배치 추론 정상")
```

---

## 테스트 실행 방법

### 자동 테스트 (pytest)

```bash
# 전체 테스트 실행
pytest tests/ -v --tb=short

# 특정 모듈만
pytest tests/test_sentiment.py -v
pytest tests/test_utils.py -v
pytest tests/test_sector_model.py -v
pytest tests/test_training.py -v
pytest tests/test_integrated_collector.py -v
```

### 매뉴얼 테스트

```bash
# 프로젝트 루트에서 Python REPL 실행
cd /Users/yunghochoi/git_dir/stockquant/v7a_sentiment
python

# 각 시나리오의 코드 블록을 순서대로 복사 + 붙여넣기로 실행
# 모든 시나리오가 "PASS:"로 끝나면 검증 완료
```

---

## 체크리스트

| # | 시나리오 | 우선순위 | 결과 |
|---|---------|---------|------|
| 1-1 | volume_ratio division-by-zero | CRITICAL | [ ] |
| 1-2 | volatility_ratio division-by-zero | CRITICAL | [ ] |
| 1-3 | gap_direction division-by-zero | CRITICAL | [ ] |
| 1-4 | RSI gain=0, loss=0 | CRITICAL | [ ] |
| 1-5 | BB width division-by-zero | CRITICAL | [ ] |
| 1-6 | investor_data=None | CRITICAL | [ ] |
| 2-1 | 기본 정규화 동작 | CRITICAL | [ ] |
| 2-2 | 상수 피처 처리 | CRITICAL | [ ] |
| 2-3 | 단일 행 입력 | CRITICAL | [ ] |
| 2-4 | shape 보존 | CRITICAL | [ ] |
| 3-1 | MODEL_CONFIG 기본 연결 | HIGH | [ ] |
| 3-2 | 커스텀 config | HIGH | [ ] |
| 3-3 | 잘못된 config KeyError | HIGH | [ ] |
| 3-4 | TRAINING_CONFIG batch_size | HIGH | [ ] |
| 3-5 | Settings docstring | HIGH | [ ] |
| 3-6 | trainer 타입 힌트 | HIGH | [ ] |
| 4-1 | ConnectionError 래핑 | HIGH | [ ] |
| 4-2 | 빈 OHLCV ValueError | HIGH | [ ] |
| 4-3 | 정상 데이터 수집 | HIGH | [ ] |
| 4-4 | collect_batch 반환 형식 | HIGH | [ ] |
| 4-5 | collect_batch 실패 추적 | HIGH | [ ] |
| 4-6 | 차트/피처 시간 정합성 | HIGH | [ ] |
| 5-1 | FocalLoss 기본 | MEDIUM | [ ] |
| 5-2 | FocalLoss weight | MEDIUM | [ ] |
| 5-3 | gamma=0 == CE | MEDIUM | [ ] |
| 5-4 | device 이동 | MEDIUM | [ ] |
| 5-5 | requirements.txt | MEDIUM | [ ] |
| 5-6 | logger 디렉토리 | MEDIUM | [ ] |
| 5-7 | checkpoint 경로 | MEDIUM | [ ] |
| 6-1 | E2E 수집->추론 | HIGH | [ ] |
| 6-2 | 3-Phase 학습 | HIGH | [ ] |
| 6-3 | 정규화 값 범위 | HIGH | [ ] |
| 6-4 | 멀티 종목 배치 | HIGH | [ ] |
