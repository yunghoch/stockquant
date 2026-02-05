# LASPS v7a 바이브코딩 마스터 가이드
# PRD 기준 통합본 (4개 문서 → 1개)

---

## 1. 핵심 상수 (Quick Reference)

```python
# PRD v7a 기준 핵심 상수
TIME_SERIES_SHAPE = (60, 25)      # OHLCV(5) + 지표(15) + 감성(5)
CHART_IMAGE_SHAPE = (3, 224, 224)
NUM_SECTORS = 20                   # 전체 20개 업종
NUM_CLASSES = 3                    # SELL(0), HOLD(1), BUY(2)
PREDICTION_HORIZON = 5             # 5일 후 수익률
LABEL_THRESHOLD = 0.03             # ±3%
MONTHLY_COST = 30                  # USD (50% 절감)
```

---

## 2. 프로젝트 구조

```
lasps/
├── config/
│   ├── settings.py           # 환경설정, API 키
│   ├── model_config.py       # 모델 하이퍼파라미터
│   ├── sector_config.py      # ⭐ 섹터 매핑 (20개)
│   └── tr_config.py          # 키움 TR 설정
├── data/
│   ├── collectors/
│   │   ├── kiwoom_collector.py   # OPT10001/10081/10059/10014
│   │   ├── sector_collector.py   # 업종 정보 수집
│   │   ├── dart_collector.py     # 부채비율
│   │   └── integrated_collector.py
│   ├── processors/
│   │   ├── technical_indicators.py  # 15개 지표
│   │   ├── chart_generator.py       # 캔들차트 (224x224)
│   │   └── market_sentiment.py      # ⭐ 5차원 감성
│   └── datasets/
│       └── stock_dataset.py   # (60,25) + sector_id
├── models/
│   ├── linear_transformer.py  # Branch 1: input_dim=25
│   ├── chart_cnn.py           # Branch 2: 이미지
│   ├── sector_aware_model.py  # ⭐ 20 Sector Heads
│   └── qvm_screener.py        # QVM 팩터
├── training/
│   ├── trainer.py             # ⭐ ThreePhaseTrainer
│   ├── backbone_trainer.py    # Phase 1
│   ├── sector_head_trainer.py # Phase 2
│   └── loss_functions.py
├── services/
│   ├── predictor.py           # Sector-Aware 예측
│   ├── llm_analyst.py         # Top 10 Claude 분석
│   └── integrated_predictor.py
├── api/
│   └── main.py                # FastAPI
├── utils/
│   ├── logger.py, helpers.py, constants.py, metrics.py
├── tests/
│   ├── test_collectors.py, test_models.py
│   ├── test_sentiment.py      # ⭐ 시장 감성 테스트
│   └── test_sector_model.py   # ⭐ 섹터 모델 테스트
└── scripts/
    ├── train.py, daily_batch.py, historical_data.py
```

---

## 3. 섹터 코드 매핑 (PRD 기준 - 20개 전체)

```python
# config/sector_config.py

SECTOR_CODES = {
    # 코드: (인덱스, 업종명, 종목수 추정)
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
    """키움 업종코드 → sector_id 변환"""
    if sector_code in SECTOR_CODES:
        return SECTOR_CODES[sector_code][0]
    return -1  # Unknown

def get_sector_name(sector_id: int) -> str:
    """sector_id → 업종명 변환"""
    for code, (sid, name, _) in SECTOR_CODES.items():
        if sid == sector_id:
            return name
    return "Unknown"

def get_sector_info(stock_code: str, kiwoom) -> dict:
    """OPT10001에서 업종 정보 추출"""
    response = kiwoom.request("OPT10001", 종목코드=stock_code)
    sector_code = response["업종코드"]
    return {
        "code": stock_code,
        "name": response["종목명"],
        "sector_code": sector_code,
        "sector_name": response["업종명"],
        "sector_id": get_sector_id(sector_code)
    }
```

---

## 4. 시장 감성 5차원 (PRD 기준)

```python
# data/processors/market_sentiment.py

import numpy as np
import pandas as pd
from typing import Optional

class MarketSentimentCalculator:
    """시장 기반 감성 5차원 계산기 (PRD 기준)
    
    모든 데이터는 키움 OpenAPI에서 수집
    Claude API 불필요 → 비용 50% 절감
    """
    
    LOOKBACK = 20
    
    DEFAULT_VALUES = {
        "volume_ratio": 0.33,
        "volatility_ratio": 0.33,
        "gap_direction": 0.0,
        "rsi_norm": 0.5,
        "foreign_inst_flow": 0.0
    }
    
    def calculate(
        self, 
        ohlcv_df: pd.DataFrame, 
        investor_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """전체 시장 감성 지표 계산
        
        Args:
            ohlcv_df: 일봉 데이터 (date, open, high, low, close, volume)
            investor_df: 투자자별 데이터 (date, foreign_net, inst_net)
            
        Returns:
            DataFrame with 5 sentiment features
        """
        df = ohlcv_df.copy()
        
        # 1. volume_ratio: clip(0, 3) / 3 → 0~1
        df['volume_ma20'] = df['volume'].rolling(self.LOOKBACK).mean()
        df['volume_ratio'] = (df['volume'] / df['volume_ma20']).clip(0, 3) / 3
        
        # 2. volatility_ratio: true_range / atr_20 → clip(0, 3) / 3 → 0~1
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr_20'] = df['true_range'].rolling(self.LOOKBACK).mean()
        df['volatility_ratio'] = (df['true_range'] / df['atr_20']).clip(0, 3) / 3
        
        # 3. gap_direction: clip(-0.1, 0.1) * 10 → -1~+1  ⚠️ PRD 기준 ±10%
        df['prev_close'] = df['close'].shift(1)
        df['gap_pct'] = (df['open'] - df['prev_close']) / df['prev_close']
        df['gap_direction'] = df['gap_pct'].clip(-0.1, 0.1) * 10
        
        # 4. rsi_norm: RSI(14) / 100 → 0~1
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_norm'] = df['rsi'] / 100
        
        # 5. foreign_inst_flow: sign * log10 방식 → -1~+1  ⚠️ PRD 기준
        if investor_df is not None and len(investor_df) > 0:
            merged = df.merge(investor_df, on='date', how='left')
            merged['total_flow'] = merged['foreign_net'].fillna(0) + merged['inst_net'].fillna(0)
            merged['flow_sign'] = np.sign(merged['total_flow'])
            merged['flow_magnitude'] = np.minimum(1, np.log10(abs(merged['total_flow']) + 1) / 8)
            merged['foreign_inst_flow'] = merged['flow_sign'] * merged['flow_magnitude']
            df = merged
        else:
            df['foreign_inst_flow'] = 0.0
        
        # 결측치 처리
        result = df[['date', 'volume_ratio', 'volatility_ratio', 
                     'gap_direction', 'rsi_norm', 'foreign_inst_flow']].copy()
        for col, default in self.DEFAULT_VALUES.items():
            result[col] = result[col].fillna(default)
        
        return result
    
    def get_feature_names(self) -> list:
        return ["volume_ratio", "volatility_ratio", "gap_direction", 
                "rsi_norm", "foreign_inst_flow"]
```

---

## 5. 시계열 피처 구성 (25차원)

```python
# 일별 피처 벡터 구성

FEATURE_STRUCTURE = {
    # OHLCV (5개) - 인덱스 0~4
    "ohlcv": {
        "indices": (0, 5),
        "features": ["open", "high", "low", "close", "volume"],
        "normalization": "min-max per stock"
    },
    
    # 기술지표 (15개) - 인덱스 5~19
    "indicators": {
        "indices": (5, 20),
        "features": [
            "ma5", "ma20", "ma60", "ma120",           # 추세 4개
            "rsi",                                     # 모멘텀 1개
            "macd", "macd_signal", "macd_hist",       # MACD 3개
            "bb_upper", "bb_middle", "bb_lower", "bb_width",  # 볼린저 4개
            "atr",                                     # 변동성 1개
            "obv", "volume_ma20"                       # 거래량 2개
        ]
    },
    
    # 시장 감성 (5개) - 인덱스 20~24
    "market_sentiment": {
        "indices": (20, 25),
        "features": [
            "volume_ratio",       # 0~1 (거래량/20일평균)
            "volatility_ratio",   # 0~1 (변동성/20일평균)
            "gap_direction",      # -1~+1 (갭 방향, ±10%)
            "rsi_norm",           # 0~1 (RSI/100)
            "foreign_inst_flow"   # -1~+1 (외국인+기관, log10 방식)
        ]
    }
}

# 별도 입력: sector_id (0~19) → 해당 섹터 Head 선택용
```

---

## 6. 모델 설정

```python
# config/model_config.py

MODEL_CONFIG = {
    "num_sectors": 20,
    
    "linear_transformer": {
        "input_dim": 25,       # OHLCV(5) + 지표(15) + 감성(5)
        "hidden_dim": 128,
        "num_layers": 4,
        "num_heads": 4,
        "dropout": 0.2,
        "sequence_length": 60
    },
    
    "cnn": {
        "input_channels": 3,
        "conv_channels": [32, 64, 128, 256],
        "output_dim": 128,
        "dropout": 0.3
    },
    
    "fusion": {
        "shared_dim": 128,          # Concat(128+128)=256 → 128
        "sector_head_hidden": 64,   # 128 → 64 → 3
        "num_classes": 3,
        "dropout": 0.3
    }
}

MARKET_SENTIMENT_CONFIG = {
    "lookback_period": 20,
    "features": [
        "volume_ratio", "volatility_ratio", 
        "gap_direction", "rsi_norm", "foreign_inst_flow"
    ],
    "default_values": {
        "volume_ratio": 0.33,
        "volatility_ratio": 0.33,
        "gap_direction": 0.0,
        "rsi_norm": 0.5,
        "foreign_inst_flow": 0.0
    }
}

# 아키텍처 요약:
# Transformer(60,25)→128 + CNN(3,224,224)→128 
#   → Concat(256) → SharedFC(128) → SectorHead[sector_id](64→3)
```

---

## 7. Sector-Aware 모델 구현

```python
# models/sector_aware_model.py

import torch
import torch.nn as nn

class SectorAwareFusionModel(nn.Module):
    """섹터 인식 2-Branch Fusion 모델 (PRD 기준)
    
    공통 Backbone으로 범용 패턴 학습 +
    섹터별 Head로 업종 특성 반영
    """
    
    def __init__(self, num_sectors=20, ts_input_dim=25):
        super().__init__()
        
        # ========== 공통 Backbone ==========
        self.ts_encoder = LinearTransformerEncoder(
            input_dim=ts_input_dim,
            hidden_dim=128,
            num_layers=4,
            num_heads=4,
            dropout=0.2
        )
        
        self.cnn = ChartCNN(
            conv_channels=[32, 64, 128, 256],
            output_dim=128
        )
        
        self.shared_fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # ========== 섹터별 Head (20개) ==========
        self.sector_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 3)  # SELL, HOLD, BUY
            ) for _ in range(num_sectors)
        ])
        
        self.num_sectors = num_sectors
    
    def forward(self, time_series, chart_image, sector_id):
        """
        Args:
            time_series: (batch, 60, 25)
            chart_image: (batch, 3, 224, 224)
            sector_id: (batch,) - 0~19
        """
        # 공통 Backbone
        ts_feat = self.ts_encoder(time_series)      # (B, 128)
        img_feat = self.cnn(chart_image)            # (B, 128)
        fused = torch.cat([ts_feat, img_feat], dim=1)  # (B, 256)
        shared_feat = self.shared_fusion(fused)     # (B, 128)
        
        # 섹터별 Head 선택
        batch_size = time_series.size(0)
        logits = torch.zeros(batch_size, 3, device=time_series.device)
        
        for i in range(batch_size):
            sid = sector_id[i].item()
            logits[i] = self.sector_heads[sid](shared_feat[i:i+1]).squeeze(0)
        
        return {
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=1),
            "shared_features": shared_feat
        }
    
    def forward_efficient(self, time_series, chart_image, sector_id):
        """배치 내 동일 섹터 그룹핑으로 효율적 처리"""
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
    
    # ========== 3-Phase 학습용 메서드 ==========
    def freeze_backbone(self):
        """Phase 2: Backbone 동결"""
        for param in self.ts_encoder.parameters():
            param.requires_grad = False
        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.shared_fusion.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Phase 3: Backbone 해제"""
        for param in self.ts_encoder.parameters():
            param.requires_grad = True
        for param in self.cnn.parameters():
            param.requires_grad = True
        for param in self.shared_fusion.parameters():
            param.requires_grad = True
    
    def get_sector_head_params(self, sector_id: int):
        """특정 섹터 Head의 파라미터만 반환"""
        return self.sector_heads[sector_id].parameters()
```

---

## 8. 3-Phase 학습 전략

```python
# training/trainer.py

TRAINING_CONFIG = {
    "batch_size": 128,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "gradient_clip": 1.0,
}

THREE_PHASE_CONFIG = {
    # Phase 1: Backbone 학습
    "phase1_backbone": {
        "epochs": 30,
        "lr": 1e-4,
        "scheduler": "cosine",
        "warmup_epochs": 5,
        "trainable": ["ts_encoder", "cnn", "shared_fusion", "all_heads"],
        "data": "전체 610만 샘플",
        "목표": "범용 패턴 학습"
    },
    
    # Phase 2: Sector Heads 학습
    "phase2_sector_heads": {
        "epochs_per_sector": 10,
        "lr": 5e-4,
        "scheduler": "step",
        "step_size": 5,
        "gamma": 0.5,
        "trainable": ["sector_heads[sector_id]"],
        "frozen": ["ts_encoder", "cnn", "shared_fusion"],
        "data": "섹터별 분리",
        "min_samples": 10000,
        "목표": "섹터 특성 미세 조정"
    },
    
    # Phase 3: End-to-End 미세 조정
    "phase3_finetune": {
        "epochs": 5,
        "lr": 1e-5,
        "scheduler": "cosine",
        "trainable": "all",
        "data": "전체",
        "목표": "전체 최적화"
    }
}

class ThreePhaseTrainer:
    """3-Phase 학습 관리자"""
    
    def train_phase1(self, model, train_loader, val_loader):
        """Backbone 학습"""
        # 전체 파라미터 학습
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        # ... 30 epochs
    
    def train_phase2(self, model, sector_data_loaders):
        """Sector Heads 학습"""
        model.freeze_backbone()
        
        for sector_id in range(model.num_sectors):
            if sector_id not in sector_data_loaders:
                continue
            
            optimizer = torch.optim.AdamW(
                model.get_sector_head_params(sector_id), 
                lr=5e-4
            )
            # ... 10 epochs per sector
    
    def train_phase3(self, model, train_loader, val_loader):
        """End-to-End 미세 조정"""
        model.unfreeze_backbone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        # ... 5 epochs

# 예상 학습 시간
TRAINING_TIME = {
    "RTX_3090": "~15시간",
    "RTX_4090": "~9시간",
    "A100": "~5시간"
}
```

---

## 9. 기술지표 15개

```python
# data/processors/technical_indicators.py

INDICATOR_PARAMS = {
    # 추세 (4개) - index 5~8
    "MA5": {"window": 5},
    "MA20": {"window": 20},
    "MA60": {"window": 60},
    "MA120": {"window": 120},
    
    # 모멘텀 (4개) - index 9~12
    "RSI": {"period": 14},
    "MACD": {"fast": 12, "slow": 26, "signal": 9},
    "MACD_SIGNAL": {},
    "MACD_HIST": {},
    
    # 변동성 (5개) - index 13~17
    "BB_UPPER": {"window": 20, "std": 2},
    "BB_MIDDLE": {},
    "BB_LOWER": {"window": 20, "std": 2},
    "BB_WIDTH": {},
    "ATR": {"period": 14},
    
    # 거래량 (2개) - index 18~19
    "OBV": {},
    "VOLUME_MA20": {"window": 20}
}

INDICATOR_NAMES = [
    "ma5", "ma20", "ma60", "ma120",
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_middle", "bb_lower", "bb_width", "atr",
    "obv", "volume_ma20"
]
```

---

## 10. 키움 OpenAPI TR 코드

```python
# config/tr_config.py

TR_CODES = {
    "OPT10001": {
        "name": "주식기본정보요청",
        "output": ["종목코드", "종목명", "시가총액", "PER", "PBR", "ROE", 
                   "업종코드", "업종명"],
        "interval": 0.2,
        "용도": "⭐ 섹터 정보 수집"
    },
    
    "OPT10081": {
        "name": "주식일봉차트조회요청",
        "output": ["일자", "시가", "고가", "저가", "현재가", "거래량"],
        "interval": 0.2,
        "history": "10년+",
        "용도": "OHLCV, 기술지표, 시장감성(4개)"
    },
    
    "OPT10059": {
        "name": "종목별투자자기관별요청",
        "output": ["일자", "외국인순매수", "기관계순매수", "개인순매수"],
        "interval": 0.5,
        "용도": "⭐ foreign_inst_flow 계산"
    },
    
    "OPT10014": {
        "name": "공매도추이요청",
        "output": ["일자", "공매도량", "공매도비중"],
        "interval": 0.5,
        "용도": "공매도 데이터"
    }
}
```

---

## 11. 라벨 정의

```python
LABEL_DEFINITION = {
    "prediction_horizon": 5,  # 5일 후
    
    "thresholds": {
        "SELL": "future_return <= -3%",   # label = 0
        "HOLD": "-3% < return < +3%",     # label = 1
        "BUY": "future_return >= +3%"     # label = 2
    },
    
    "code": """
    future_return = (close[t+5] - close[t]) / close[t]
    
    if future_return >= 0.03:
        label = 2  # BUY
    elif future_return <= -0.03:
        label = 0  # SELL
    else:
        label = 1  # HOLD
    """
}

DATA_SPLIT = {
    "train": "2015-01 ~ 2022-12 (8년)",
    "val": "2023-01 ~ 2023-12 (1년)",
    "test": "2024-01 ~ 2024-12 (1년)",
    "주의": "시간순 분할 필수 (미래 데이터 누출 방지)"
}
```

---

## 12. 일일 배치 프로세스 (50분)

```
[장 마감 후 16:30 시작]

1.  키움 로그인 ─────────────────────────────────  1분
2.  전체 종목 기본정보 (OPT10001) ─────────────── 15분
    └── ⭐ 업종코드, 업종명 포함
3.  DART 부채비율 수집 ────────────────────────────  5분
4.  QVM 스크리닝 → 50종목
5.  상세 데이터 수집 (50종목) ───────────────────── 15분
    ├── 일봉 OHLCV (OPT10081)
    ├── 투자자별 (OPT10059) ⭐ 외국인/기관
    └── 공매도 (OPT10014)
6.  ⭐ 시장 감성 계산 (50종목) ──────────────────────  1분
7.  기술지표 + 차트 이미지 생성 ───────────────────  5분
8.  시계열 데이터 구성 (60, 25) ───────────────────  1분
9.  ⭐ Sector-Aware 예측 ─────────────────────────  1분
    └── 종목별 sector_id로 해당 Head 선택
10. LLM 상세 분석 (Top 10) ────────────────────────  5분
11. 리포트 생성 ───────────────────────────────────  1분

총 소요 시간: 약 50분
```

---

## 13. 구현 체크리스트

### Phase 1: 데이터 파이프라인 (Week 1-2)
- [ ] config/sector_config.py (SECTOR_CODES 20개, 3자리 코드)
- [ ] data/processors/market_sentiment.py (5차원, PRD 공식)
- [ ] data/collectors/kiwoom_collector.py (OPT10001/10081/10059/10014)
- [ ] data/processors/technical_indicators.py (15개)
- [ ] data/processors/chart_generator.py (224x224, 캔들+MA+BB)
- [ ] data/datasets/stock_dataset.py ((60,25) + sector_id)

### Phase 2: 모델 개발 (Week 3-4)
- [ ] models/linear_transformer.py (input_dim=25, hidden=128)
- [ ] models/chart_cnn.py ([32,64,128,256]→128)
- [ ] models/sector_aware_model.py (20 Heads, freeze/unfreeze)
- [ ] models/qvm_screener.py

### Phase 3: 학습 (Week 5)
- [ ] training/trainer.py (ThreePhaseTrainer)

### Phase 4-5: 서비스 (Week 6-8)
- [ ] services/predictor.py (Sector-Aware)
- [ ] services/llm_analyst.py (Top 10 Claude 분석)
- [ ] scripts/daily_batch.py
- [ ] api/main.py (FastAPI)

---

## 14. 코딩 규칙

```python
# 타입 힌트 필수
def predict(
    time_series: torch.Tensor,   # (batch, 60, 25)
    chart_image: torch.Tensor,   # (batch, 3, 224, 224)
    sector_id: torch.Tensor      # (batch,) 0~19
) -> Dict[str, torch.Tensor]:
    """Google docstring 스타일
    
    Args:
        time_series: 시계열 데이터
        chart_image: 차트 이미지
        sector_id: 섹터 인덱스
        
    Returns:
        예측 결과 딕셔너리
    """
    pass

# 네이밍
class SectorAwareFusionModel:     # PascalCase (클래스)
def calculate_market_sentiment():  # snake_case (함수)
NUM_SECTORS = 20                   # UPPER_SNAKE_CASE (상수)
```

---

## 15. 의존성

```txt
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
pyqt5>=5.15.0      # 키움 API (32bit Python 필수)
anthropic>=0.18.0  # Claude API (Top 10 분석용)
ta>=0.10.0         # 기술지표
mplfinance>=0.12.0 # 캔들차트
pillow>=9.0.0      # 이미지 처리
fastapi>=0.100.0
uvicorn>=0.22.0
python-dotenv>=1.0.0
loguru>=0.7.0
tqdm>=4.65.0
pytest>=7.0.0

# 제거됨: sentence-transformers (Text Encoder 불필요)
```

---

## 16. 비용 분석

| 항목 | v6 (이전) | v7a (현재) |
|------|----------|-----------|
| 뉴스 감성 (Claude) | $30/월 | **$0** |
| LLM 상세 분석 | $30/월 | $30/월 |
| **총 비용** | **$60/월** | **$30/월** |

**50% 절감!**

---

*PRD v7a-market-sentiment-sector 기준 통합본*
*생성일: 2025-02-05*
