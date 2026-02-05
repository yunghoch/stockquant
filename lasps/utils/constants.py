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
