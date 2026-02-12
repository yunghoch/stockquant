# 시계열 구조
TIME_SERIES_LENGTH = 60
OHLCV_DIM = 5
INDICATOR_DIM = 15
SENTIMENT_DIM = 5
TEMPORAL_DIM = 3  # weekday, month, day (v2: 선형 정규화)
TOTAL_FEATURE_DIM = OHLCV_DIM + INDICATOR_DIM + SENTIMENT_DIM + TEMPORAL_DIM  # 28

# 차트 이미지
CHART_IMAGE_SIZE = 224
CHART_IMAGE_CHANNELS = 3

# 분류
NUM_CLASSES = 3
CLASS_NAMES = {0: "SELL", 1: "HOLD", 2: "BUY"}

# 라벨
PREDICTION_HORIZON = 5
LABEL_THRESHOLD = 0.03

# 섹터 (v3: 20개 → 13개 병합)
NUM_SECTORS = 13
SECTOR_NAMES = {
    0: "전기전자", 1: "금융업", 2: "서비스업", 3: "의약품",
    4: "유통업", 5: "철강금속", 6: "기계", 7: "화학",
    8: "건설업", 9: "음식료품", 10: "운수", 11: "제조업", 12: "기타",
}

# 기존 20개 섹터 → 13개 섹터 매핑
OLD_TO_NEW_SECTOR = {
    0: 0,   # 전기전자 → 전기전자
    1: 1,   # 금융업 → 금융업
    2: 2,   # 서비스업 → 서비스업
    3: 3,   # 의약품 → 의약품
    4: 10,  # 운수장비 → 운수
    5: 4,   # 유통업 → 유통업
    6: 5,   # 철강금속 → 철강금속
    7: 6,   # 기계 → 기계
    8: 7,   # 화학 → 화학
    9: 8,   # 건설업 → 건설업
    10: 12, # 섬유의복 → 기타
    11: 9,  # 음식료품 → 음식료품
    12: 12, # 비금속광물 → 기타
    13: 12, # 종이목재 → 기타
    14: 10, # 운수창고 → 운수
    15: 12, # 통신업 → 기타
    16: 12, # 전기가스 → 기타
    17: 11, # 제조업(기타) → 제조업
    18: 12, # 농업임업어업 → 기타
    19: 12, # 광업 → 기타
}

# 피처 인덱스 범위
OHLCV_INDICES = (0, 5)
INDICATOR_INDICES = (5, 20)
SENTIMENT_INDICES = (20, 25)
TEMPORAL_INDICES = (25, 28)

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
TEMPORAL_FEATURES = ["weekday", "month", "day"]  # v2: 선형 정규화
ALL_FEATURES = OHLCV_FEATURES + INDICATOR_FEATURES + SENTIMENT_FEATURES + TEMPORAL_FEATURES
