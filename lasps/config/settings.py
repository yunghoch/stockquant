import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings:
    """Application settings loaded from environment variables."""

    KIWOOM_ACCOUNT: str = os.getenv("KIWOOM_ACCOUNT", "")
    DART_API_KEY: str = os.getenv("DART_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    MODEL_PATH: Path = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "checkpoints")))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

    # SQLite
    DATABASE_PATH: Path = Path(
        os.getenv("DATABASE_PATH", str(BASE_DIR / "data" / "lasps.db"))
    )

    # Data Quality Filtering
    MIN_DAILY_PRICE_ROWS: int = 1500  # 최소 6년치 데이터 (학습 포함 기준)

    # Class Weights for Imbalanced Classification
    # SELL=24.4%, HOLD=53.6%, BUY=21.9% → 역수 정규화
    # SELL: 1/0.244 = 4.10 → normalized 2.05
    # HOLD: 1/0.536 = 1.87 → normalized 0.93
    # BUY:  1/0.219 = 4.57 → normalized 2.28
    CLASS_WEIGHTS: tuple = (2.05, 0.93, 2.28)  # (SELL, HOLD, BUY)

    @property
    def DATABASE_URL(self) -> str:
        return f"sqlite:///{self.DATABASE_PATH}"


settings = Settings()
