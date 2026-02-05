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
