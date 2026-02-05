import sys
from pathlib import Path
from loguru import logger


def setup_logger(level: str = "INFO") -> None:
    """Configure loguru logger with stderr and file outputs.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

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
        str(log_dir / "lasps_{time:YYYY-MM-DD}.log"),
        level=level,
        rotation="1 day",
        retention="30 days",
    )
