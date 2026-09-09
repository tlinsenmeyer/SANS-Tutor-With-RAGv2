# src/utils/logger.py
import logging
import uuid
from pathlib import Path
from logging.handlers import RotatingFileHandler

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Colorized console output
# ---------------------------------------------------------------------------
class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[41m",  # Red background
    }
    RESET = "\033[0m"

    def format(self, record):
        level_color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{level_color}{record.levelname}{self.RESET}"
        return super().format(record)


# ---------------------------------------------------------------------------
# Create a correlation ID for each ingestion run
# Prefect will pass this into tasks
# ---------------------------------------------------------------------------
def new_correlation_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Main logger factory
# ---------------------------------------------------------------------------
def get_pipeline_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger  # Avoid duplicate handlers

    # -----------------------------------------------------------------------
    # Console Handler (colorized)
    # -----------------------------------------------------------------------
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = ColorFormatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    # -----------------------------------------------------------------------
    # Rotating File Handler (pipeline.log)
    # -----------------------------------------------------------------------
    fh = RotatingFileHandler(
        LOG_DIR / "pipeline.log",
        maxBytes=5_000_000,  # 5 MB
        backupCount=5,
        encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # -----------------------------------------------------------------------
    # Error-only log file
    # -----------------------------------------------------------------------
    eh = RotatingFileHandler(
        LOG_DIR / "errors.log",
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8"
    )
    eh.setLevel(logging.ERROR)
    eh.setFormatter(fh_formatter)
    logger.addHandler(eh)

    return logger
