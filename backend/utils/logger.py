"""
Project-wide logger setup.

Use `get_logger(__name__)` in every module instead of print(). Logs are
written both to stdout (for development) and to a rotating file in
`artifacts/logs/`. Format includes timestamp, level, module, and message.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from backend.config import LOGS_DIR

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-30s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Track loggers we have already configured so we don't double-add handlers.
_configured: set[str] = set()


def get_logger(name: str, log_file: str = "nids.log") -> logging.Logger:
    """Return a configured logger.

    Args:
        name: Usually `__name__` from the calling module.
        log_file: Filename inside artifacts/logs/. Different subsystems
            can use different files (e.g. "training.log", "inference.log").

    Returns:
        A logging.Logger ready to use.
    """
    logger = logging.getLogger(name)

    if name in _configured:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler - everything to stdout.
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Rotating file handler - 10 MB per file, keep last 5.
    log_path = Path(LOGS_DIR) / log_file
    file_handler = RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Don't propagate to root - prevents duplicate messages.
    logger.propagate = False
    _configured.add(name)
    return logger
