"""
Quanto: General Purpose LLM Quantization Tool
Logging utilities.
"""

from __future__ import annotations

import logging
import time


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a configured logger with timestamp formatting.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger


def log_with_timestamp(message: str, prefix: str = "") -> None:
    """
    Print a message with timestamp prefix.

    This is a simple utility for cases where a full logger is not needed.

    Args:
        message: Message to print
        prefix: Optional prefix (e.g., module name)
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if prefix:
        print(f"[{timestamp}] {prefix}: {message}")
    else:
        print(f"[{timestamp}] {message}")


class Timer:
    """
    Context manager for timing operations.

    Usage:
        with Timer("model loading") as t:
            load_model()
        print(f"Took {t.elapsed:.2f}s")
    """

    def __init__(self, name: str = "operation", logger: logging.Logger | None = None):
        self.name = name
        self.logger = logger
        self.start_time: float = 0
        self.elapsed: float = 0

    def __enter__(self) -> Timer:
        self.start_time = time.time()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed = time.time() - self.start_time
        if self.logger:
            self.logger.info(f"{self.name} completed in {self.elapsed:.2f}s")
