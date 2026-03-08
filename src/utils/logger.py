"""
Logging utilities for Audio Event Detection system.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "audio_event_detection",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.

    Args:
        name: Logger name.
        log_level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Path to log file. If None, file logging is disabled.
        console: Whether to log to console.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "audio_event_detection") -> logging.Logger:
    """Get an existing logger by name."""
    return logging.getLogger(name)
