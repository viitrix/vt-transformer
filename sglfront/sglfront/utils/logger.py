from __future__ import annotations

import os
import sys
from typing import Literal

_LOG_LEVEL: int | None = None


def init_logger(
    name: str,
    suffix: str = "",
    *,
    strip_file: bool = True,
    level: str | None = None,
    use_pid: bool | None = None,
):
    """Initialize the logger for the module with colors and pretty formatting."""
    import logging

    global _LOG_LEVEL
    if _LOG_LEVEL is None:
        LEVEL_MAP: dict[str, int] = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

        level = level or os.getenv("LOG_LEVEL", "").upper()
        _LOG_LEVEL = LEVEL_MAP.get(level, logging.INFO)

    if strip_file:
        suffix = os.path.basename(suffix)

    if suffix:
        suffix = f"|{suffix}"

    if use_pid is None:
        use_pid = os.getenv("LOG_PID", "0").lower() in ("1", "true", "yes")

    if use_pid:
        pid = os.getpid()
        suffix = f"|pid={pid}{suffix}"

    class ColorFormatter(logging.Formatter):
        COLORS: dict[str, str] = {
            "DEBUG": "\033[36m",
            "INFO": "\033[32m",
            "WARNING": "\033[33m",
            "ERROR": "\033[31m",
            "CRITICAL": "\033[35m",
        }
        RESET = "\033[0m"
        BOLD = "\033[1m"

        def format(self, record):
            timestamp = self.formatTime(record, "[%Y-%m-%d|%H:%M:%S{suffix}]")
            timestamp = timestamp.format(suffix=suffix)

            level_color = self.COLORS.get(record.levelname, "")
            colored_level = f"{level_color}{record.levelname:<8}{self.RESET}"
            message = record.getMessage()
            return f"{self.BOLD}{timestamp}{self.RESET} {colored_level} {message}"

    logger = logging.getLogger(name)
    logger.setLevel(_LOG_LEVEL)
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    formatter = ColorFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
