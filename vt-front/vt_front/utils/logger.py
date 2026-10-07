from __future__ import annotations

_LOG_LEVEL = None


def init_logger(
    name: str,
    suffix: str = "",
    *,
    strip_file: bool = True,
    level: str | None = None,
    use_pid: bool = None,
):
    """Initialize the logger for the module with colors and pretty formatting."""
    import logging
    import os
    import sys

    global _LOG_LEVEL
    if _LOG_LEVEL is None:
        LEVEL_MAP = {
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

    # Color formatter class
    class ColorFormatter(logging.Formatter):
        """Formatter with colors and pretty output"""

        # ANSI color codes
        COLORS = {
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[32m",  # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[35m",  # Magenta
        }
        RESET = "\033[0m"
        BOLD = "\033[1m"

        def format(self, record):
            # Format timestamp like: [YYYY-MM-DD|HH:MM:SS|suffix]
            timestamp = self.formatTime(record, "[%Y-%m-%d|%H:%M:%S{suffix}]")
            timestamp = timestamp.format(suffix=suffix)

            # Get color for log level
            level_color = self.COLORS.get(record.levelname, "")

            # Format the message
            colored_level = f"{level_color}{record.levelname:<8}{self.RESET}"
            message = record.getMessage()

            # Pretty format: [timestamp] LEVEL message
            return f"{self.BOLD}{timestamp}{self.RESET} {colored_level} {message}"

    logger = logging.getLogger(name)
    logger.setLevel(_LOG_LEVEL)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    formatter = ColorFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger
