"""Logging helpers."""

import logging
import sys
from logging import Logger, StreamHandler
from logging.handlers import TimedRotatingFileHandler

FORMATTER = logging.Formatter("%(asctime)s - %(name)s — %(levelname)s — %(message)s")
LOG_FILE = "learnergy.log"


def get_console_handler() -> StreamHandler:
    """Return the configured console handler."""

    handler = StreamHandler(sys.stdout)
    handler.setFormatter(FORMATTER)
    return handler


def get_timed_file_handler() -> TimedRotatingFileHandler:
    """Return the configured rotating file handler."""

    handler = TimedRotatingFileHandler(LOG_FILE, delay=True, when="midnight")
    handler.setFormatter(FORMATTER)
    return handler


def get_logger(logger_name: str) -> Logger:
    """Return a configured package logger without duplicating handlers."""

    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        logger.addHandler(get_console_handler())
        logger.addHandler(get_timed_file_handler())
        logger.propagate = False
    return logger
