# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Configure Learnergy console and rotating-file loggers."""

import logging
import sys
from logging import Logger, StreamHandler
from logging.handlers import TimedRotatingFileHandler

FORMATTER = logging.Formatter("%(asctime)s - %(name)s — %(levelname)s — %(message)s")
LOG_FILE = "learnergy.log"


def get_console_handler() -> StreamHandler:
    """Create a console handler using the package formatter.

    Returns:
        A handler bound to the current standard-output stream.

    """

    handler = StreamHandler(sys.stdout)
    handler.setFormatter(FORMATTER)

    return handler


def get_timed_file_handler() -> TimedRotatingFileHandler:
    """Create a lazily opened log-file handler with midnight rotation.

    The caller owns the returned handler and is responsible for closing it.

    Returns:
        A rotating handler configured for the package log-file path.

    """

    handler = TimedRotatingFileHandler(LOG_FILE, delay=True, when="midnight")
    handler.setFormatter(FORMATTER)

    return handler


def get_logger(logger_name: str) -> Logger:
    """Return a package logger without duplicating its handlers.

    A logger without local handlers receives console and rotating-file handlers.
    Existing local handlers and settings are left unchanged.

    Args:
        logger_name: Name of the logger to retrieve.

    Returns:
        Logger with package handlers or its existing local configuration.

    """

    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        logger.addHandler(get_console_handler())
        logger.addHandler(get_timed_file_handler())
        logger.propagate = False

    return logger
