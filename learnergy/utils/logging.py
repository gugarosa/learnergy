"""Logging-based methods and helpers.
"""

import logging
import sys
from logging import Logger, StreamHandler
from logging.handlers import TimedRotatingFileHandler

FORMATTER = logging.Formatter("%(asctime)s - %(name)s — %(levelname)s — %(message)s")
LOG_FILE = "learnergy.log"


def get_console_handler() -> StreamHandler:
    """Gets a console handler to handle logging into console.

    Returns:
        (StreamHandler): Handler to output information into console.

    """

    # Creates a stream handler for logger
    console_handler = StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)

    return console_handler


def get_timed_file_handler() -> TimedRotatingFileHandler:
    """Gets a timed file handler to handle logging into files.

    Returns:
        (TimedRotatingFileHandler): Handler to output information into timed files.

    """

    # Creates a timed rotating file handler for logger
    file_handler = TimedRotatingFileHandler(LOG_FILE, delay=True, when="midnight")
    file_handler.setFormatter(FORMATTER)

    return file_handler


def get_logger(logger_name: str) -> Logger:
    """Gets a logger and make it avaliable for further use.

    Args:
        logger_name: The name of the logger.

    Returns:
        (Logger): Logger instance.

    """

    # Creates a logger object
    logger = logging.getLogger(logger_name)

    # Sets an log level
    logger.setLevel(logging.DEBUG)

    # Adds the desired handlers
    logger.addHandler(get_console_handler())
    logger.addHandler(get_timed_file_handler())

    # True or False for propagating logs
    logger.propagate = False

    return logger
