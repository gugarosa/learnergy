"""Learnergy exception types."""

import builtins

from learnergy.utils import logging

logger = logging.get_logger(__name__)


class Error(Exception):
    """Base exception that records the Learnergy error category."""

    def __init__(self, cls: str, msg: str) -> None:
        super().__init__(msg)
        logger.error("%s: %s.", cls, msg)


class ArgumentError(Error):
    def __init__(self, error: str) -> None:
        super().__init__("ArgumentError", error)


class BuildError(Error):
    def __init__(self, error: str) -> None:
        super().__init__("BuildError", error)


class SizeError(Error):
    def __init__(self, error: str) -> None:
        super().__init__("SizeError", error)


class TypeError(Error, builtins.TypeError):
    def __init__(self, error: str) -> None:
        super().__init__("TypeError", error)


class ValueError(Error, builtins.ValueError):
    def __init__(self, error: str) -> None:
        super().__init__("ValueError", error)
