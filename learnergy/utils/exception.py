# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Define logged Learnergy exception categories."""

import builtins

from learnergy.utils.logging import get_logger

logger = get_logger(__name__)


class Error(Exception):
    """Represent an error with a Learnergy diagnostic category."""

    def __init__(self, cls: str, msg: str) -> None:
        """Store the original message and log its error category.

        The exception retains the supplied message unchanged.

        Args:
            cls: Category identifying the error in diagnostics.
            msg: Message stored in the exception arguments.

        """

        super().__init__(msg)
        message = str(msg).rstrip(".")
        logger.error(f"`exception={cls}` was raised: {message}.")


class ArgumentError(Error):
    """Represent an invalid argument combination."""

    def __init__(self, error: str) -> None:
        """Initialize an argument error.

        Args:
            error: Message describing the invalid argument combination.

        """

        super().__init__("ArgumentError", error)


class BuildError(Error):
    """Represent a failure to construct a model or its configuration."""

    def __init__(self, error: str) -> None:
        """Initialize a construction error.

        Args:
            error: Message describing the construction failure.

        """

        super().__init__("BuildError", error)


class SizeError(Error):
    """Represent incompatible collection or configuration sizes."""

    def __init__(self, error: str) -> None:
        """Initialize a size error.

        Args:
            error: Message describing the incompatible sizes.

        """

        super().__init__("SizeError", error)


class TypeError(Error, builtins.TypeError):
    """Represent an invalid value type with builtin TypeError compatibility."""

    def __init__(self, error: str) -> None:
        """Initialize a type error.

        Args:
            error: Message describing the invalid value type.

        """

        super().__init__("TypeError", error)


class ValueError(Error, builtins.ValueError):
    """Represent an invalid value with builtin ValueError compatibility."""

    def __init__(self, error: str) -> None:
        """Initialize a value error.

        Args:
            error: Message describing the invalid value.

        """

        super().__init__("ValueError", error)
