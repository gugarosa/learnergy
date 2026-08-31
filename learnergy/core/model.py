"""Standard model-related implementation."""

from collections.abc import Callable
from typing import Any

import torch

import learnergy.utils.exception as e


def _validated_property(
    name: str,
    validator: Callable[[Any, Any], bool] | None = None,
    error: type[Exception] = ValueError,
    message: str = "invalid value",
) -> property:
    storage_name = f"_{name}"

    def getter(instance):
        return getattr(instance, storage_name)

    def setter(instance, value):
        if validator is not None and not validator(instance, value):
            raise error(message)
        setattr(instance, storage_name, value)

    return property(getter, setter)


class Model(torch.nn.Module):
    """Base class for Learnergy models."""

    device = _validated_property(
        "device",
        lambda _, value: value in ("cpu", "cuda"),
        e.TypeError,
        "`device` should be `cpu` or `cuda`",
    )
    history = _validated_property("history")

    def __init__(self, use_gpu: bool = False) -> None:
        """Initialization method.

        Args:
            use_gpu: Whether GPU should be used or not.

        """

        super().__init__()
        torch.set_default_dtype(torch.float32)
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.history = {}

    def dump(self, **kwargs) -> None:
        """Dumps any amount of keyword documents to lists in the history property."""

        for k, v in kwargs.items():
            self.history.setdefault(k, []).append(v)
