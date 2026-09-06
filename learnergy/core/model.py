# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Provide shared model state and validated attributes."""

from collections.abc import Callable
from typing import Any

import torch

import learnergy.utils.exception as e


def _validated_property(
    name: str,
    validator: Callable[[Any, Any], bool] | None = None,
    error: type[Exception] = ValueError,
    message: str = "`value` is invalid.",
    *,
    doc: str | None = None,
) -> property:
    storage_name = f"_{name}"

    def _get(instance: Any) -> Any:
        return getattr(instance, storage_name)

    def _set(instance: Any, value: Any) -> None:
        if validator is not None and not validator(instance, value):
            raise error(message)

        setattr(instance, storage_name, value)

    return property(_get, _set, doc=doc)


class Model(torch.nn.Module):
    """Provide device selection and metric history for energy-based models."""

    device = _validated_property(
        "device",
        lambda _, value: value in ("cpu", "cuda"),
        e.TypeError,
        "`device` should be `cpu` or `cuda`.",
        doc="Computation device selected during initialization.",
    )
    history = _validated_property("history", doc="Recorded values grouped into lists by metric name.")

    def __init__(self, use_gpu: bool = False) -> None:
        """Initialize device selection and an empty metric history.

        Initialization sets PyTorch's process-wide default floating-point dtype to float32.

        Args:
            use_gpu: Whether to select CUDA when it is available.

        """

        super().__init__()
        torch.set_default_dtype(torch.float32)
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.history = {}

    def dump(self, **kwargs: Any) -> None:
        """Append values to their per-metric history lists.

        Existing history is retained and values are stored without conversion or copying.

        Args:
            **kwargs: Values keyed by the metric names to update.

        """

        for k, v in kwargs.items():
            self.history.setdefault(k, []).append(v)
