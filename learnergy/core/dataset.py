# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Adapt array-backed samples to the PyTorch dataset interface."""

from collections.abc import Callable
from typing import Any

import torch

import learnergy.utils.exception as e
from learnergy.core.model import _validated_property
from learnergy.utils.logging import get_logger

logger = get_logger(__name__)


class Dataset(torch.utils.data.Dataset):
    """Expose samples and targets with an optional sample transform."""

    data = _validated_property("data", doc="Backing sample collection, retained without copying.")
    targets = _validated_property("targets", doc="Targets indexed alongside the sample collection.")
    transform = _validated_property(
        "transform",
        lambda _, value: value is None or callable(value),
        e.TypeError,
        "`transform` should be callable or None.",
        doc="Optional transformation applied when a sample is accessed.",
    )

    def __init__(
        self,
        data: Any,
        targets: Any,
        transform: Callable[[Any], Any] | None = None,
        show_log: bool = True,
    ) -> None:
        """Store sample and target references for indexed access.

        Data is not copied and transforms are applied on access rather than during initialization.

        Args:
            data: Indexable sample collection.
            targets: Indexable target collection aligned with the samples.
            transform: Optional callable that transforms an individual sample.
            show_log: Whether to log dataset creation.

        Raises:
            TypeError: The transform is neither callable nor None.

        """

        self.data = data
        self.targets = targets
        self.transform = transform

        if show_log:
            logger.info("Creating class: Dataset.")
            logger.info("Class created.")

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.targets[idx]

    def __len__(self) -> int:
        return len(self.data)
