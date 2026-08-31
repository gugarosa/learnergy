"""Dataset helpers."""

from collections.abc import Callable

import torch

import learnergy.utils.exception as e
from learnergy.core.model import _validated_property
from learnergy.utils import logging

logger = logging.get_logger(__name__)


class Dataset(torch.utils.data.Dataset):
    """Wrap samples and targets with an optional sample transform."""

    data = _validated_property("data")
    targets = _validated_property("targets")
    transform = _validated_property(
        "transform",
        lambda _, value: value is None or callable(value),
        e.TypeError,
        "`transform` should be a callable or None",
    )

    def __init__(
        self,
        data,
        targets,
        transform: Callable | None = None,
        show_log: bool = True,
    ) -> None:
        self.data = data
        self.targets = targets
        self.transform = transform

        if show_log:
            logger.info("Creating class: Dataset.")
            logger.info("Class created.")

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.targets[idx]

    def __len__(self) -> int:
        return len(self.data)
