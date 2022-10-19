"""Dataset-related classes.
"""

from typing import Optional, Tuple
from xmlrpc.client import Boolean

import numpy as np
import torch

import learnergy.utils.exception as e
from learnergy.utils import logging

logger = logging.get_logger(__name__)


class Dataset(torch.utils.data.Dataset):
    """A custom dataset class, inherited from PyTorch's dataset."""

    def __init__(
        self, data: np.array, targets: np.array, 
        transform: Optional[callable] = None, show_log: Optional[Boolean] = True
    ) -> None:
        """Initialization method.

        Args:
            data: An n-dimensional array containing the data.
            targets: An 1-dimensional array containing the data's labels.
            transform: Optional transform to be applied over a sample.

        """

        self.data = data
        self.targets = targets
        self.transform = transform

        if show_log:
            logger.info("Creating class: Dataset.")
            logger.info("Class created.")
            logger.debug(
                "Data: %s | Targets: %s | Transforms: %s.",
                self.data.shape,
                self.targets.shape,
                self.transform,
            )

    @property
    def data(self) -> np.array:
        """An n-dimensional array containing the data."""

        return self._data

    @data.setter
    def data(self, data: np.array) -> None:
        self._data = data

    @property
    def targets(self) -> np.array:
        """An 1-dimensional array containing the data's labels."""

        return self._targets

    @targets.setter
    def targets(self, targets: np.array) -> None:
        self._targets = targets

    @property
    def transform(self) -> callable:
        """Optional transform to be applied over a sample."""

        return self._transform

    @transform.setter
    def transform(self, transform: callable) -> None:
        if not (hasattr(transform, "__call__") or transform is None):
            raise e.TypeError("`transform` should be a callable or None")

        self._transform = transform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """A private method that will be the base for PyTorch's iterator getting a new sample.

        Args:
            idx: The idx of desired sample.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor]): Data and label tensors.

        """

        x = self.data[idx]
        y = self.targets[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self) -> int:
        """A private method that will be the base for PyTorch's iterator getting dataset's length.

        Returns:
            (int): Length of dataset.

        """

        return len(self.data)
