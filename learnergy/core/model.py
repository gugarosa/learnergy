"""Standard model-related implementation.
"""

from typing import Any, Dict, Optional

import torch

import learnergy.utils.exception as e
from learnergy.utils import logging

logger = logging.get_logger(__name__)


class Model(torch.nn.Module):
    """The Model class is the basis for any custom model.

    One can configure, if necessary, different properties or methods that
    can be used throughout all childs.

    """

    def __init__(self, use_gpu: Optional[bool] = False) -> None:
        """Initialization method.

        Args:
            use_gpu: Whether GPU should be used or not.

        """

        super(Model, self).__init__()

        self.device = "cpu"
        if torch.cuda.is_available() and use_gpu:
            self.device = "cuda"

        self.history = {}

        # Sets default tensor type to float to avoid value errors
        torch.set_default_tensor_type(torch.FloatTensor)

        logger.debug("Device: %s.", self.device)

    @property
    def device(self) -> str:
        """Indicates which device is being used for computation."""

        return self._device

    @device.setter
    def device(self, device: str) -> None:
        if device not in ["cpu", "cuda"]:
            raise e.TypeError("`device` should be `cpu` or `cuda`")

        self._device = device

    @property
    def history(self) -> Dict[str, Any]:
        """Dictionary containing historical values from the model."""

        return self._history

    @history.setter
    def history(self, history: Dict[str, Any]) -> None:
        self._history = history

    def dump(self, **kwargs) -> None:
        """Dumps any amount of keyword documents to lists in the history property."""

        for k, v in kwargs.items():
            if k not in self.history.keys():
                self.history[k] = []

            self.history[k].append(v)
