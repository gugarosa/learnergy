import torch

import learnergy.utils.exception as e
import learnergy.utils.logging as l

logger = l.get_logger(__name__)


class Model(torch.nn.Module):
    """The Model class is the basis for any custom model.

    One can configure, if necessary, different properties or methods that
    can be used throughout all childs.

    """

    def __init__(self, use_gpu=False):
        """Initialization method.

        Args:
            use_gpu (bool): Whether GPU should be used or not.

        """

        # Override its parent class
        super(Model, self).__init__()

        # Creates a cpu-based device property
        self.device = 'cpu'

        # Checks if GPU is avaliable
        if torch.cuda.is_available() and use_gpu:
            # If yes, change the device property to `cuda`
            self.device = 'cuda'

        # Creating an empty dictionary to hold historical values
        self.history = {}

        # Setting default tensor type to float
        torch.set_default_tensor_type(torch.FloatTensor)

        logger.debug(f'Device: {self.device}.')

    @property
    def device(self):
        """str: Indicates which device is being used for computation.

        """

        return self._device

    @device.setter
    def device(self, device):
        if device not in ['cpu', 'cuda']:
            raise e.TypeError('`device` should be `cpu` or `cuda`')

        self._device = device

    @property
    def history(self):
        """dict: Dictionary containing historical values from the model.

        """

        return self._history

    @history.setter
    def history(self, history):
        if not isinstance(history, dict):
            raise e.TypeError('`history` should be a dictionary')

        self._history = history

    def dump(self, **kwargs):
        """Dumps any amount of keyword documents to lists in the history property.

        """

        # Iterate through key-word arguments
        for k, v in kwargs.items():
            # Check if there is already an instance of current
            if k not in self.history.keys():
                # If not, creates an empty list
                self.history[k] = []

            # Appends the new value to the list
            self.history[k].append(v)
