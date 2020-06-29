import numpy as np
import torch

import learnergy.utils.exception as e
import learnergy.utils.logging as l

logger = l.get_logger(__name__)


class Dataset(torch.utils.data.Dataset):
    """A custom dataset class, inherited from PyTorch's dataset.

    """

    def __init__(self, data, targets, transform=None):
        """Initialization method.

        Args:
            data (np.array): An n-dimensional array containing the data.
            targets (np.array): An 1-dimensional array containing the data's labels.
            transform (callable): Optional transform to be applied over a sample.

        """

        logger.info('Creating class: Dataset.')

        # Samples array
        self.data = data

        # Labels array
        self.targets = targets

        # Transform callable
        self.transform = transform

        logger.info('Class created.')
        logger.debug(f'Data: {self.data.shape} | Targets: {self.targets.shape} | Transforms: {self.transform}.')

    @property
    def data(self):
        """np.array: An n-dimensional array containing the data.

        """

        return self._data

    @data.setter
    def data(self, data):
        if not isinstance(data, np.ndarray):
            raise e.TypeError('`data` should be a numpy array')

        self._data = data

    @property
    def targets(self):
        """np.array: An 1-dimensional array containing the data's labels.
        
        """

        return self._targets

    @targets.setter
    def targets(self, targets):
        if not isinstance(targets, np.ndarray):
            raise e.TypeError('`targets` should be a numpy array')

        self._targets = targets

    @property
    def transform(self):
        """callable: Optional transform to be applied over a sample.
        
        """

        return self._transform

    @transform.setter
    def transform(self, transform):
        if not hasattr(transform, '__call__'):
            raise e.TypeError('`transform` should be a callable')

        self._transform = transform

    def __getitem__(self, idx):
        """A private method that will be the base for PyTorch's iterator getting a new sample.

        Args:
            idx (int): The idx of desired sample.

        """

        # Gets a sample based on its index
        x = self.data[idx]

        # Gets a sample's label based on its index
        y = self.targets[idx]

        # If there is any transform to be applied
        if self.transform:
            # Applies the transform
            x = self.transform(x)

        return x, y

    def __len__(self):
        """A private method that will be the base for PyTorch's iterator getting dataset's length.

        """

        return len(self.data)
