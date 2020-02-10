import learnergy.utils.logging as l
import torch
from torch.utils import data

logger = l.get_logger(__name__)


class Dataset(data.Dataset):
    """A custom dataset class, inherited from PyTorch's dataset.

    """

    def __init__(self, X, Y):
        """Initialization method.

        Args:
            X (np.array): An n-dimensional array containing the data.
            Y (np.array): An 1-dimensional array containing the data's labels.

        """

        logger.info('Creating class: Dataset.')

        # Samples array
        self._X = X

        # Labels array
        self._Y = Y

        logger.info('Class created.')
        logger.debug(f'X: {X.shape} | Y: {Y.shape}.')

    @property
    def X(self):
        """np.array: An n-dimensional array containing the data.

        """

        return self._X

    @property
    def Y(self):
        """np.array: An 1-dimensional array containing the data's labels.

        """

        return self._Y

    def __getitem__(self, index):
        """A private method that will be the base for PyTorch's iterator getting a new sample.

        Args:
            index (int): The index of desired sample.

        """

        # Gets a sample based on its index
        x = torch.from_numpy(self.X[index]).float()

        # Gets a sample's label based on its index
        y = torch.from_numpy((self.Y[index]).reshape([1, 1]))

        return x, y

    def __len__(self):
        """A private method that will be the base for PyTorch's iterator getting dataset's length.

        """

        return len(self.X)
