import torch
from torch.utils import data


class Dataset(data.Dataset):
    """
    """

    def __init__(self, X, Y):
        """
        """

        self.X = X

        self.Y = Y

    def __getitem__(self, index):
        """
        """

        x = torch.from_numpy(self.X[index])
        y = torch.from_numpy((self.Y[index]).reshape([1, 1]))

        return x, y

    def __len__(self):
        """
        """

        return len(self.X)
