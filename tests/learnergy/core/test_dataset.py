import numpy as np
import pytest
from learnergy.core import dataset


def test_dataset():
    def transform(x):
        return x

    data = np.asarray([[1, 2], [2, 4]])
    targets = np.asarray([1, 2])

    new_dataset = dataset.Dataset(data, targets, transform)

    assert len(new_dataset) == 2
    assert len(new_dataset.data) == 2
    assert len(new_dataset.targets) == 2

    x, y = new_dataset[0]

    assert x.shape == (2, )
    assert y.shape == ()
