import numpy as np
import pytest

from learnergy.core import dataset


def test_dataset():
    data = np.asarray([[1, 2], [2, 4]])
    targets = np.asarray([1, 2])

    new_dataset = dataset.Dataset(data, targets)
    
    assert len(new_dataset) == 2
    assert len(new_dataset.data) == 2
    assert len(new_dataset.targets) == 2

    x, y = new_dataset[0]

    assert x.shape == (2, )
    assert y.shape == ()
