import numpy as np
import pytest

from learnergy.core import dataset


def test_dataset():
    X = np.asarray([[1, 2], [2, 4]])
    Y = np.asarray([1, 2])

    new_dataset = dataset.Dataset(X, Y)
    
    assert len(new_dataset) == 2
    assert len(new_dataset.X) == 2
    assert len(new_dataset.Y) == 2

    x, y = new_dataset[0]

    assert len(x) == 2
    assert len(y) == 1
