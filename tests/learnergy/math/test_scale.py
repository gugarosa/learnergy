import numpy as np
import pytest

from learnergy.math import scale


def test_unitary_scale():
    array = np.array([1, 2, 3, 4, 5])

    unitary_array = scale.unitary_scale(array)

    assert np.min(unitary_array) == 0

    assert np.round(np.max(unitary_array)) == 1
