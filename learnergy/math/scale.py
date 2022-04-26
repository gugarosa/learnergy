"""Scaling-related mathematical functions.
"""

import numpy as np

import learnergy.utils.constants as c


def unitary_scale(x: np.array) -> np.array:
    """Scales an array between 0 and 1.

    Args:
        x: A numpy array to be scaled.

    Returns:
        (np.array): Scaled array.

    """

    # Makes sure the array is float typed
    x = x.astype("float32")

    # Gathers array minimum and subtract
    x -= x.min()

    # Normalizes the array using its maximum
    x *= 1.0 / (x.max() + c.EPSILON)

    return x
