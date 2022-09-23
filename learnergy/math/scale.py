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

    x = x.astype("float32")

    x -= x.min()
    x *= 1.0 / (x.max() + c.EPSILON)

    return x
