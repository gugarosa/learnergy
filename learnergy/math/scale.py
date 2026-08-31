"""Scaling helpers."""

import numpy as np

from learnergy.utils.constants import EPSILON


def unitary_scale(x: np.ndarray) -> np.ndarray:
    """Scale an array to the interval from zero to one."""

    scaled = x.astype("float32")
    scaled -= scaled.min()
    scaled /= scaled.max() + EPSILON
    return scaled
