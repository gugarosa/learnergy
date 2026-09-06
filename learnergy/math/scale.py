# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Normalize numerical arrays."""

import numpy as np

from learnergy.utils.constants import EPSILON


def unitary_scale(x: np.ndarray) -> np.ndarray:
    """Return a float32 copy scaled by its value range.

    The input is not modified and finite constant arrays become zero arrays.

    Args:
        x: Numeric array to scale.

    Returns:
        Float32 array of the same shape, normalized with an epsilon guard.

    """

    scaled = x.astype("float32")
    scaled -= scaled.min()
    scaled /= scaled.max() + EPSILON
    return scaled
