"""Metrics-related mathematical functions.
"""

import torch
from skimage.metrics import structural_similarity as ssim

from learnergy.utils import logging

logger = logging.get_logger(__name__)


def calculate_ssim(v: torch.Tensor, x: torch.Tensor) -> float:
    """Calculates the structural similarity of images.

    Args:
        v: Reconstructed images.
        x: Original images.

    Returns:
        (float): Structural similarity between input images.

    """

    total_ssim = 0.0

    x = x.cpu().detach().numpy()
    v = v.cpu().detach().numpy()

    width = x.shape[1]
    height = x.shape[2]

    for z in range(v.shape[0]):
        x_indexed = x[z]
        v_indexed = v[z, :].reshape((width, height))

        total_ssim += ssim(
            x_indexed, v_indexed, data_range=x_indexed.max() - x_indexed.min()
        )

    return total_ssim / v.shape[0]
