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

    # Defines the total structural similarity
    total_ssim = 0.0

    # Detaches and sends to numpy both tensors
    v = v.cpu().detach().numpy()
    x = x.cpu().detach().numpy()

    # Gathers the width and height of original images
    width = x.shape[1]
    height = x.shape[2]

    # Iterates through every image
    for z in range(v.shape[0]):
        # Gathers the actual image
        x_indexed = x[z]

        # Reshapes the reconstructed image
        v_indexed = v[z, :].reshape((width, height))

        # Sums up to the total similarity
        total_ssim += ssim(
            x_indexed, v_indexed, data_range=x_indexed.max() - x_indexed.min()
        )

    return total_ssim / v.shape[0]
