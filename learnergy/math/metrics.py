"""Image similarity metrics."""

import torch
from skimage.metrics import structural_similarity


def calculate_ssim(v: torch.Tensor, x: torch.Tensor) -> float:
    """Calculate the mean structural similarity of reconstructed images.

    Args:
        v: Reconstructed images, with each image flattened or shaped like its original.
        x: Original grayscale images with shape (batch, height, width).

    Raises:
        ValueError: The batches contain different numbers of images.

    """

    originals = x.detach().cpu().numpy()
    reconstructed = v.detach().cpu().numpy()
    height, width = originals.shape[1:3]

    return sum(
        structural_similarity(
            original,
            rebuilt.reshape(height, width),
            data_range=original.max() - original.min(),
        )
        for original, rebuilt in zip(originals, reconstructed, strict=True)
    ) / len(reconstructed)
