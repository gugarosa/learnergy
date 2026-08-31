"""Image similarity metrics."""

import torch
from skimage.metrics import structural_similarity


def calculate_ssim(v: torch.Tensor, x: torch.Tensor) -> float:
    """Calculate the mean structural similarity of reconstructed images."""

    originals = x.detach().cpu().numpy()
    reconstructed = v.detach().cpu().numpy()
    width, height = originals.shape[1:3]

    return sum(
        structural_similarity(
            original,
            rebuilt.reshape(width, height),
            data_range=original.max() - original.min(),
        )
        for original, rebuilt in zip(originals, reconstructed)
    ) / len(reconstructed)
