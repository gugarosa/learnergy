# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Calculate image similarity metrics."""

import torch
from skimage.metrics import structural_similarity


def calculate_ssim(v: torch.Tensor, x: torch.Tensor) -> float:
    """Calculate the mean structural similarity of reconstructed images.

    Tensors are detached and copied to CPU as needed before comparison.
    Each original image supplies its own intensity range to the SSIM calculation.

    Args:
        v: Reconstructed images, with each image flattened or shaped like its original.
        x: Original grayscale images with shape (batch, height, width).

    Returns:
        Mean SSIM over the paired images.

    Raises:
        ValueError: Batch lengths, image shapes, or SSIM window requirements are incompatible.

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
