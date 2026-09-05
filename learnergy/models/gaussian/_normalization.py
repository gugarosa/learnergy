"""Batch standardization shared by Gaussian models."""

import torch


def standardize(samples: torch.Tensor) -> torch.Tensor:
    """Use sample variance for full batches and center singleton batches at zero."""

    correction = 1 if len(samples) > 1 else 0
    std = samples.std(dim=0, correction=correction, keepdim=True)
    return (samples - samples.mean(dim=0, keepdim=True)) / (
        std + torch.finfo(samples.dtype).eps
    )
