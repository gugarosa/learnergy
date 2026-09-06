# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Provide batch standardization shared by Gaussian models."""

import torch


def _standardize(samples: torch.Tensor) -> torch.Tensor:
    # Singleton batches use population variance to avoid an undefined sample standard deviation
    correction = 1 if len(samples) > 1 else 0
    std = samples.std(dim=0, correction=correction, keepdim=True)
    return (samples - samples.mean(dim=0, keepdim=True)) / (std + torch.finfo(samples.dtype).eps)
