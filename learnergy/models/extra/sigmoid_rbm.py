# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Provide an RBM with deterministic sigmoid visible states.

Visible sampling returns the same sigmoid tensor as both probabilities and states, preserving its gradient graph and
performing no Bernoulli draw.

"""

import torch
import torch.nn.functional as F

from learnergy.models.bernoulli.rbm import RBM


class SigmoidRBM(RBM):
    """Implement an RBM with deterministic sigmoid visible states."""

    def visible_sampling(self, h: torch.Tensor, scale: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate deterministic sigmoid visible states.

        Args:
            h: Hidden values shaped ``(batch_size, n_hidden)``.
            scale: Whether to divide visible activations by the temperature.

        Returns:
            Visible probabilities followed by the same tensor as states, shaped ``(batch_size, n_visible)``.

        """

        activations = F.linear(h, self.W, self.a)
        if scale:
            activations = activations / self.T

        probs = torch.sigmoid(activations)
        return probs, probs


class SigmoidRBM4Deep(SigmoidRBM):
    """Provide the sigmoid RBM training default used by deep models."""

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 128,
        epochs: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fit the model with a one-epoch default.

        Behavior, metric persistence, and return order otherwise match ``RBM.fit``.

        Args:
            dataset: Dataset yielding ``(sample, target)`` pairs.
            batch_size: Maximum number of samples in each batch.
            epochs: Number of training passes over the dataset.

        Returns:
            Final scalar MSE tensor followed by the final scalar log pseudo-likelihood tensor.

        """

        return super().fit(dataset, batch_size=batch_size, epochs=epochs)
