"""Sigmoid-Bernoulli Restricted Boltzmann Machine."""

import torch
import torch.nn.functional as F

from learnergy.models.bernoulli.rbm import RBM


class SigmoidRBM(RBM):
    """RBM with deterministic sigmoid visible units."""

    def visible_sampling(
        self, h: torch.Tensor, scale: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute visible probabilities and use them as the visible state."""

        activations = F.linear(h, self.W, self.a)
        if scale:
            activations = activations / self.T

        probs = torch.sigmoid(activations)
        return probs, probs


class SigmoidRBM4Deep(SigmoidRBM):
    """Sigmoid RBM variant used during layer-wise DBN training."""

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 128,
        epochs: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return super().fit(dataset, batch_size=batch_size, epochs=epochs)
