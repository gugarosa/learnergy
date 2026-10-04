# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Provide a recurrent temporal RBM with learned Gaussian visible variance.

As in ``VarianceGaussianRBM``, effective variance is ``sigma**2`` plus dtype-dependent epsilon. Sampling returns
conditional means before random states, and Contrastive Divergence uses those states as negative particles. Inputs are
not standardized. Training clips the gradient norm to one and bounds trainable scales to the interval [0.1, 10].
A nonfinite gradient norm raises an error rather than updating the parameters.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from learnergy.core.model import _validated_property
from learnergy.models.temporal.rtrbm import RTRBM


class RTVarianceGaussianRBM(RTRBM):
    """Implement a recurrent temporal RBM with a learned scale for each visible unit."""

    sigma = _validated_property("sigma", doc="Learnable visible scale whose square determines the variance.")

    def __init__(
        self,
        n_visible: int = 128,
        n_hidden: int = 128,
        steps: int = 1,
        learning_rate: float = 0.001,
        momentum: float = 0.0,
        decay: float = 0.0,
        temperature: float = 1.0,
        use_gpu: bool = False,
    ) -> None:
        """Initialize a recurrent Gaussian-Bernoulli RBM with learned visible variance.

        Args:
            n_visible: Number of visible units at each timestep.
            n_hidden: Number of hidden units at each timestep.
            steps: Number of Contrastive Divergence sampling steps.
            learning_rate: Learning rate used by SGD.
            momentum: Momentum used by SGD.
            decay: Weight decay used by SGD.
            temperature: Positive temperature applied during scaled hidden sampling.
            use_gpu: Whether to select CUDA when it is available.

        Raises:
            TypeError: A unit count or step count is not an integer.
            ValueError: A unit count, step count, optimizer value, or temperature is invalid.

        """

        super().__init__(
            n_visible,
            n_hidden,
            steps,
            learning_rate,
            momentum,
            decay,
            temperature,
            use_gpu,
        )

        self.sigma = nn.Parameter(torch.ones(n_visible))
        self.to(self.W.device)
        self.optimizer.add_param_group({"params": self.sigma})

    def pre_activation(self, v: torch.Tensor, h_prev: torch.Tensor | None = None, scale: bool = False) -> torch.Tensor:
        """Compute hidden activations using variance-scaled visible inputs.

        Args:
            v: Visible tensor shaped ``(batch_size, n_visible)``.
            h_prev: Context shaped ``(batch_size, n_hidden)`` or None to use ``h0``.
            scale: Whether to divide hidden activations by the sampling temperature.

        Returns:
            Hidden activations shaped ``(batch_size, n_hidden)`` with gradients preserved.

        Raises:
            learnergy.utils.exception.SizeError: Visible or context dimensions do not match the model.

        """

        variance = self.sigma.square() + torch.finfo(v.dtype).eps

        return super().pre_activation(v / variance, h_prev, scale)

    def visible_sampling(self, h: torch.Tensor, scale: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample Gaussian visible states using the learned visible variance.

        Args:
            h: Hidden tensor shaped ``(batch_size, n_hidden)``.
            scale: Accepted for sampling API compatibility without changing the visible distribution.

        Returns:
            Conditional means followed by sampled states, each shaped ``(batch_size, n_visible)``.

        """

        activations = F.linear(h, self.W, self.a)
        variance = self.sigma.square() + torch.finfo(activations.dtype).eps
        std = variance.sqrt().expand_as(activations)
        states = torch.normal(activations, std)

        return activations, states

    def energy(self, samples: torch.Tensor, h_prev: torch.Tensor | None = None) -> torch.Tensor:
        """Compute Gaussian free energy with learned variance and recurrent context.

        Args:
            samples: Visible tensor shaped ``(batch_size, n_visible)``.
            h_prev: Context shaped ``(batch_size, n_hidden)`` or None to use ``h0``.

        Returns:
            Free energies shaped ``(batch_size,)`` with gradients preserved.

        Raises:
            learnergy.utils.exception.SizeError: Visible or context dimensions do not match the model.

        """

        variance = self.sigma.square() + torch.finfo(samples.dtype).eps
        activations = self.pre_activation(samples, h_prev)
        v = ((samples - self.a).square() / (2 * variance)).sum(dim=1)
        h = F.softplus(activations).sum(dim=1)

        energy = v - h

        return energy

    def _update(self, cost: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0, error_if_nonfinite=True)
        self.optimizer.step()

        if self.sigma.requires_grad:
            with torch.no_grad():
                self.sigma.clamp_(min=0.1, max=10.0)
