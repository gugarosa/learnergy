# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Bernoulli-Bernoulli Restricted Boltzmann Machines with Dropout and DropConnect.

Dropout masks hidden probabilities independently for each sample, while DropConnect masks the shared
visible-to-hidden weights. Both variants sample fresh Bernoulli masks on every hidden-sampling call.

References:
    N. Srivastava, et al. Dropout: a simple way to prevent neural networks from overfitting.
    The journal of machine learning research (2014).

"""

import torch
import torch.nn.functional as F

import learnergy.utils.exception as e
from learnergy.core.model import _validated_property
from learnergy.models.bernoulli.rbm import RBM


class DropoutRBM(RBM):
    """Implement a Bernoulli-Bernoulli RBM with hidden-unit dropout."""

    p = _validated_property(
        "p",
        lambda _, value: 0 <= value <= 1,
        e.ValueError,
        "`p` should be between 0 and 1.",
        doc="Probability of dropping each hidden unit.",
    )

    def __init__(
        self,
        n_visible: int = 128,
        n_hidden: int = 128,
        steps: int = 1,
        learning_rate: float = 0.1,
        momentum: float = 0.0,
        decay: float = 0.0,
        temperature: float = 1.0,
        dropout: float = 0.5,
        use_gpu: bool = False,
    ) -> None:
        """Initialize a Bernoulli-Bernoulli RBM with dropout.

        Args:
            n_visible: Number of visible units.
            n_hidden: Number of hidden units.
            steps: Number of Contrastive Divergence sampling steps.
            learning_rate: Learning rate used by stochastic gradient descent.
            momentum: Momentum used by stochastic gradient descent.
            decay: Weight decay used by stochastic gradient descent.
            temperature: Positive temperature applied during scaled sampling.
            dropout: Probability of dropping each hidden unit.
            use_gpu: Whether to use CUDA when it is available.

        Raises:
            ValueError: If a unit count, step count, optimizer value, temperature, or dropout probability is invalid.

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

        self.p = dropout

    def hidden_sampling(self, v: torch.Tensor, scale: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample dropout-masked hidden units conditioned on visible units.

        Args:
            v: Visible tensor shaped ``(batch_size, n_visible)``.
            scale: Whether to divide activations by the sampling temperature.

        Returns:
            Dropout-masked hidden probabilities and states shaped ``(batch_size, n_hidden)``, in that order.

        """

        activations = F.linear(v, self.W.t(), self.b)

        mask = (
            torch.full(
                (activations.size(0), activations.size(1)),
                1 - self.p,
                dtype=torch.float,
                device=self.device,
            )
        ).bernoulli()

        if scale:
            probs = torch.mul(torch.sigmoid(torch.div(activations, self.T)), mask)
        else:
            probs = torch.mul(torch.sigmoid(activations), mask)

        states = torch.bernoulli(probs)

        return probs, states

    def reconstruct(self, dataset: torch.utils.data.Dataset) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct a dataset with dropout temporarily disabled and restored afterward.

        Args:
            dataset: Dataset yielding visible samples and ignored targets.

        Returns:
            Scalar reconstruction error and visible probabilities shaped ``(len(dataset), n_visible)``.

        Notes:
            Autograd tracking is not explicitly disabled.

        """

        p = self.p
        self.p = 0
        try:
            return super().reconstruct(dataset)
        finally:
            self.p = p


class DropConnectRBM(DropoutRBM):
    """Implement a Bernoulli-Bernoulli RBM with visible-to-hidden DropConnect."""

    def hidden_sampling(self, v: torch.Tensor, scale: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample hidden units using dropout-masked visible-to-hidden weights.

        Args:
            v: Visible tensor shaped ``(batch_size, n_visible)``.
            scale: Whether to divide activations by the sampling temperature.

        Returns:
            Hidden probabilities and Bernoulli states, each shaped ``(batch_size, n_hidden)``, in that order.

        """

        mask = (
            torch.full(
                (self.W.size(0), self.W.size(1)),
                1 - self.p,
                dtype=torch.float,
                device=self.device,
            )
        ).bernoulli()

        activations = F.linear(v, torch.mul(self.W, mask).t(), self.b)

        if scale:
            probs = torch.sigmoid(torch.div(activations, self.T))
        else:
            probs = torch.sigmoid(activations)

        states = torch.bernoulli(probs)

        return probs, states
