# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Provide a recurrent temporal RBM with fixed-variance Gaussian visible units.

Normalization follows ``GaussianRBM`` using batch-local statistics over the combined sequence and time dimensions.
Standardized inputs are detached. Training and reconstruction use deterministic visible conditional means, while
generative sampling adds unit Gaussian noise. Both operations return continuous visible values rather than sigmoid
values. Training clips the total gradient norm to one.

"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import learnergy.utils.exception as e
from learnergy.core.model import _validated_property
from learnergy.models.gaussian._normalization import _standardize
from learnergy.models.temporal.rtrbm import RTRBM


class RTGaussianRBM(RTRBM):
    """Implement a recurrent temporal RBM with unit-variance Gaussian visible units."""

    normalize = _validated_property("normalize", doc="Whether training and reconstruction batches are standardized.")
    input_normalize = _validated_property(
        "input_normalize", doc="Whether forward inputs are standardized and detached before hidden sampling."
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
        use_gpu: bool = False,
        normalize: bool = True,
        input_normalize: bool = True,
    ) -> None:
        """Initialize a recurrent Gaussian-Bernoulli RBM.

        Args:
            n_visible: Number of visible units at each timestep.
            n_hidden: Number of hidden units at each timestep.
            steps: Number of Contrastive Divergence sampling steps.
            learning_rate: Learning rate used by SGD.
            momentum: Momentum used by SGD.
            decay: Weight decay used by SGD.
            temperature: Positive temperature applied during scaled sampling.
            use_gpu: Whether to select CUDA when it is available.
            normalize: Whether to standardize training and reconstruction batches over sequences and time.
            input_normalize: Whether to standardize and detach sequence inputs passed through ``forward``.

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

        self.normalize = normalize
        self.input_normalize = input_normalize

    def energy(self, samples: torch.Tensor, h_prev: torch.Tensor | None = None) -> torch.Tensor:
        """Compute Gaussian free energy conditioned on recurrent context.

        Args:
            samples: Visible tensor shaped ``(batch_size, n_visible)``.
            h_prev: Context shaped ``(batch_size, n_hidden)`` or None to use ``h0``.

        Returns:
            Free energies shaped ``(batch_size,)`` with gradients preserved.

        """

        activations = self.pre_activation(samples, h_prev)
        quadratic = 0.5 * (samples - self.a).square().sum(dim=1)

        return quadratic - F.softplus(activations).sum(dim=1)

    def visible_sampling(self, h: torch.Tensor, scale: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate deterministic Gaussian visible conditional values.

        This method follows ``GaussianRBM.visible_sampling``. Generative ``sample`` adds unit Gaussian noise to the
        conditional mean instead of treating the sigmoid values as continuous observations.

        Args:
            h: Hidden tensor shaped ``(batch_size, n_hidden)``.
            scale: Whether to divide visible activations by the sampling temperature.

        Returns:
            Sigmoid values followed by deterministic visible states, each shaped ``(batch_size, n_visible)``.

        """

        states = F.linear(h, self.W, self.a)
        if scale:
            states = states / self.T

        return torch.sigmoid(states), states

    def fit_subseries(self, sequence: torch.Tensor) -> torch.Tensor:
        """Train one sequence batch with optional standardization and gradient clipping.

        Standardization pools all batch timesteps, uses sample standard deviation when more than one observation is
        present, and centers a singleton observation to zero. Standardized inputs are detached before training.

        Args:
            sequence: Floating-point tensor shaped ``(batch_size, sequence_length, n_visible)``.

        Returns:
            Detached scalar error summed over time and features and averaged over sequences in the training space.

        """

        if self.normalize:
            sequence = self._standardize_sequence(sequence)

        return super().fit_subseries(sequence)

    def reconstruct(self, dataset: torch.utils.data.Dataset) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct continuous sequence values in one optionally standardized batch.

        Standardization uses batch-local statistics over all timesteps and detaches its result. Reconstruction values
        remain in that standardized space and retain gradients through the model.

        Args:
            dataset: Nonempty dataset yielding ``(sequence, target)`` pairs with ignored targets.

        Returns:
            Detached scalar MSE followed by continuous means shaped ``(len(dataset), time, n_visible)``.

        Raises:
            ValueError: The dataset is empty.

        """

        if len(dataset) == 0:
            raise e.ValueError("`dataset` should contain at least one sequence.")
        batches = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
        samples, _ = next(iter(batches))
        samples = samples.to(self.W)
        if self.normalize:
            samples = self._standardize_sequence(samples)
        mse, _, states = self._reconstruct(samples)

        return mse, states

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_normalize:
            x = self._standardize_sequence(x)

        return super().forward(x)

    def _standardize_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        self._validate_sequence(sequence)

        return _standardize(sequence.reshape(-1, self.n_visible)).reshape_as(sequence).detach()

    def _update(self, cost: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0, error_if_nonfinite=True)
        self.optimizer.step()

    def _sample_visible(self, hidden: torch.Tensor) -> torch.Tensor:
        mean = self.visible_sampling(hidden)[1]

        return mean + torch.randn_like(mean)
