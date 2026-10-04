# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Provide a Bernoulli Recurrent Temporal Restricted Boltzmann Machine.

Sequences use ``(batch_size, sequence_length, n_visible)`` layout. Each sequence starts from the learned ``h0``;
subsequent hidden biases depend on the preceding hidden probabilities, not sampled states. Training differentiates
through this recurrence while detaching the negative Gibbs particles. No recurrent state persists between calls.
Calling the model returns hidden probabilities shaped ``(batch_size, sequence_length, n_hidden)`` with input
gradients preserved.

References:
    I. Sutskever, G. Hinton, G. Taylor. The Recurrent Temporal Restricted Boltzmann Machine. NeurIPS (2008).

"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import learnergy.utils.exception as e
from learnergy.core.model import _validated_property
from learnergy.models.bernoulli.rbm import RBM
from learnergy.utils.logging import get_logger

logger = get_logger(__name__)


def _validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise e.TypeError(f"`{name}` should be an integer.")
    if value <= 0:
        raise e.ValueError(f"`{name}` should be greater than 0.")


class RTRBM(RBM):
    """Implement a Bernoulli RBM with recurrent hidden biases."""

    W_prime = _validated_property("W_prime", doc="Recurrent weight matrix shaped (n_hidden, n_hidden).")
    h0 = _validated_property("h0", doc="Learnable initial recurrent context shaped (n_hidden,).")

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
    ) -> None:
        """Initialize the RBM parameters and learned recurrent context.

        Args:
            n_visible: Number of visible units at each timestep.
            n_hidden: Number of hidden units at each timestep.
            steps: Number of Contrastive Divergence sampling steps.
            learning_rate: Learning rate used by SGD.
            momentum: Momentum used by SGD.
            decay: Weight decay used by SGD.
            temperature: Positive temperature applied during scaled sampling.
            use_gpu: Whether to select CUDA when it is available.

        Raises:
            TypeError: A unit count or step count is not an integer.
            ValueError: A unit count, step count, optimizer value, or temperature is invalid.

        """

        for name, value in (("n_visible", n_visible), ("n_hidden", n_hidden), ("steps", steps)):
            _validate_positive_integer(name, value)

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

        self.W_prime = nn.Parameter(torch.randn(n_hidden, n_hidden) * 0.01)
        self.h0 = nn.Parameter(torch.zeros(n_hidden))
        self.to(self.W.device)
        self.optimizer.add_param_group({"params": [self.W_prime, self.h0]})

    def pre_activation(self, v: torch.Tensor, h_prev: torch.Tensor | None = None, scale: bool = False) -> torch.Tensor:
        """Compute hidden activations conditioned on the preceding recurrent context.

        Args:
            v: Visible tensor shaped ``(batch_size, n_visible)``.
            h_prev: Context shaped ``(batch_size, n_hidden)`` or None to expand the learned ``h0``.
            scale: Whether to divide activations by the sampling temperature.

        Returns:
            Hidden activations shaped ``(batch_size, n_hidden)`` with gradients preserved.

        Raises:
            learnergy.utils.exception.SizeError: Visible or context dimensions do not match the model.

        """

        if v.ndim != 2 or v.shape[0] == 0 or v.shape[1] != self.n_visible:
            raise e.SizeError("`v` should have shape (batch_size, n_visible) with a nonempty batch.")
        if h_prev is None:
            h_prev = self.h0.unsqueeze(0).expand(v.shape[0], -1)
        if h_prev.shape != (v.shape[0], self.n_hidden):
            raise e.SizeError("`h_prev` should have shape (batch_size, n_hidden).")

        activations = F.linear(v, self.W.t()) + F.linear(h_prev, self.W_prime, self.b)
        if scale:
            activations = activations / self.T

        return activations

    def hidden_sampling(
        self, v: torch.Tensor, h_prev: torch.Tensor | None = None, scale: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample hidden units conditioned on visible units and recurrent context.

        Args:
            v: Visible tensor shaped ``(batch_size, n_visible)``.
            h_prev: Context shaped ``(batch_size, n_hidden)`` or None to use ``h0``.
            scale: Whether to divide hidden activations by the sampling temperature.

        Returns:
            Hidden probabilities followed by Bernoulli states, each shaped ``(batch_size, n_hidden)``.

        """

        probabilities = torch.sigmoid(self.pre_activation(v, h_prev, scale))

        return probabilities, torch.bernoulli(probabilities)

    def energy(self, samples: torch.Tensor, h_prev: torch.Tensor | None = None) -> torch.Tensor:
        """Compute Bernoulli free energy conditioned on recurrent context.

        Args:
            samples: Visible tensor shaped ``(batch_size, n_visible)``.
            h_prev: Context shaped ``(batch_size, n_hidden)`` or None to use ``h0``.

        Returns:
            Free energies shaped ``(batch_size,)`` with gradients through both arguments and model parameters.

        """

        activations = self.pre_activation(samples, h_prev)

        return -torch.mv(samples, self.a) - F.softplus(activations).sum(dim=1)

    def pseudo_likelihood(self, samples: torch.Tensor, h_prev: torch.Tensor | None = None) -> torch.Tensor:
        """Estimate Bernoulli log pseudo-likelihood while holding recurrent context fixed.

        Visible values are rounded before one randomly selected bit per sample is flipped. This bit-flip estimator
        is intended for Bernoulli observations, not continuous Gaussian likelihood evaluation.

        Args:
            samples: Visible tensor shaped ``(batch_size, n_visible)``.
            h_prev: Context shaped ``(batch_size, n_hidden)`` or None to use ``h0``.

        Returns:
            Scalar log pseudo-likelihood with gradients preserved.

        """

        binary = samples.round()
        energy = self.energy(binary, h_prev)
        indexes = torch.randint(self.n_visible, (len(samples), 1), device=samples.device)
        bits = torch.zeros_like(binary).scatter_(1, indexes, 1)
        flipped = torch.where(bits == 0, binary, 1 - binary)

        return (self.n_visible * F.logsigmoid(self.energy(flipped, h_prev) - energy)).mean()

    def gibbs_sampling(
        self, v: torch.Tensor, h_prev: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run Contrastive Divergence at one timestep with fixed recurrent context.

        Args:
            v: Initial visible tensor shaped ``(batch_size, n_visible)``.
            h_prev: Context shaped ``(batch_size, n_hidden)`` or None to use ``h0``.

        Returns:
            Positive hidden probabilities and states, negative hidden probabilities and states, and visible states.

        """

        _validate_positive_integer("steps", self.steps)
        positive_probabilities, positive_states = self.hidden_sampling(v, h_prev)
        negative_states = positive_states

        for _ in range(self.steps):
            _, visible_states = self.visible_sampling(negative_states, scale=True)
            negative_probabilities, negative_states = self.hidden_sampling(visible_states, h_prev, scale=True)

        return positive_probabilities, positive_states, negative_probabilities, negative_states, visible_states

    def cd_step(self, v: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Update parameters with Contrastive Divergence for one timestep.

        Negative particles are detached before differentiation. Use ``fit_subseries`` rather than repeated updates
        with a shared autograd graph to train across a complete sequence.

        Args:
            v: Visible tensor shaped ``(batch_size, n_visible)``.
            h_prev: Recurrent context shaped ``(batch_size, n_hidden)``.

        Returns:
            Detached scalar squared reconstruction error summed over features and averaged over the batch.

        """

        with torch.no_grad():
            visible_states = self.gibbs_sampling(v, h_prev)[-1]
        visible_states = visible_states.detach()
        cost = self.energy(v, h_prev).mean() - self.energy(visible_states, h_prev).mean()
        self._update(cost)

        return ((v - visible_states).square().sum() / len(v)).detach()

    def fit_subseries(self, sequence: torch.Tensor) -> torch.Tensor:
        """Update parameters with one backward pass through a complete batch of sequences.

        Recurrence uses observed inputs and hidden probabilities with gradients intact. Negative Gibbs particles are
        detached. Each call resets context to ``h0`` and updates the optimizer without appending metric history.

        Args:
            sequence: Floating-point tensor shaped ``(batch_size, sequence_length, n_visible)``.

        Returns:
            Detached scalar squared error summed over time and features and averaged over sequences.

        Raises:
            TypeError: The input is not a floating-point tensor.
            ValueError: The input contains nonfinite values.
            learnergy.utils.exception.SizeError: The input has invalid or empty sequence dimensions.

        """

        self._validate_sequence(sequence)
        h_prev = self.h0.unsqueeze(0).expand(len(sequence), -1)
        cost = sequence.new_zeros(())
        mse = sequence.new_zeros(())

        for visible in sequence.unbind(dim=1):
            with torch.no_grad():
                negative = self.gibbs_sampling(visible, h_prev)[-1]
            negative = negative.detach()
            step_cost = self.energy(visible, h_prev).mean() - self.energy(negative, h_prev).mean()
            cost = cost + step_cost
            mse = mse + ((visible - negative).square().sum() / len(sequence)).detach()
            h_prev, _ = self.hidden_sampling(visible, h_prev)

        self._update(cost)

        return mse

    def fit(self, dataset: torch.utils.data.Dataset, batch_size: int = 128, epochs: int = 10) -> torch.Tensor:
        """Train with shuffled batches of complete sequences and append epoch metrics.

        Inputs are moved to the model's device and dtype. Each batch uses an independent recurrent graph, and epoch
        MSE is the mean of the per-batch errors returned by ``fit_subseries``. History stores MSE and time as floats.

        Args:
            dataset: Nonempty dataset yielding ``(sequence, target)`` pairs with ignored targets.
            batch_size: Positive maximum number of sequences in each batch.
            epochs: Positive number of complete training passes.

        Returns:
            Final-epoch detached scalar MSE tensor on the model's device.

        Raises:
            TypeError: A batch size or epoch count is not an integer.
            ValueError: The dataset is empty or a batch size or epoch count is not positive.

        """

        _validate_positive_integer("batch_size", batch_size)
        _validate_positive_integer("epochs", epochs)
        if len(dataset) == 0:
            raise e.ValueError("`dataset` should contain at least one sequence.")
        batches = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        for epoch in range(epochs):
            start = time.time()
            mse = self.W.new_zeros(())
            for samples, _ in batches:
                mse += self.fit_subseries(samples.to(self.W))
            mse /= len(batches)

            self.dump(mse=mse.item(), time=time.time() - start)
            logger.info("Epoch %d/%d - MSE: %f", epoch + 1, epochs, mse.item())

        return mse

    def reconstruct(self, dataset: torch.utils.data.Dataset) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct complete sequences in one non-shuffled batch using observed recurrent context.

        Inputs are moved to the model's device and dtype. Hidden probabilities from each observed timestep condition
        the next timestep. Reconstruction retains autograd tracking, while the returned error is detached.

        Args:
            dataset: Nonempty dataset yielding ``(sequence, target)`` pairs with ignored targets.

        Returns:
            Scalar sampled-state MSE followed by visible conditional values shaped ``(len(dataset), time, n_visible)``.

        Raises:
            ValueError: The dataset is empty.

        """

        if len(dataset) == 0:
            raise e.ValueError("`dataset` should contain at least one sequence.")
        batches = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
        samples, _ = next(iter(batches))
        mse, values, _ = self._reconstruct(samples.to(self.W))

        return mse, values

    def sample(self, n_samples: int = 1, n_steps: int = 10, gibbs_steps: int = 100) -> torch.Tensor:
        """Generate independent sequences using Gibbs sampling at each timestep.

        Every sequence starts from ``h0``. Each timestep initializes a fresh hidden chain, samples visible units, and
        updates recurrent context from the generated observation. This operation does not track gradients or normalize
        generated values, and it does not mutate parameters or training mode.

        Args:
            n_samples: Positive number of sequences to generate.
            n_steps: Positive number of timesteps per sequence.
            gibbs_steps: Positive number of Gibbs transitions per timestep.

        Returns:
            Detached visible states shaped ``(n_samples, n_steps, n_visible)`` on the model's device and dtype.

        Raises:
            TypeError: A sampling count is not an integer.
            ValueError: A sampling count is not positive.

        """

        for name, value in (("n_samples", n_samples), ("n_steps", n_steps), ("gibbs_steps", gibbs_steps)):
            _validate_positive_integer(name, value)

        with torch.no_grad():
            h_prev = self.h0.unsqueeze(0).expand(n_samples, -1)
            outputs = []
            for _ in range(n_steps):
                hidden = torch.bernoulli(self.W.new_full((n_samples, self.n_hidden), 0.5))
                for _ in range(gibbs_steps):
                    visible = self._sample_visible(hidden)
                    _, hidden = self.hidden_sampling(visible, h_prev)
                outputs.append(visible)
                h_prev, _ = self.hidden_sampling(visible, h_prev)

            return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_sequence(x)
        h_prev = self.h0.unsqueeze(0).expand(len(x), -1)
        outputs = []
        for visible in x.unbind(dim=1):
            h_prev, _ = self.hidden_sampling(visible, h_prev)
            outputs.append(h_prev)

        return torch.stack(outputs, dim=1)

    def _validate_sequence(self, sequence: torch.Tensor) -> None:
        if not isinstance(sequence, torch.Tensor) or not sequence.is_floating_point():
            raise e.TypeError("`sequence` should be a floating-point tensor.")
        if sequence.ndim != 3 or 0 in sequence.shape or sequence.shape[-1] != self.n_visible:
            raise e.SizeError("`sequence` should have nonempty shape (batch_size, sequence_length, n_visible).")
        if not torch.isfinite(sequence).all():
            raise e.ValueError("`sequence` should contain only finite values.")

    def _update(self, cost: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()

    def _sample_visible(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.visible_sampling(hidden)[1]

    def _reconstruct(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_sequence(sequence)
        h_prev = self.h0.unsqueeze(0).expand(len(sequence), -1)
        values = []
        states = []
        for visible in sequence.unbind(dim=1):
            h_prev, hidden = self.hidden_sampling(visible, h_prev)
            visible_values, visible_states = self.visible_sampling(hidden)
            values.append(visible_values)
            states.append(visible_states)

        reconstructed = torch.stack(states, dim=1)
        mse = ((sequence - reconstructed).square().sum() / len(sequence)).detach()

        return mse, torch.stack(values, dim=1), reconstructed
