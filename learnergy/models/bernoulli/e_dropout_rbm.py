# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Bernoulli-Bernoulli Restricted Boltzmann Machine with Energy-based Dropout.

Energy-based Dropout derives a hidden mask from positive- and negative-phase sampling. Hidden sampling uses an
all-ones fallback whenever the stored mask shape does not match the current activation shape.

References:
    M. Roder, G. H. de Rosa, A. L. D. Rossi, J. P. Papa.
    Energy-based Dropout in Restricted Boltzmann Machines: Why Do Not Go Random.
    IEEE Transactions on Emerging Topics in Computational Intelligence (2020).

"""

import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from learnergy.core.model import _validated_property
from learnergy.models.bernoulli.rbm import RBM


class EDropoutRBM(RBM):
    """Implement a Bernoulli-Bernoulli RBM with energy-based dropout."""

    M = _validated_property("M", doc="Current hidden-unit dropout mask.")

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
        """Initialize a Bernoulli-Bernoulli RBM with energy-based dropout.

        Args:
            n_visible: Number of visible units.
            n_hidden: Number of hidden units.
            steps: Number of Contrastive Divergence sampling steps.
            learning_rate: Learning rate used by stochastic gradient descent.
            momentum: Momentum used by stochastic gradient descent.
            decay: Weight decay used by stochastic gradient descent.
            temperature: Positive temperature applied during scaled sampling.
            use_gpu: Whether to use CUDA when it is available.

        Raises:
            ValueError: If a unit count, step count, optimizer value, or temperature is invalid.

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

        self.M = torch.empty(0, device=self.device)

    def hidden_sampling(self, v: torch.Tensor, scale: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample mask-filtered hidden units conditioned on visible units.

        Args:
            v: Visible tensor shaped ``(batch_size, n_visible)``.
            scale: Whether to divide activations by the sampling temperature.

        Returns:
            Mask-filtered hidden probabilities and states shaped ``(batch_size, n_hidden)``, in that order.

        """

        activations = F.linear(v, self.W.t(), self.b)
        mask = self.M if self.M.shape == activations.shape else torch.ones_like(activations)

        if scale:
            probs = torch.mul(torch.sigmoid(torch.div(activations, self.T)), mask)
        else:
            probs = torch.mul(torch.sigmoid(activations), mask)

        states = torch.bernoulli(probs)

        return probs, states

    def total_energy(self, h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute mean joint energy for paired hidden and visible states.

        Args:
            h: Hidden-state tensor shaped ``(batch_size, n_hidden)``.
            v: Visible-state tensor shaped ``(batch_size, n_visible)``.

        Returns:
            Scalar mean joint-energy tensor with gradients preserved.

        """

        e_h = -torch.mv(h, self.b)
        e_v = -torch.mv(v, self.a)
        e_rec = -torch.sum(torch.mm(v, self.W) * h, dim=1)

        energy = torch.mean(e_h + e_v + e_rec)

        return energy

    def energy_dropout(self, e: torch.Tensor, p_prob: torch.Tensor, n_prob: torch.Tensor) -> None:
        """Replace the stored mask using energy-based importance sampling.

        Args:
            e: Scalar energy difference between negative and positive phases.
            p_prob: Positive-phase hidden probabilities shaped ``(batch_size, n_hidden)``.
            n_prob: Negative-phase hidden probabilities shaped ``(batch_size, n_hidden)``.

        """

        eps = torch.finfo(p_prob.dtype).eps
        I = n_prob / (p_prob + eps) / (torch.abs(e) + eps)
        I = I / torch.max(I, 0)[0].clamp_min(eps)

        p = torch.rand((I.size(0), I.size(1)), device=self.device)

        self.M = (I < p).float()

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 128,
        epochs: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update model parameters, metric history, and the stored mask with shuffled energy-dropout batches.

        Args:
            dataset: Dataset yielding visible samples and ignored targets.
            batch_size: Maximum number of samples per training batch.
            epochs: Number of complete training passes.

        Returns:
            Final-epoch scalar mean squared error and detached log pseudo-likelihood tensors in that order.

        """

        batches = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        for _ in range(epochs):
            start = time.time()

            mse, pl = 0, 0

            for samples, _ in batches:
                batch_size = samples.size(0)

                self.M = torch.ones((batch_size, self.n_hidden), device=self.device)

                samples = samples.reshape(len(samples), self.n_visible).to(self.device)

                (
                    pos_hidden_probs,
                    pos_hidden_states,
                    neg_hidden_probs,
                    neg_hidden_states,
                    visible_states,
                ) = self.gibbs_sampling(samples)

                e = self.total_energy(pos_hidden_states, samples)
                e1 = self.total_energy(neg_hidden_states, visible_states)

                self.energy_dropout(e1 - e, pos_hidden_probs, neg_hidden_probs)

                _, _, _, _, visible_states = self.gibbs_sampling(samples)
                visible_states = visible_states.detach()

                cost = torch.mean(self.energy(samples)) - torch.mean(self.energy(visible_states))

                self.optimizer.zero_grad()
                cost.backward()
                self.optimizer.step()

                batch_mse = torch.div(torch.sum(torch.pow(samples - visible_states, 2)), batch_size)
                batch_pl = self.pseudo_likelihood(samples).detach()

                mse += batch_mse
                pl += batch_pl

            mse /= len(batches)
            pl /= len(batches)

            end = time.time()

            self.dump(mse=mse.item(), pl=pl.item(), time=end - start)

        return mse, pl

    def reconstruct(self, dataset: torch.utils.data.Dataset) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct an entire dataset and leave the stored energy-dropout mask filled with ones.

        Args:
            dataset: Dataset yielding visible samples and ignored targets.

        Returns:
            Scalar reconstruction error and visible probabilities shaped ``(len(dataset), n_visible)``.

        Notes:
            Autograd tracking is not explicitly disabled.

        """

        mse = 0
        batch_size = len(dataset)

        batches = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        for samples, _ in batches:
            self.M = torch.ones((batch_size, self.n_hidden), device=self.device)

            samples = samples.reshape(len(samples), self.n_visible).to(self.device)

            _, pos_hidden_states = self.hidden_sampling(samples)
            visible_probs, visible_states = self.visible_sampling(pos_hidden_states)

            batch_mse = torch.div(torch.sum(torch.pow(samples - visible_states, 2)), batch_size)
            mse += batch_mse

        mse /= len(batches)

        return mse, visible_probs
