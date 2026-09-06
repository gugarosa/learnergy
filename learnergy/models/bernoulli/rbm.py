# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Bernoulli-Bernoulli Restricted Boltzmann Machine.

The model uses Contrastive Divergence for training. Calling the model returns hidden probabilities while still
sampling hidden states as part of the forward operation.

References:
    G. Hinton. A practical guide to training restricted Boltzmann machines.
    Neural networks: Tricks of the trade (2012).

"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.data import DataLoader

import learnergy.utils.exception as e
from learnergy.core.model import Model, _validated_property


class RBM(Model):
    """Implement a Bernoulli-Bernoulli Restricted Boltzmann Machine."""

    n_visible = _validated_property(
        "n_visible",
        lambda _, value: value > 0,
        e.ValueError,
        "`n_visible` should be > 0.",
        doc="Number of visible units.",
    )
    n_hidden = _validated_property(
        "n_hidden",
        lambda _, value: value > 0,
        e.ValueError,
        "`n_hidden` should be > 0.",
        doc="Number of hidden units.",
    )
    steps = _validated_property(
        "steps",
        lambda _, value: value > 0,
        e.ValueError,
        "`steps` should be > 0.",
        doc="Number of Contrastive Divergence steps.",
    )
    lr = _validated_property(
        "lr",
        lambda _, value: value >= 0,
        e.ValueError,
        "`lr` should be >= 0.",
        doc="Stored learning-rate setting used at initialization.",
    )
    momentum = _validated_property(
        "momentum",
        lambda _, value: value >= 0,
        e.ValueError,
        "`momentum` should be >= 0.",
        doc="Stored momentum setting used at initialization.",
    )
    decay = _validated_property(
        "decay",
        lambda _, value: value >= 0,
        e.ValueError,
        "`decay` should be >= 0.",
        doc="Stored weight-decay setting used at initialization.",
    )
    T = _validated_property(
        "T",
        lambda _, value: value > 0,
        e.ValueError,
        "`T` should be > 0.",
        doc="Sampling temperature.",
    )
    W = _validated_property("W", doc="Visible-to-hidden weight matrix.")
    a = _validated_property("a", doc="Visible-unit bias vector.")
    b = _validated_property("b", doc="Hidden-unit bias vector.")
    optimizer = _validated_property("optimizer", doc="Stochastic gradient descent optimizer.")

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
        """Initialize a Bernoulli-Bernoulli RBM.

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

        super().__init__(use_gpu=use_gpu)

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        self.steps = steps
        self.lr = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.T = temperature

        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.a = nn.Parameter(torch.zeros(n_visible))
        self.b = nn.Parameter(torch.zeros(n_hidden))

        self.to(self.device)
        self.optimizer = opt.SGD(self.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)

    def pre_activation(self, v: torch.Tensor, scale: bool = False) -> torch.Tensor:
        """Compute hidden-unit pre-activations.

        Args:
            v: Visible tensor shaped ``(batch_size, n_visible)``.
            scale: Whether to divide activations by the sampling temperature.

        Returns:
            Hidden pre-activations shaped ``(batch_size, n_hidden)`` with gradients preserved.

        """

        activations = F.linear(v, self.W.t(), self.b)

        if scale:
            activations = torch.div(activations, self.T)

        return activations

    def hidden_sampling(self, v: torch.Tensor, scale: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample hidden units conditioned on visible units.

        Args:
            v: Visible tensor shaped ``(batch_size, n_visible)``.
            scale: Whether to divide activations by the sampling temperature.

        Returns:
            A tuple of hidden probabilities and Bernoulli states, each shaped ``(batch_size, n_hidden)``.

        """

        activations = F.linear(v, self.W.t(), self.b)

        if scale:
            probs = torch.sigmoid(torch.div(activations, self.T))
        else:
            probs = torch.sigmoid(activations)

        states = torch.bernoulli(probs)

        return probs, states

    def visible_sampling(self, h: torch.Tensor, scale: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample visible units conditioned on hidden units.

        Args:
            h: Hidden tensor shaped ``(batch_size, n_hidden)``.
            scale: Whether to divide activations by the sampling temperature.

        Returns:
            A tuple of visible probabilities and Bernoulli states, each shaped ``(batch_size, n_visible)``.

        """

        activations = F.linear(h, self.W, self.a)

        if scale:
            probs = torch.sigmoid(torch.div(activations, self.T))
        else:
            probs = torch.sigmoid(activations)

        states = torch.bernoulli(probs)

        return probs, states

    def gibbs_sampling(
        self, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run Contrastive Divergence sampling from visible units.

        Args:
            v: Visible tensor shaped ``(batch_size, n_visible)``.

        Returns:
            Positive hidden probabilities and states, negative hidden probabilities and states, and visible states.

        """

        pos_hidden_probs, pos_hidden_states = self.hidden_sampling(v)
        neg_hidden_states = pos_hidden_states

        for _ in range(self.steps):
            _, visible_states = self.visible_sampling(neg_hidden_states, True)
            neg_hidden_probs, neg_hidden_states = self.hidden_sampling(visible_states, True)

        return (
            pos_hidden_probs,
            pos_hidden_states,
            neg_hidden_probs,
            neg_hidden_states,
            visible_states,
        )

    def energy(self, samples: torch.Tensor) -> torch.Tensor:
        """Compute free energy for visible samples.

        Args:
            samples: Visible tensor shaped ``(batch_size, n_visible)``.

        Returns:
            Free-energy tensor shaped ``(batch_size,)`` with gradients preserved.

        """

        activations = F.linear(samples, self.W.t(), self.b)

        # Softplus keeps the hidden contribution numerically stable
        s = nn.Softplus()

        h = torch.sum(s(activations), dim=1)
        v = torch.mv(samples, self.a)

        energy = -v - h

        return energy

    def pseudo_likelihood(self, samples: torch.Tensor) -> torch.Tensor:
        """Estimate log pseudo-likelihood by flipping one random visible unit per sample.

        Args:
            samples: Visible tensor shaped ``(batch_size, n_visible)``.

        Returns:
            Scalar log pseudo-likelihood tensor with gradients preserved.

        """

        samples_binary = torch.round(samples)
        energy = self.energy(samples_binary)

        indexes = torch.randint(0, self.n_visible, size=(samples.size(0), 1), device=self.device)
        bits = torch.zeros(samples.size(0), samples.size(1), device=self.device)
        bits = bits.scatter_(1, indexes, 1)

        samples_binary = torch.where(bits == 0, samples_binary, 1 - samples_binary)
        energy1 = self.energy(samples_binary)

        pl = torch.mean(self.n_visible * F.logsigmoid(energy1 - energy))

        return pl

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 128,
        epochs: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update model parameters and metric history with shuffled mini-batch Contrastive Divergence.

        Args:
            dataset: Dataset yielding visible samples and ignored targets.
            batch_size: Maximum number of samples per training batch.
            epochs: Number of complete training passes.

        Returns:
            Final-epoch detached scalar mean squared error and log pseudo-likelihood tensors in that order.

        """

        batches = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        for _ in range(epochs):
            start = time.time()

            mse, pl = 0, 0

            for samples, _ in batches:
                samples = samples.reshape(len(samples), self.n_visible).to(self.device)

                _, _, _, _, visible_states = self.gibbs_sampling(samples)
                visible_states = visible_states.detach()

                cost = torch.mean(self.energy(samples)) - torch.mean(self.energy(visible_states))

                self.optimizer.zero_grad()
                cost.backward()
                self.optimizer.step()

                batch_mse = torch.div(torch.sum(torch.pow(samples - visible_states, 2)), samples.size(0)).detach()
                batch_pl = self.pseudo_likelihood(samples).detach()

                mse += batch_mse
                pl += batch_pl

            mse /= len(batches)
            pl /= len(batches)

            end = time.time()

            self.dump(mse=mse.item(), pl=pl.item(), time=end - start)

        return mse, pl

    def reconstruct(self, dataset: torch.utils.data.Dataset) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct an entire dataset in one unshuffled batch.

        Args:
            dataset: Dataset yielding visible samples and ignored targets.

        Returns:
            Scalar reconstruction error and visible probabilities shaped ``(len(dataset), n_visible)``.

        Notes:
            Autograd tracking is not explicitly disabled.

        """

        mse = 0

        batches = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)

        for samples, _ in batches:
            samples = samples.reshape(len(samples), self.n_visible).to(self.device)

            _, pos_hidden_states = self.hidden_sampling(samples)
            visible_probs, visible_states = self.visible_sampling(pos_hidden_states)

            batch_mse = torch.div(torch.sum(torch.pow(samples - visible_states, 2)), samples.size(0))
            mse += batch_mse

        mse /= len(batches)

        return mse, visible_probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.hidden_sampling(x)

        return x
