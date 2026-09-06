# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Provide dense Gaussian-visible Restricted Boltzmann Machines.

``GaussianRBM`` standardizes each training batch when normalization is enabled and uses unit visible variance.
Forward input standardization is separately controlled and detaches the standardized tensor before hidden sampling.
The ReLU and SeLU variants replace Bernoulli hidden states with deterministic activations. ``VarianceGaussianRBM``
instead learns a visible scale whose squared value plus machine epsilon defines the variance.

References:
    K. Cho, A. Ilin, T. Raiko. Improved learning of Gaussian-Bernoulli restricted Boltzmann machines.
    International conference on artificial neural networks (2011).
    G. Hinton. A practical guide to training restricted Boltzmann machines.
    Neural networks: Tricks of the trade (2012).
    G. Klambauer et al. Self-normalizing neural networks. Proceedings, NIPS (2017).

"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from learnergy.core.model import _validated_property
from learnergy.models.bernoulli.rbm import RBM
from learnergy.models.gaussian._normalization import _standardize


class GaussianRBM(RBM):
    """Implement an RBM with standardized Gaussian visible units."""

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
        """Initialize a Gaussian-Bernoulli RBM.

        Args:
            n_visible: Number of visible units.
            n_hidden: Number of hidden units.
            steps: Number of Gibbs sampling steps.
            learning_rate: Learning rate used by SGD.
            momentum: Momentum used by SGD.
            decay: Weight decay used by SGD.
            temperature: Positive temperature used by scaled sampling.
            use_gpu: Whether to select CUDA when it is available.
            normalize: Whether to standardize each training and reconstruction batch.
            input_normalize: Whether to standardize and detach inputs passed through ``forward``.

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

    def energy(self, samples: torch.Tensor) -> torch.Tensor:
        """Calculate the free energy of visible samples.

        Args:
            samples: Visible samples shaped ``(batch_size, n_visible)``.

        Returns:
            Free energy for each sample shaped ``(batch_size,)``.

        """

        activations = F.linear(samples, self.W.t(), self.b)

        h = torch.sum(F.softplus(activations), dim=1)
        v = 0.5 * torch.sum((samples - self.a) ** 2, dim=1)

        energy = v - h

        return energy

    def visible_sampling(self, h: torch.Tensor, scale: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the Gaussian visible conditional values.

        Args:
            h: Hidden values shaped ``(batch_size, n_hidden)``.
            scale: Whether to divide visible activations by the temperature.

        Returns:
            Sigmoid probabilities followed by deterministic visible states, each shaped ``(batch_size, n_visible)``.

        """

        activations = F.linear(h, self.W, self.a)

        if scale:
            states = torch.div(activations, self.T)
        else:
            states = activations

        probs = torch.sigmoid(states)

        return probs, states

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 128,
        epochs: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fit the model with non-shuffled contrastive-divergence batches.

        Each dataset item must contain a sample and an ignored target. When ``normalize`` is true, each batch is
        standardized and detached before flattening. Float copies of each epoch's metrics are appended to history.

        Args:
            dataset: Dataset yielding ``(sample, target)`` pairs.
            batch_size: Maximum number of samples in each batch.
            epochs: Number of training passes over the dataset.

        Returns:
            Final scalar MSE tensor followed by the final scalar log pseudo-likelihood tensor.

        """

        batches = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        for _ in range(epochs):
            start = time.time()

            mse = 0
            pl = 0

            for samples, _ in batches:
                if self.normalize:
                    samples = _standardize(samples).detach()

                samples = samples.reshape(len(samples), self.n_visible).to(self.device)

                _, _, _, _, visible_states = self.gibbs_sampling(samples)
                visible_states = visible_states.detach()

                cost = torch.mean(self.energy(samples)) - torch.mean(self.energy(visible_states))

                self.optimizer.zero_grad()
                cost.backward()
                self.optimizer.step()

                batch_size = samples.size(0)

                batch_mse = torch.div(torch.sum(torch.pow(samples - visible_states, 2)), batch_size).detach()
                batch_pl = self.pseudo_likelihood(samples).detach()

                mse += batch_mse
                pl += batch_pl

            mse /= len(batches)
            pl /= len(batches)

            end = time.time()

            self.dump(mse=mse.item(), pl=pl.item(), time=end - start)

        return mse, pl

    def reconstruct(self, dataset: torch.utils.data.Dataset) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct all dataset samples in one non-shuffled batch.

        Each dataset item must contain a sample and an ignored target. Input samples are standardized and detached when
        ``normalize`` is true.

        Args:
            dataset: Dataset yielding ``(sample, target)`` pairs.

        Returns:
            Scalar reconstruction MSE tensor followed by visible probabilities shaped ``(len(dataset), n_visible)``.

        """

        mse = 0
        batch_size = len(dataset)

        batches = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        for samples, _ in batches:
            if self.normalize:
                samples = _standardize(samples).detach()

            samples = samples.reshape(len(samples), self.n_visible).to(self.device)

            _, pos_hidden_states = self.hidden_sampling(samples)
            visible_probs, visible_states = self.visible_sampling(pos_hidden_states)

            batch_mse = torch.div(torch.sum(torch.pow(samples - visible_states, 2)), batch_size)
            mse += batch_mse

        mse /= len(batches)

        return mse, visible_probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_normalize:
            x = _standardize(x).detach()

        x, _ = self.hidden_sampling(x)

        return x


class GaussianReluRBM(GaussianRBM):
    """Implement a Gaussian-visible RBM with deterministic ReLU hidden units."""

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
        normalize: bool = True,
        input_normalize: bool = True,
    ) -> None:
        """Initialize a Gaussian-ReLU RBM.

        Args:
            n_visible: Number of visible units.
            n_hidden: Number of hidden units.
            steps: Number of Gibbs sampling steps.
            learning_rate: Learning rate used by SGD.
            momentum: Momentum used by SGD.
            decay: Weight decay used by SGD.
            temperature: Positive temperature used by scaled sampling.
            use_gpu: Whether to select CUDA when it is available.
            normalize: Whether to standardize each training and reconstruction batch.
            input_normalize: Whether to standardize and detach inputs passed through ``forward``.

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
            normalize,
            input_normalize,
        )

    def hidden_sampling(self, v: torch.Tensor, scale: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate deterministic ReLU hidden values.

        Args:
            v: Visible values shaped ``(batch_size, n_visible)``.
            scale: Whether to divide hidden activations by the temperature.

        Returns:
            ReLU activations followed by the same tensor as hidden states, shaped ``(batch_size, n_hidden)``.

        """

        activations = F.linear(v, self.W.t(), self.b)

        if scale:
            probs = F.relu(torch.div(activations, self.T))
        else:
            probs = F.relu(activations)

        states = probs

        return probs, states


class GaussianSeluRBM(GaussianRBM):
    """Implement a Gaussian-visible RBM with deterministic SeLU hidden units."""

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
        normalize: bool = False,
        input_normalize: bool = True,
    ) -> None:
        """Initialize a Gaussian-SeLU RBM.

        Args:
            n_visible: Number of visible units.
            n_hidden: Number of hidden units.
            steps: Number of Gibbs sampling steps.
            learning_rate: Learning rate used by SGD.
            momentum: Momentum used by SGD.
            decay: Weight decay used by SGD.
            temperature: Positive temperature used by scaled sampling.
            use_gpu: Whether to select CUDA when it is available.
            normalize: Whether to standardize each training and reconstruction batch.
            input_normalize: Whether to standardize and detach inputs passed through ``forward``.

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
            normalize,
            input_normalize,
        )

    def hidden_sampling(self, v: torch.Tensor, scale: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate deterministic SeLU hidden values.

        Args:
            v: Visible values shaped ``(batch_size, n_visible)``.
            scale: Whether to divide hidden activations by the temperature.

        Returns:
            SeLU activations followed by the same tensor as hidden states, shaped ``(batch_size, n_hidden)``.

        """

        activations = F.linear(v, self.W.t(), self.b)

        if scale:
            probs = F.selu(torch.div(activations, self.T))
        else:
            probs = F.selu(activations)

        states = probs

        return probs, states


class VarianceGaussianRBM(RBM):
    """Implement a Gaussian-visible RBM with a learned visible scale."""

    sigma = _validated_property("sigma", doc="Learnable visible scale whose square determines the variance.")

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
        """Initialize a Gaussian-Bernoulli RBM with learned visible variance.

        Args:
            n_visible: Number of visible units.
            n_hidden: Number of hidden units.
            steps: Number of Gibbs sampling steps.
            learning_rate: Learning rate used by SGD.
            momentum: Momentum used by SGD.
            decay: Weight decay used by SGD.
            temperature: Positive temperature used by scaled hidden sampling.
            use_gpu: Whether to select CUDA when it is available.

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
        self.to(self.device)
        self.optimizer.add_param_group({"params": self.sigma})

    def hidden_sampling(self, v: torch.Tensor, scale: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample Bernoulli hidden states using the learned visible variance.

        Args:
            v: Visible values shaped ``(batch_size, n_visible)``.
            scale: Whether to divide hidden activations by the temperature.

        Returns:
            Hidden probabilities followed by sampled states, each shaped ``(batch_size, n_hidden)``.

        """

        variance = self.sigma.square() + torch.finfo(v.dtype).eps
        activations = F.linear(v / variance, self.W.t(), self.b)

        if scale:
            probs = torch.sigmoid(torch.div(activations, self.T))
        else:
            probs = torch.sigmoid(activations)

        states = torch.bernoulli(probs)

        return probs, states

    def visible_sampling(self, h: torch.Tensor, scale: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample Gaussian visible states using the learned visible variance.

        Args:
            h: Hidden values shaped ``(batch_size, n_hidden)``.
            scale: Accepted for sampling API compatibility without changing the visible distribution.

        Returns:
            Conditional means followed by sampled states, each shaped ``(batch_size, n_visible)``.

        """

        activations = F.linear(h, self.W, self.a)
        variance = self.sigma.square() + torch.finfo(activations.dtype).eps
        std = variance.sqrt().expand_as(activations)
        states = torch.normal(activations, std)

        return activations, states

    def energy(self, samples: torch.Tensor) -> torch.Tensor:
        """Calculate free energy using the learned visible variance.

        Args:
            samples: Visible samples shaped ``(batch_size, n_visible)``.

        Returns:
            Free energy for each sample shaped ``(batch_size,)``.

        """

        variance = self.sigma.square() + torch.finfo(samples.dtype).eps
        activations = F.linear(samples / variance, self.W.t(), self.b)

        h = torch.sum(F.softplus(activations), dim=1)
        v = torch.sum((samples - self.a).square() / (2 * variance), dim=1)

        energy = v - h

        return energy


class GaussianRBM4deep(GaussianRBM):
    """Provide the Gaussian RBM training default used by deep models."""

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 128,
        epochs: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fit the model with a one-epoch default.

        Behavior, normalization, metric persistence, and return order otherwise match ``GaussianRBM.fit``.

        Args:
            dataset: Dataset yielding ``(sample, target)`` pairs.
            batch_size: Maximum number of samples in each batch.
            epochs: Number of training passes over the dataset.

        Returns:
            Final scalar MSE tensor followed by the final scalar log pseudo-likelihood tensor.

        """

        return super().fit(dataset, batch_size=batch_size, epochs=epochs)


class GaussianReluRBM4deep(GaussianReluRBM, GaussianRBM4deep):
    """Provide the Gaussian-ReLU RBM variant used by deep models."""
