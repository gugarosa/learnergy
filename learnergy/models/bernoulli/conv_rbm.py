# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Convolutional Bernoulli-Bernoulli Restricted Boltzmann Machine.

The model uses Contrastive Divergence for training. Calling the model returns hidden probabilities, optionally
processed by max pooling, while still sampling hidden states as part of the forward operation.

References:
    H. Lee, et al.
    Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations.
    Proceedings of the 26th annual international conference on machine learning (2009).

"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.data import DataLoader

import learnergy.utils.exception as e
from learnergy.core.model import Model, _validated_property


class ConvRBM(Model):
    """Implement a convolutional Bernoulli-Bernoulli Restricted Boltzmann Machine."""

    visible_shape = _validated_property("visible_shape", doc="Spatial shape of visible samples.")
    filter_shape = _validated_property(
        "filter_shape",
        lambda self, value: all(
            filter_size < visible_size for filter_size, visible_size in zip(value, self.visible_shape)
        ),
        e.ValueError,
        "`filter_shape` should be smaller than `visible_shape`.",
        doc="Spatial shape of each convolutional filter.",
    )
    hidden_shape = _validated_property("hidden_shape", doc="Spatial shape of hidden feature maps.")
    n_filters = _validated_property(
        "n_filters",
        lambda _, value: value > 0,
        e.ValueError,
        "`n_filters` should be > 0.",
        doc="Number of convolutional filters.",
    )
    n_channels = _validated_property(
        "n_channels",
        lambda _, value: value > 0,
        e.ValueError,
        "`n_channels` should be > 0.",
        doc="Number of visible channels.",
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
    maxpooling = _validated_property(
        "maxpooling",
        lambda _, value: isinstance(value, bool),
        e.ValueError,
        "`maxpooling` should be a boolean.",
        doc="Whether forward outputs are processed by max pooling.",
    )
    W = _validated_property("W", doc="Convolutional filter weights.")
    a = _validated_property("a", doc="Visible-channel bias vector.")
    b = _validated_property("b", doc="Hidden-filter bias vector.")
    optimizer = _validated_property("optimizer", doc="Stochastic gradient descent optimizer.")

    def __init__(
        self,
        visible_shape: tuple[int, int] = (28, 28),
        filter_shape: tuple[int, int] = (7, 7),
        n_filters: int = 5,
        n_channels: int = 1,
        steps: int = 1,
        learning_rate: float = 0.1,
        momentum: float = 0.0,
        decay: float = 0.0,
        maxpooling: bool = False,
        pooling_kernel: int = 2,
        use_gpu: bool = False,
    ) -> None:
        """Initialize a convolutional Bernoulli-Bernoulli RBM.

        Args:
            visible_shape: Height and width of each visible sample.
            filter_shape: Height and width of each convolutional filter.
            n_filters: Number of convolutional filters.
            n_channels: Number of visible channels.
            steps: Number of Contrastive Divergence sampling steps.
            learning_rate: Learning rate used by stochastic gradient descent.
            momentum: Momentum used by stochastic gradient descent.
            decay: Weight decay used by stochastic gradient descent.
            maxpooling: Whether to apply max pooling to forward outputs.
            pooling_kernel: Kernel size used by the max-pooling layer.
            use_gpu: Whether to use CUDA when it is available.

        Raises:
            ValueError: If a shape, count, optimizer value, pooling option, or pooling kernel is invalid.

        """

        super().__init__(use_gpu=use_gpu)

        if len(visible_shape) != 2 or min(visible_shape) <= 0:
            raise e.ValueError("`visible_shape` should contain two positive values.")
        if len(filter_shape) != 2 or min(filter_shape) <= 0:
            raise e.ValueError("`filter_shape` should contain two positive values.")
        if pooling_kernel <= 0:
            raise e.ValueError("`pooling_kernel` should be > 0.")

        self.visible_shape = visible_shape
        self.filter_shape = filter_shape
        self.hidden_shape = (
            visible_shape[0] - filter_shape[0] + 1,
            visible_shape[1] - filter_shape[1] + 1,
        )

        self.n_filters = n_filters
        self.n_channels = n_channels

        self.steps = steps
        self.lr = learning_rate
        self.momentum = momentum
        self.decay = decay

        self.maxpooling = maxpooling
        self.maxpol2d = nn.MaxPool2d(kernel_size=pooling_kernel, stride=2, padding=1) if maxpooling else None

        self.W = nn.Parameter(torch.randn(n_filters, n_channels, filter_shape[0], filter_shape[1]) * 0.01)
        self.a = nn.Parameter(torch.zeros(n_channels))
        self.b = nn.Parameter(torch.zeros(n_filters))

        self.to(self.device)
        self.optimizer = opt.SGD(self.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)

    def hidden_sampling(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample hidden feature maps conditioned on visible samples.

        Args:
            v: Visible tensor shaped ``(batch_size, n_channels, *visible_shape)``.

        Returns:
            Hidden probabilities and states shaped ``(batch_size, n_filters, *hidden_shape)``, in that order.

        """

        activations = F.conv2d(v, self.W, bias=self.b)
        probs = torch.sigmoid(activations)
        states = torch.bernoulli(probs)

        return probs, states

    def visible_sampling(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample visible units conditioned on hidden feature maps.

        Args:
            h: Hidden tensor shaped ``(batch_size, n_filters, *hidden_shape)``.

        Returns:
            Visible probabilities and states shaped ``(batch_size, n_channels, *visible_shape)``, in that order.

        """

        activations = F.conv_transpose2d(h, self.W, bias=self.a)
        probs = torch.sigmoid(activations)
        states = torch.bernoulli(probs)

        return probs, states

    def gibbs_sampling(
        self, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run Contrastive Divergence sampling from visible samples.

        Args:
            v: Visible tensor shaped ``(batch_size, n_channels, *visible_shape)``.

        Returns:
            Positive hidden probabilities and states, negative hidden probabilities and states, and visible states.

        """

        pos_hidden_probs, pos_hidden_states = self.hidden_sampling(v)
        neg_hidden_states = pos_hidden_states

        for _ in range(self.steps):
            _, visible_states = self.visible_sampling(neg_hidden_states)
            neg_hidden_probs, neg_hidden_states = self.hidden_sampling(visible_states)

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
            samples: Visible tensor shaped ``(batch_size, n_channels, *visible_shape)``.

        Returns:
            Free-energy tensor shaped ``(batch_size,)`` with gradients preserved.

        """

        activations = F.conv2d(samples, self.W, bias=self.b)

        # Softplus keeps the hidden contribution numerically stable
        s = nn.Softplus()

        h = torch.sum(s(activations), dim=(1, 2, 3))
        v = torch.sum(samples * self.a.view(1, -1, 1, 1), dim=(1, 2, 3))

        energy = -v - h

        return energy

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 128,
        epochs: int = 10,
    ) -> torch.Tensor:
        """Update model parameters and metric history with shuffled mini-batch Contrastive Divergence.

        Args:
            dataset: Dataset yielding image-like samples and ignored targets.
            batch_size: Maximum number of samples per training batch.
            epochs: Number of complete training passes.

        Returns:
            Final-epoch detached scalar mean squared error tensor.

        """

        batches = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        for _ in range(epochs):
            start = time.time()

            mse = 0

            for samples, _ in batches:
                samples = samples.reshape(
                    len(samples),
                    self.n_channels,
                    self.visible_shape[0],
                    self.visible_shape[1],
                ).to(self.device)

                _, _, _, _, visible_states = self.gibbs_sampling(samples)
                visible_states = visible_states.detach()

                cost = torch.mean(self.energy(samples)) - torch.mean(self.energy(visible_states))

                self.optimizer.zero_grad()
                cost.backward()
                self.optimizer.step()

                batch_size = samples.size(0)
                batch_mse = torch.div(torch.sum(torch.pow(samples - visible_states, 2)), batch_size).detach()

                mse += batch_mse

            mse /= len(batches)

            end = time.time()

            self.dump(mse=mse.item(), time=end - start)

        return mse

    def reconstruct(self, dataset: torch.utils.data.Dataset) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct an entire dataset in one unshuffled batch.

        Args:
            dataset: Dataset yielding image-like samples and ignored targets.

        Returns:
            Scalar reconstruction error and probabilities shaped ``(len(dataset), n_channels, *visible_shape)``.

        Notes:
            Autograd tracking is not explicitly disabled.

        """

        mse = 0

        batch_size = len(dataset)
        batches = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        for samples, _ in batches:
            samples = samples.reshape(
                len(samples),
                self.n_channels,
                self.visible_shape[0],
                self.visible_shape[1],
            ).to(self.device)

            _, pos_hidden_states = self.hidden_sampling(samples)
            visible_probs, visible_states = self.visible_sampling(pos_hidden_states)

            batch_mse = torch.div(torch.sum(torch.pow(samples - visible_states, 2)), batch_size)
            mse += batch_mse

        mse /= len(batches)

        return mse, visible_probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.hidden_sampling(x)
        if self.maxpooling:
            x = self.maxpol2d(x)

        return x
