# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Provide convolutional RBMs with Gaussian visible units.

Training reshapes samples to ``(batch_size, n_channels, *visible_shape)`` and standardizes them across the batch
dimension when normalization is enabled. Forward propagation uses the same optional standardization without detaching
the graph, then returns ReLU6 hidden values with optional max pooling.

"""

import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from learnergy.core.model import _validated_property
from learnergy.models.bernoulli.conv_rbm import ConvRBM
from learnergy.models.gaussian._normalization import _standardize


class GaussianConvRBM(ConvRBM):
    """Implement a convolutional RBM with Gaussian visible units."""

    normalize = _validated_property(
        "normalize", doc="Whether batches are standardized and visible outputs use raw activations."
    )

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
        normalize: bool = True,
    ) -> None:
        """Initialize a Gaussian convolutional RBM.

        Args:
            visible_shape: Height and width of each visible sample.
            filter_shape: Height and width of each convolutional filter.
            n_filters: Number of hidden convolutional filters.
            n_channels: Number of visible channels.
            steps: Number of Gibbs sampling steps.
            learning_rate: Learning rate used by SGD.
            momentum: Momentum used by SGD.
            decay: Weight decay used by SGD.
            maxpooling: Whether forward propagation applies max pooling.
            pooling_kernel: Kernel size used when max pooling is enabled.
            use_gpu: Whether to select CUDA when it is available.
            normalize: Whether to standardize training and forward batches.

        """

        super().__init__(
            visible_shape=visible_shape,
            filter_shape=filter_shape,
            n_filters=n_filters,
            n_channels=n_channels,
            steps=steps,
            learning_rate=learning_rate,
            momentum=momentum,
            decay=decay,
            maxpooling=maxpooling,
            pooling_kernel=pooling_kernel,
            use_gpu=use_gpu,
        )

        self.normalize = normalize

    def hidden_sampling(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate convolutional hidden values.

        Args:
            v: Visible values shaped ``(batch_size, n_channels, *visible_shape)``.

        Returns:
            ReLU6 values followed by raw activations, each shaped ``(batch_size, n_filters, *hidden_shape)``.

        """

        activations = F.conv2d(v, self.W, bias=self.b)
        return F.relu6(activations), activations

    def visible_sampling(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate convolutional visible values.

        The detached output contains raw activations when ``normalize`` is true and sigmoid values otherwise.

        Args:
            h: Hidden values shaped ``(batch_size, n_filters, *hidden_shape)``.

        Returns:
            Detached output then raw activations, shaped ``(batch_size, n_channels, *visible_shape)`` in that order.

        """

        activations = F.conv_transpose2d(h, self.W, bias=self.a)
        probs = activations if self.normalize else torch.sigmoid(activations)
        return probs.detach(), activations

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 128,
        epochs: int = 10,
        log: bool = False,
    ) -> torch.Tensor:
        """Fit the model with shuffled contrastive-divergence batches.

        Each dataset item must contain a sample and an ignored target. Float copies of each epoch's MSE and elapsed time
        are appended to history.

        Args:
            dataset: Dataset yielding ``(sample, target)`` pairs.
            batch_size: Maximum number of samples in each batch.
            epochs: Number of training passes over the dataset.
            log: Accepted without changing metric collection or logging.

        Returns:
            Final scalar reconstruction MSE tensor.

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

                if self.normalize:
                    samples = _standardize(samples)

                _, _, _, _, visible_states = self.gibbs_sampling(samples)
                visible_states = visible_states.detach()

                cost = self.energy(samples).mean() - self.energy(visible_states).mean()

                self.optimizer.zero_grad()
                cost.backward()
                self.optimizer.step()

                mse += ((samples - visible_states) ** 2).sum() / samples.size(0)

            mse /= len(batches)
            self.dump(mse=mse.item(), time=time.time() - start)

        return mse

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            x = _standardize(x)

        x, _ = self.hidden_sampling(x)
        if self.maxpooling:
            x = self.maxpol2d(x)

        return x


class GaussianConvRBM4Deep(GaussianConvRBM):
    """Provide the Gaussian convolutional RBM variant used by deeper models."""

    def visible_sampling(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate visible values for a deeper convolutional layer.

        The detached output contains raw activations when ``normalize`` is true and ReLU6 values otherwise.

        Args:
            h: Hidden values shaped ``(batch_size, n_filters, *hidden_shape)``.

        Returns:
            Detached output then raw activations, shaped ``(batch_size, n_channels, *visible_shape)`` in that order.

        """

        activations = F.conv_transpose2d(h, self.W, bias=self.a)
        probs = activations.detach() if self.normalize else F.relu6(activations).detach()
        return probs, activations
