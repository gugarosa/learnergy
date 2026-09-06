# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Provide a convolutional Deep Belief Network.

Greedy training encodes later-layer minibatches under ``torch.no_grad()`` and stores detached CPU features with their
original targets. Forward propagation chains hidden samplers directly without Gaussian input standardization and applies
each configured pooling layer, while reconstruction deliberately traverses the stack without pooling or unpooling.

"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import learnergy.utils.exception as e
from learnergy.core.model import Model, _validated_property
from learnergy.models.bernoulli.conv_rbm import ConvRBM
from learnergy.models.gaussian.gaussian_conv_rbm import GaussianConvRBM, GaussianConvRBM4Deep

MODELS = {
    "bernoulli": ConvRBM,
    "gaussian": GaussianConvRBM,
    "gaussiandeep": GaussianConvRBM4Deep,
}


class ConvDBN(Model):
    """Stack convolutional RBMs for greedy layer-wise training and inference."""

    visible_shape = _validated_property("visible_shape", doc="Height and width of the input samples.")
    filter_shape = _validated_property("filter_shape", doc="Convolutional filter shapes in layer order.")
    n_filters = _validated_property("n_filters", doc="Filter counts in layer order.")
    n_channels = _validated_property(
        "n_channels",
        lambda _, value: value > 0,
        e.ValueError,
        "`n_channels` should be greater than 0.",
        doc="Number of channels in the input samples.",
    )
    n_layers = _validated_property(
        "n_layers",
        lambda _, value: value > 0,
        e.ValueError,
        "`n_layers` should be greater than 0.",
        doc="Number of stacked convolutional RBM layers.",
    )
    steps = _validated_property(
        "steps",
        lambda self, value: len(value) == self.n_layers,
        e.SizeError,
        "`steps` should match the number of layers.",
        doc="Gibbs sampling step counts in layer order.",
    )
    lr = _validated_property(
        "lr",
        lambda self, value: len(value) == self.n_layers,
        e.SizeError,
        "`lr` should match the number of layers.",
        doc="Learning rates used when constructing each RBM.",
    )
    momentum = _validated_property(
        "momentum",
        lambda self, value: len(value) == self.n_layers,
        e.SizeError,
        "`momentum` should match the number of layers.",
        doc="Momentum settings used when constructing each RBM.",
    )
    decay = _validated_property(
        "decay",
        lambda self, value: len(value) == self.n_layers,
        e.SizeError,
        "`decay` should match the number of layers.",
        doc="Weight-decay settings used when constructing each RBM.",
    )
    models = _validated_property("models", doc="Registered convolutional RBM layers in training order.")

    def __init__(
        self,
        model: str = "bernoulli",
        visible_shape: tuple[int, int] = (28, 28),
        filter_shape: tuple[tuple[int, int], ...] = ((7, 7),),
        n_filters: tuple[int, ...] = (16,),
        n_channels: int = 1,
        steps: tuple[int, ...] = (1,),
        learning_rate: tuple[float, ...] = (0.1,),
        momentum: tuple[float, ...] = (0.0,),
        decay: tuple[float, ...] = (0.0,),
        maxpooling: bool | tuple[bool, ...] = (False, False),
        pooling_kernel: int | tuple[int, ...] = (2, 2),
        use_gpu: bool = False,
    ) -> None:
        """Initialize a convolutional Deep Belief Network.

        Scalar pooling settings are repeated for every layer. Sequence settings are truncated or padded with ``False``
        for pooling flags and ``2`` for pooling kernels. Layers after the first always use ``GaussianConvRBM4Deep``.

        Args:
            model: Model name used by the first layer.
            visible_shape: Height and width of input samples.
            filter_shape: Convolutional filter shapes in layer order.
            n_filters: Filter counts in layer order.
            n_channels: Number of channels in input samples.
            steps: Gibbs sampling step counts in layer order.
            learning_rate: Learning rates in layer order.
            momentum: SGD momentum values in layer order.
            decay: SGD weight-decay values in layer order.
            maxpooling: Pooling flags supplied as one value or in layer order.
            pooling_kernel: Pooling kernel sizes supplied as one value or in layer order.
            use_gpu: Whether to select CUDA when it is available.

        Raises:
            ValueError: ``model``, ``n_filters``, or ``n_channels`` is invalid.
            learnergy.utils.exception.SizeError: ``filter_shape`` does not contain one value per layer.

        """

        super().__init__(use_gpu=use_gpu)

        if model not in MODELS:
            raise e.ValueError(f"`model` contains unknown model type `{model}`.")
        if not n_filters or any(value <= 0 for value in n_filters):
            raise e.ValueError("`n_filters` should contain only positive values.")

        self.visible_shape = visible_shape
        self.filter_shape = tuple(filter_shape)
        self.n_filters = tuple(n_filters)
        self.n_channels = n_channels
        self.n_layers = len(self.n_filters)
        self.steps = tuple(steps)
        self.lr = tuple(learning_rate)
        self.momentum = tuple(momentum)
        self.decay = tuple(decay)

        if len(self.filter_shape) != self.n_layers:
            raise e.SizeError("`filter_shape` should match the number of layers.")

        if isinstance(maxpooling, bool):
            maxpooling = (maxpooling,) * self.n_layers
        else:
            maxpooling = tuple(maxpooling)[: self.n_layers]
            maxpooling += (False,) * (self.n_layers - len(maxpooling))

        if isinstance(pooling_kernel, int):
            pooling_kernel = (pooling_kernel,) * self.n_layers
        else:
            pooling_kernel = tuple(pooling_kernel)[: self.n_layers]
            pooling_kernel += (2,) * (self.n_layers - len(pooling_kernel))

        self.maxpooling = maxpooling
        self.pooling_kernel = pooling_kernel
        self.maxpol2d = []
        self.models = nn.ModuleList()

        layer_shape = visible_shape
        layer_channels = n_channels
        for i in range(self.n_layers):
            model_class = MODELS[model] if i == 0 else MODELS["gaussiandeep"]
            layer = model_class(
                visible_shape=layer_shape,
                filter_shape=self.filter_shape[i],
                n_filters=self.n_filters[i],
                n_channels=layer_channels,
                steps=self.steps[i],
                learning_rate=self.lr[i],
                momentum=self.momentum[i],
                decay=self.decay[i],
                maxpooling=self.maxpooling[i],
                pooling_kernel=self.pooling_kernel[i],
                use_gpu=use_gpu,
            )
            self.models.append(layer)
            self.maxpol2d.append(layer.maxpol2d)

            layer_shape = layer.hidden_shape
            if self.maxpooling[i]:
                kernel = self.pooling_kernel[i]
                layer_shape = tuple((size + 2 - kernel) // 2 + 1 for size in layer_shape)
            layer_channels = self.n_filters[i]

        self.to(self.device)

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 128,
        epochs: tuple[int, ...] = (10, 10),
        log: bool = True,
    ) -> list[torch.Tensor]:
        """Fit each convolutional RBM layer greedily.

        ``epochs`` is truncated to the layer count or extended by repeating its final value. The first layer delegates
        dataset access to its RBM. Later layers shuffle the original dataset, encode each minibatch through preceding
        layers without gradients, and fit the current layer on detached CPU features for one epoch. Layer parameters,
        optimizers, and metric histories persist after this call.

        Args:
            dataset: Dataset yielding ``(sample, target)`` pairs.
            batch_size: Maximum number of samples in each batch.
            epochs: Training epoch counts in layer order.
            log: Value forwarded to Gaussian layer ``fit`` methods without changing their current logging behavior.

        Returns:
            Final scalar MSE tensors in layer order.

        Raises:
            learnergy.utils.exception.SizeError: ``epochs`` is empty.

        """

        epochs = tuple(epochs)
        if not epochs:
            raise e.SizeError("`epochs` should contain at least one value.")
        epochs = epochs[: self.n_layers] + (epochs[-1],) * max(0, self.n_layers - len(epochs))

        mse = []

        for i, model in enumerate(self.models):
            if i == 0:
                model_mse = model.fit(
                    dataset,
                    batch_size=batch_size,
                    epochs=epochs[i],
                    **({"log": log} if isinstance(model, GaussianConvRBM) else {}),
                )
            else:
                batches = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                for _ in range(epochs[i]):
                    model_mse = 0
                    for samples, labels in batches:
                        samples = samples.reshape(
                            len(samples),
                            self.n_channels,
                            self.visible_shape[0],
                            self.visible_shape[1],
                        ).to(self.device)
                        with torch.no_grad():
                            for previous_model in self.models[:i]:
                                samples, _ = previous_model.hidden_sampling(samples)
                                if previous_model.maxpooling:
                                    samples = previous_model.maxpol2d(samples)

                        encoded = TensorDataset(samples.detach().cpu(), labels)
                        batch_mse = model.fit(
                            encoded,
                            batch_size=len(samples),
                            epochs=1,
                            **({"log": log} if isinstance(model, GaussianConvRBM) else {}),
                        )
                        model_mse += batch_mse

                    model_mse /= len(batches)

            mse.append(model_mse)

        return mse

    def reconstruct(self, dataset: torch.utils.data.Dataset) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct all dataset samples through the unpooled convolutional stack.

        The method processes one non-shuffled batch and does not disable gradient tracking.

        Args:
            dataset: Dataset yielding ``(sample, target)`` pairs.

        Returns:
            Scalar MSE tensor then visible output shaped ``(len(dataset), n_channels, *visible_shape)``.

        """

        batch_size = len(dataset)
        batches = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        for samples, _ in batches:
            samples = samples.reshape(
                batch_size,
                self.n_channels,
                self.visible_shape[0],
                self.visible_shape[1],
            ).to(self.device)

            hidden_probs = samples
            for model in self.models:
                hidden_probs, _ = model.hidden_sampling(hidden_probs)

            visible_probs = hidden_probs
            for model in reversed(self.models):
                visible_probs, visible_states = model.visible_sampling(visible_probs)

            mse = ((samples - visible_states) ** 2).sum() / batch_size

        return mse, visible_probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for model in self.models:
            x, _ = model.hidden_sampling(x)
            if model.maxpooling:
                x = model.maxpol2d(x)

        return x
