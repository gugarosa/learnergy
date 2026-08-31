"""Convolutional Deep Belief Network."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import learnergy.utils.exception as e
from learnergy.core.model import Model, _validated_property
from learnergy.models.bernoulli.conv_rbm import ConvRBM
from learnergy.models.gaussian.gaussian_conv_rbm import (
    GaussianConvRBM,
    GaussianConvRBM4Deep,
)

MODELS = {
    "bernoulli": ConvRBM,
    "gaussian": GaussianConvRBM,
    "gaussiandeep": GaussianConvRBM4Deep,
}


class ConvDBN(Model):
    """Stack convolutional RBMs and train them layer by layer."""

    visible_shape = _validated_property("visible_shape")
    filter_shape = _validated_property("filter_shape")
    n_filters = _validated_property("n_filters")
    n_channels = _validated_property(
        "n_channels",
        lambda _, value: value > 0,
        e.ValueError,
        "`n_channels` should be > 0",
    )
    n_layers = _validated_property(
        "n_layers", lambda _, value: value > 0, e.ValueError, "`n_layers` should be > 0"
    )
    steps = _validated_property(
        "steps",
        lambda self, value: len(value) == self.n_layers,
        e.SizeError,
        "`steps` should match the number of layers",
    )
    lr = _validated_property(
        "lr",
        lambda self, value: len(value) == self.n_layers,
        e.SizeError,
        "`lr` should match the number of layers",
    )
    momentum = _validated_property(
        "momentum",
        lambda self, value: len(value) == self.n_layers,
        e.SizeError,
        "`momentum` should match the number of layers",
    )
    decay = _validated_property(
        "decay",
        lambda self, value: len(value) == self.n_layers,
        e.SizeError,
        "`decay` should match the number of layers",
    )
    models = _validated_property("models")

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
        """Initialize a convolutional Deep Belief Network."""

        super().__init__(use_gpu=use_gpu)

        if model not in MODELS:
            raise e.ValueError(f"unknown model type: {model}")
        if not n_filters or any(value <= 0 for value in n_filters):
            raise e.ValueError("`n_filters` should contain positive values")

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
            raise e.SizeError("`filter_shape` should match the number of layers")

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
                layer_shape = tuple(
                    (size + 2 - kernel) // 2 + 1 for size in layer_shape
                )
            layer_channels = self.n_filters[i]

        self.to(self.device)

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 128,
        epochs: tuple[int, ...] = (10, 10),
        log: bool = True,
    ) -> list[torch.Tensor]:
        """Fit each convolutional RBM layer."""

        epochs = tuple(epochs)
        if not epochs:
            raise e.SizeError("`epochs` should contain at least one value")
        epochs = epochs[: self.n_layers] + (epochs[-1],) * max(
            0, self.n_layers - len(epochs)
        )

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
                            **(
                                {"log": log}
                                if isinstance(model, GaussianConvRBM)
                                else {}
                            ),
                        )
                        model_mse += batch_mse

                    model_mse /= len(batches)

            mse.append(model_mse)

        return mse

    def reconstruct(
        self, dataset: torch.utils.data.Dataset
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct a dataset through an unpooled convolutional stack."""

        batch_size = len(dataset)
        batches = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

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
        """Return the representation produced by the final layer."""

        for model in self.models:
            x, _ = model.hidden_sampling(x)
            if model.maxpooling:
                x = model.maxpol2d(x)

        return x
