# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Provide a Deep Belief Network trained by greedy layer-wise pretraining.

The first RBM consumes the supplied dataset directly. Each later RBM is trained from shuffled minibatches encoded by
all preceding layers under ``torch.no_grad()``, with detached CPU features paired with their original targets. Forward
propagation chains the first value returned by each hidden sampler with ordinary gradient tracking and does not call the
layers' ``forward`` methods.

"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import learnergy.utils.exception as e
from learnergy.core.model import Model, _validated_property
from learnergy.models.bernoulli.dropout_rbm import DropoutRBM
from learnergy.models.bernoulli.e_dropout_rbm import EDropoutRBM
from learnergy.models.bernoulli.rbm import RBM
from learnergy.models.extra.sigmoid_rbm import SigmoidRBM, SigmoidRBM4Deep
from learnergy.models.gaussian.gaussian_rbm import (
    GaussianRBM,
    GaussianRBM4deep,
    GaussianReluRBM,
    GaussianReluRBM4deep,
    GaussianSeluRBM,
    VarianceGaussianRBM,
)

MODELS = {
    "bernoulli": RBM,
    "dropout": DropoutRBM,
    "e_dropout": EDropoutRBM,
    "gaussian": GaussianRBM,
    "gaussian4deep": GaussianRBM4deep,
    "gaussian_relu": GaussianReluRBM,
    "gaussian_relu4deep": GaussianReluRBM4deep,
    "gaussian_selu": GaussianSeluRBM,
    "sigmoid": SigmoidRBM,
    "sigmoid4deep": SigmoidRBM4Deep,
    "variance_gaussian": VarianceGaussianRBM,
}


class DBN(Model):
    """Stack RBMs for greedy layer-wise training and inference."""

    n_visible = _validated_property(
        "n_visible",
        lambda _, value: value > 0,
        e.ValueError,
        "`n_visible` should be greater than 0.",
        doc="Number of visible units in the first layer.",
    )
    n_hidden = _validated_property("n_hidden", doc="Hidden-unit counts in layer order.")
    n_layers = _validated_property(
        "n_layers",
        lambda _, value: value > 0,
        e.ValueError,
        "`n_layers` should be greater than 0.",
        doc="Number of stacked RBM layers.",
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
    T = _validated_property(
        "T",
        lambda self, value: len(value) == self.n_layers,
        e.SizeError,
        "`T` should match the number of layers.",
        doc="Temperatures used when constructing each RBM.",
    )
    models = _validated_property("models", doc="Registered RBM layers in training order.")

    def __init__(
        self,
        model: str | tuple[str, ...] = ("gaussian",),
        n_visible: int = 128,
        n_hidden: tuple[int, ...] = (128,),
        steps: tuple[int, ...] = (1,),
        learning_rate: tuple[float, ...] = (0.1,),
        momentum: tuple[float, ...] = (0.0,),
        decay: tuple[float, ...] = (0.0,),
        temperature: tuple[float, ...] = (1.0,),
        use_gpu: bool = False,
        normalize: bool = True,
        input_normalize: bool = True,
    ) -> None:
        """Initialize a Deep Belief Network.

        A single model name configures the first layer, while omitted later names become ``sigmoid4deep``. Gaussian
        names after the first layer are replaced by their deep-training variants.

        Args:
            model: Model name for the first layer or model names in layer order.
            n_visible: Number of visible units in the first layer.
            n_hidden: Hidden-unit counts in layer order.
            steps: Gibbs sampling step counts in layer order.
            learning_rate: Learning rates in layer order.
            momentum: SGD momentum values in layer order.
            decay: SGD weight-decay values in layer order.
            temperature: Sampling temperatures in layer order.
            use_gpu: Whether to select CUDA when it is available.
            normalize: Whether Gaussian layer fit and reconstruct methods standardize their input batches.
            input_normalize: Whether Gaussian layer ``forward`` methods standardize and detach their inputs.

        Raises:
            ValueError: A unit count or model name is invalid.
            learnergy.utils.exception.SizeError: A layer-wise sequence has an invalid length.

        """

        super().__init__(use_gpu=use_gpu)

        if not n_hidden or any(value <= 0 for value in n_hidden):
            raise e.ValueError("`n_hidden` should contain only positive values.")

        self.n_visible = n_visible
        self.n_hidden = tuple(n_hidden)
        self.n_layers = len(self.n_hidden)
        self.steps = tuple(steps)
        self.lr = tuple(learning_rate)
        self.momentum = tuple(momentum)
        self.decay = tuple(decay)
        self.T = tuple(temperature)

        model_names = (model,) if isinstance(model, str) else tuple(model)
        if len(model_names) > self.n_layers:
            raise e.SizeError("`model` should not contain more entries than `n_hidden`.")
        model_names += ("sigmoid4deep",) * (self.n_layers - len(model_names))

        unknown = set(model_names) - MODELS.keys()
        if unknown:
            raise e.ValueError(f"`model` contains unknown model type `{sorted(unknown)[0]}`.")

        self.models = nn.ModuleList()
        for i, model_name in enumerate(model_names):
            if i > 0:
                model_name = {
                    "gaussian": "gaussian4deep",
                    "gaussian_relu": "gaussian_relu4deep",
                    "sigmoid": "sigmoid4deep",
                }.get(model_name, model_name)
            model_class = MODELS[model_name]
            kwargs = {
                "n_visible": n_visible if i == 0 else self.n_hidden[i - 1],
                "n_hidden": self.n_hidden[i],
                "steps": self.steps[i],
                "learning_rate": self.lr[i],
                "momentum": self.momentum[i],
                "decay": self.decay[i],
                "temperature": self.T[i],
                "use_gpu": use_gpu,
            }
            if issubclass(model_class, GaussianRBM):
                kwargs.update(
                    normalize=normalize,
                    input_normalize=input_normalize,
                )
            self.models.append(model_class(**kwargs))

        self.to(self.device)

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 128,
        epochs: tuple[int, ...] = (10,),
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Fit each RBM layer greedily.

        The first layer delegates dataset access to its RBM. Later layers shuffle the original dataset for every epoch,
        encode each minibatch through preceding layers without gradients, and fit the current layer on a detached CPU
        ``TensorDataset`` for one epoch. Layer parameters, optimizers, and metric histories persist after this call.

        Args:
            dataset: Dataset yielding ``(sample, target)`` pairs.
            batch_size: Maximum number of samples in each batch.
            epochs: Training epoch counts in layer order.

        Returns:
            Final scalar MSE tensors followed by final scalar pseudo-likelihood tensors, each list in layer order.

        Raises:
            learnergy.utils.exception.SizeError: ``epochs`` does not contain one value per layer.

        """

        if len(epochs) != self.n_layers:
            raise e.SizeError(f"`epochs` should contain {self.n_layers} values.")

        mse = []
        pl = []

        for i, model in enumerate(self.models):
            if i == 0:
                model_mse, model_pl = model.fit(dataset, batch_size=batch_size, epochs=epochs[i])
            else:
                batches = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                for _ in range(epochs[i]):
                    model_mse = 0
                    model_pl = 0
                    for samples, labels in batches:
                        samples = samples.reshape(len(samples), self.n_visible).to(self.device)
                        with torch.no_grad():
                            for previous_model in self.models[:i]:
                                samples, _ = previous_model.hidden_sampling(samples)

                        encoded = TensorDataset(samples.detach().cpu(), labels)
                        batch_mse, batch_pl = model.fit(encoded, batch_size=len(samples), epochs=1)
                        model_mse += batch_mse
                        model_pl += batch_pl

                    model_mse /= len(batches)
                    model_pl /= len(batches)

            mse.append(model_mse)
            pl.append(model_pl)

        return mse, pl

    def reconstruct(self, dataset: torch.utils.data.Dataset) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct all dataset samples through the complete stack.

        The method processes one non-shuffled batch, samples upward through hidden conditionals, and then samples
        downward through visible conditionals. It does not disable gradient tracking.

        Args:
            dataset: Dataset yielding ``(sample, target)`` pairs.

        Returns:
            Scalar MSE tensor then visible probabilities shaped ``(len(dataset), n_visible)``.

        """

        batch_size = len(dataset)
        batches = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        for samples, _ in batches:
            samples = samples.reshape(batch_size, self.models[0].n_visible).to(self.device)

            hidden_probs = samples
            for model in self.models:
                hidden_probs = hidden_probs.reshape(batch_size, model.n_visible)
                hidden_probs, _ = model.hidden_sampling(hidden_probs)

            visible_probs = hidden_probs
            for model in reversed(self.models):
                visible_probs = visible_probs.reshape(batch_size, model.n_hidden)
                visible_probs, visible_states = model.visible_sampling(visible_probs)

            mse = ((samples - visible_states) ** 2).sum() / batch_size

        return mse, visible_probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for model in self.models:
            x, _ = model.hidden_sampling(x)

        return x
