# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Provide a Deep Belief Network of recurrent temporal RBMs.

Greedy pretraining materializes detached CPU features and their original targets between layers. Encoding retains the
time dimension through the stack, then averages the final hidden probabilities over time. Generation samples the top
layer and draws lower visible conditionals in a top-down pass.

"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import learnergy.utils.exception as e
from learnergy.core.model import Model, _validated_property
from learnergy.models.temporal.rt_gaussian_rbm import RTGaussianRBM
from learnergy.models.temporal.rt_variance_gaussian_rbm import RTVarianceGaussianRBM
from learnergy.models.temporal.rtrbm import RTRBM, _validate_positive_integer
from learnergy.utils.logging import get_logger

logger = get_logger(__name__)

RT_MODELS = {
    "bernoulli": RTRBM,
    "gaussian": RTGaussianRBM,
    "variance_gaussian": RTVarianceGaussianRBM,
}


class RTDBN(Model):
    """Stack recurrent temporal RBMs for greedy training and mean-pooled sequence embeddings."""

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
        doc="Number of stacked recurrent temporal RBMs.",
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
        doc="Learning rates used when constructing each layer.",
    )
    momentum = _validated_property(
        "momentum",
        lambda self, value: len(value) == self.n_layers,
        e.SizeError,
        "`momentum` should match the number of layers.",
        doc="Momentum settings used when constructing each layer.",
    )
    decay = _validated_property(
        "decay",
        lambda self, value: len(value) == self.n_layers,
        e.SizeError,
        "`decay` should match the number of layers.",
        doc="Weight-decay settings used when constructing each layer.",
    )
    T = _validated_property(
        "T",
        lambda self, value: len(value) == self.n_layers,
        e.SizeError,
        "`T` should match the number of layers.",
        doc="Sampling temperatures in layer order.",
    )
    models = _validated_property("models", doc="Registered recurrent temporal RBM layers in training order.")

    def __init__(
        self,
        model: str | tuple[str, ...] = ("variance_gaussian",),
        n_visible: int = 78,
        n_hidden: tuple[int, ...] = (64,),
        steps: tuple[int, ...] = (1,),
        learning_rate: tuple[float, ...] = (0.001,),
        momentum: tuple[float, ...] = (0.0,),
        decay: tuple[float, ...] = (0.0,),
        temperature: tuple[float, ...] = (1.0,),
        use_gpu: bool = False,
        normalize: bool = True,
        input_normalize: bool = True,
    ) -> None:
        """Initialize one recurrent temporal RBM for each layer.

        Model names are ``bernoulli``, ``gaussian``, and ``variance_gaussian``. Each layer-wise option must contain
        exactly one entry per hidden-unit count. Normalization flags apply only to fixed-variance Gaussian layers.

        Args:
            model: A single-layer model name or model names in layer order.
            n_visible: Number of visible units in the first layer.
            n_hidden: Positive hidden-unit counts in layer order.
            steps: Positive Gibbs sampling step counts in layer order.
            learning_rate: Learning rates used by SGD in layer order.
            momentum: SGD momentum settings in layer order.
            decay: SGD weight-decay settings in layer order.
            temperature: Positive sampling temperatures in layer order.
            use_gpu: Whether to select CUDA when it is available.
            normalize: Whether fixed-variance Gaussian layers standardize training and reconstruction batches.
            input_normalize: Whether fixed-variance Gaussian layers standardize and detach forward inputs.

        Raises:
            TypeError: A unit count or step count is not an integer.
            ValueError: A unit count, model name, or layer parameter is invalid.
            learnergy.utils.exception.SizeError: A layer-wise option has an invalid length.

        """

        super().__init__(use_gpu=use_gpu)
        _validate_positive_integer("n_visible", n_visible)
        if not n_hidden:
            raise e.ValueError("`n_hidden` should contain at least one layer.")
        for value in n_hidden:
            _validate_positive_integer("n_hidden", value)

        self.n_visible = n_visible
        self.n_hidden = tuple(n_hidden)
        self.n_layers = len(self.n_hidden)
        self.steps = tuple(steps)
        self.lr = tuple(learning_rate)
        self.momentum = tuple(momentum)
        self.decay = tuple(decay)
        self.T = tuple(temperature)

        model_names = (model,) if isinstance(model, str) else tuple(model)
        if len(model_names) != self.n_layers:
            raise e.SizeError("`model` should match the number of layers.")
        unknown = set(model_names) - RT_MODELS.keys()
        if unknown:
            raise e.ValueError(f"`model` contains unknown model type `{sorted(unknown)[0]}`.")

        self.models = nn.ModuleList()
        for i, name in enumerate(model_names):
            model_class = RT_MODELS[name]
            kwargs = {
                "n_visible": self.n_visible if i == 0 else self.n_hidden[i - 1],
                "n_hidden": self.n_hidden[i],
                "steps": self.steps[i],
                "learning_rate": self.lr[i],
                "momentum": self.momentum[i],
                "decay": self.decay[i],
                "temperature": self.T[i],
                "use_gpu": use_gpu,
            }
            if model_class is RTGaussianRBM:
                kwargs.update(normalize=normalize, input_normalize=input_normalize)
            self.models.append(model_class(**kwargs))

        self.to(self.device)

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 32,
        epochs: tuple[int, ...] = (30,),
        warmup_epochs: tuple[int, ...] = (15,),
    ) -> list[float]:
        """Train each layer greedily without changing existing parameter freezes or module modes.

        Learned-variance layers optionally hold ``sigma`` fixed for an initial portion of their total epoch budget.
        Warmup is capped at the requested epoch count, and omitted warmup entries mean zero. Between layers, hidden
        probabilities and original targets are materialized as a detached CPU dataset without changing earlier layers.
        Each RBM retains its optimizer and metric history, and temporary scale freezes are restored even on failure.

        Args:
            dataset: Nonempty dataset yielding ``(sequence, target)`` pairs.
            batch_size: Positive maximum number of sequences in each batch.
            epochs: Positive total training epoch counts in layer order, including warmup.
            warmup_epochs: Nonnegative scale-freezing epoch counts for the corresponding learned-variance layers.

        Returns:
            Final-epoch MSE floats in layer order.

        Raises:
            TypeError: A batch size, epoch count, or warmup count is not an integer.
            ValueError: The dataset is empty or a training count is invalid.
            learnergy.utils.exception.SizeError: Epoch or warmup counts have incompatible lengths.

        """

        _validate_positive_integer("batch_size", batch_size)
        if len(dataset) == 0:
            raise e.ValueError("`dataset` should contain at least one sequence.")
        if len(epochs) != self.n_layers:
            raise e.SizeError("`epochs` should match the number of layers.")
        if len(warmup_epochs) > self.n_layers:
            raise e.SizeError("`warmup_epochs` should not contain more entries than the number of layers.")
        for value in epochs:
            _validate_positive_integer("epochs", value)
        for value in warmup_epochs:
            if isinstance(value, bool) or not isinstance(value, int):
                raise e.TypeError("`warmup_epochs` should contain integers.")
            if value < 0:
                raise e.ValueError("`warmup_epochs` should contain nonnegative values.")

        errors = []
        current_dataset = dataset
        for i, model in enumerate(self.models):
            logger.info("Fitting RTDBN layer %d/%d", i + 1, self.n_layers)
            warmup = min(warmup_epochs[i], epochs[i]) if i < len(warmup_epochs) else 0
            if not isinstance(model, RTVarianceGaussianRBM):
                warmup = 0

            if warmup:
                scale_requires_grad = model.sigma.requires_grad
                try:
                    model.sigma.requires_grad_(False)
                    model.fit(current_dataset, batch_size=batch_size, epochs=warmup)
                finally:
                    model.sigma.requires_grad_(scale_requires_grad)
            if epochs[i] > warmup:
                model.fit(current_dataset, batch_size=batch_size, epochs=epochs[i] - warmup)
            errors.append(model.history["mse"][-1])

            if i < self.n_layers - 1:
                batches = DataLoader(current_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                encoded = []
                targets = []
                with torch.no_grad():
                    for samples, labels in batches:
                        encoded.append(model(samples.to(model.W)).cpu())
                        targets.append(labels.cpu())
                current_dataset = TensorDataset(torch.cat(encoded), torch.cat(targets))

        return errors

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode complete sequences and average the final hidden probabilities over time.

        Gradient tracking is preserved unless a fixed-variance Gaussian layer standardizes and detaches its input.
        Set ``input_normalize=False`` when end-to-end input gradients are required.

        Args:
            x: Floating-point tensor shaped ``(batch_size, sequence_length, n_visible)``.

        Returns:
            Mean-pooled sequence embeddings shaped ``(batch_size, n_hidden[-1])``.

        """

        for model in self.models:
            x = model(x)

        return x.mean(dim=1)

    def sample(self, n_samples: int = 1, n_steps: int = 10, gibbs_steps: int = 100) -> torch.Tensor:
        """Generate sequences at the top layer and sample downward through the remaining layers.

        The top layer performs recurrent Gibbs generation. Each lower layer draws its visible conditional once per
        timestep, including Gaussian noise for continuous units. Sampling does not track gradients or mutate parameters.

        Args:
            n_samples: Positive number of sequences to generate.
            n_steps: Positive number of timesteps per sequence.
            gibbs_steps: Positive number of top-layer Gibbs transitions per timestep.

        Returns:
            Detached visible sequences shaped ``(n_samples, n_steps, n_visible)`` on the model's device and dtype.

        Raises:
            TypeError: A sampling count is not an integer.
            ValueError: A sampling count is not positive.

        """

        with torch.no_grad():
            current = self.models[-1].sample(n_samples, n_steps, gibbs_steps)
            for model in reversed(self.models[:-1]):
                current = torch.stack([model._sample_visible(hidden) for hidden in current.unbind(dim=1)], dim=1)

        return current

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)
