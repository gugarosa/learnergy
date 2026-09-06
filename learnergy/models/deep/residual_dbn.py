# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Provide a Deep Belief Network with residual feature reinforcement.

Each layer combines its hidden probabilities with globally normalized positive pre-activations, then normalizes the
combined tensor by its global maximum. Residual dataset encoding runs without gradients, keeps source order and target
alignment, and stores CPU copies between layers. Hidden samplers still produce their discarded second tuple values.

"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import learnergy.utils.exception as e
from learnergy.core.model import _validated_property
from learnergy.models.deep.dbn import DBN


class ResidualDBN(DBN):
    """Stack RBMs with residual feature reinforcement between layers."""

    zetta1 = _validated_property(
        "zetta1",
        lambda _, value: value >= 0,
        e.ValueError,
        "`zetta1` should be greater than or equal to 0.",
        doc="Weight applied to hidden probabilities in residual representations.",
    )
    zetta2 = _validated_property(
        "zetta2",
        lambda _, value: value >= 0,
        e.ValueError,
        "`zetta2` should be greater than or equal to 0.",
        doc="Weight applied to normalized positive pre-activations.",
    )

    def __init__(
        self,
        model: str | tuple[str, ...] = "bernoulli",
        n_visible: int = 128,
        n_hidden: tuple[int, ...] = (128,),
        steps: tuple[int, ...] = (1,),
        learning_rate: tuple[float, ...] = (0.1,),
        momentum: tuple[float, ...] = (0.0,),
        decay: tuple[float, ...] = (0.0,),
        temperature: tuple[float, ...] = (1.0,),
        zetta1: float = 1.0,
        zetta2: float = 1.0,
        use_gpu: bool = False,
    ) -> None:
        """Initialize a residual Deep Belief Network.

        Args:
            model: Model name for the first layer or model names in layer order.
            n_visible: Number of visible units in the first layer.
            n_hidden: Hidden-unit counts in layer order.
            steps: Gibbs sampling step counts in layer order.
            learning_rate: Learning rates in layer order.
            momentum: SGD momentum values in layer order.
            decay: SGD weight-decay values in layer order.
            temperature: Sampling temperatures in layer order.
            zetta1: Nonnegative weight applied to hidden probabilities.
            zetta2: Nonnegative weight applied to normalized positive pre-activations.
            use_gpu: Whether to select CUDA when it is available.

        Raises:
            ValueError: A residual weight is negative or an inherited value is invalid.
            learnergy.utils.exception.SizeError: An inherited layer-wise sequence has an invalid length.

        """

        super().__init__(
            model=model,
            n_visible=n_visible,
            n_hidden=n_hidden,
            steps=steps,
            learning_rate=learning_rate,
            momentum=momentum,
            decay=decay,
            temperature=temperature,
            use_gpu=use_gpu,
        )

        self.zetta1 = zetta1
        self.zetta2 = zetta2

    def calculate_residual(self, pre_activations: torch.Tensor) -> torch.Tensor:
        """Normalize positive pre-activations into a residual tensor.

        ReLU values are divided by the maximum over the complete tensor plus machine epsilon. Gradient tracking is
        preserved and the input tensor is not mutated.

        Args:
            pre_activations: Pre-activation tensor of any shape.

        Returns:
            Residual tensor with the same shape, device, and dtype as ``pre_activations``.

        """

        residual = F.relu(pre_activations)
        return residual / (residual.max() + torch.finfo(pre_activations.dtype).eps)

    def _residual_forward(self, model: torch.nn.Module, samples: torch.Tensor) -> torch.Tensor:
        pre_activations = model.pre_activation(samples)
        hidden, _ = model.hidden_sampling(samples)
        encoded = hidden * self.zetta1 + self.calculate_residual(pre_activations) * self.zetta2
        return encoded / (encoded.max() + torch.finfo(encoded.dtype).eps)

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 128,
        epochs: tuple[int, ...] = (10,),
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Fit each layer from the preceding layer's residual representation.

        Every layer delegates training to its RBM. Before the next layer, the current dataset is read without shuffling
        and encoded under ``torch.no_grad()`` into a detached CPU ``TensorDataset`` whose targets retain source order.
        Layer parameters, optimizers, and metric histories persist after this call.

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

        current_dataset = dataset
        mse = []
        pl = []

        for i, model in enumerate(self.models):
            model_mse, model_pl = model.fit(current_dataset, batch_size=batch_size, epochs=epochs[i])
            mse.append(model_mse)
            pl.append(model_pl)

            if i + 1 < self.n_layers:
                features = []
                targets = []

                with torch.no_grad():
                    for samples, labels in DataLoader(current_dataset, batch_size=batch_size, shuffle=False):
                        samples = samples.reshape(len(samples), model.n_visible).to(self.device)
                        features.append(self._residual_forward(model, samples).cpu())
                        targets.append(labels.cpu())

                current_dataset = TensorDataset(torch.cat(features), torch.cat(targets))

        return mse, pl

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for model in self.models:
            x = self._residual_forward(model, x)

        return x
