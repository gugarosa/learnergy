"""Residual Deep Belief Network."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import learnergy.utils.exception as e
from learnergy.core.model import _validated_property
from learnergy.models.deep.dbn import DBN


class ResidualDBN(DBN):
    """Deep Belief Network with residual feature reinforcement."""

    zetta1 = _validated_property(
        "zetta1",
        lambda _, value: value >= 0,
        e.ValueError,
        "`zetta1` should be >= 0",
    )
    zetta2 = _validated_property(
        "zetta2",
        lambda _, value: value >= 0,
        e.ValueError,
        "`zetta2` should be >= 0",
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
        """Initialize a residual Deep Belief Network."""

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
        """Normalize positive pre-activations into a residual signal."""

        residual = F.relu(pre_activations)
        return residual / (residual.max() + torch.finfo(pre_activations.dtype).eps)

    def _residual_forward(
        self, model: torch.nn.Module, samples: torch.Tensor
    ) -> torch.Tensor:
        pre_activations = model.pre_activation(samples)
        hidden, _ = model.hidden_sampling(samples)
        encoded = (
            hidden * self.zetta1
            + self.calculate_residual(pre_activations) * self.zetta2
        )
        return encoded / (encoded.max() + torch.finfo(encoded.dtype).eps)

    def _encode_residual_dataset(
        self,
        dataset: torch.utils.data.Dataset,
        model: torch.nn.Module,
        batch_size: int,
    ) -> TensorDataset:
        features = []
        targets = []

        with torch.no_grad():
            for samples, labels in DataLoader(
                dataset, batch_size=batch_size, shuffle=False
            ):
                samples = samples.reshape(len(samples), model.n_visible).to(self.device)
                features.append(self._residual_forward(model, samples).cpu())
                targets.append(labels.cpu())

        return TensorDataset(torch.cat(features), torch.cat(targets))

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 128,
        epochs: tuple[int, ...] = (10,),
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Fit each layer using residual representations from the previous layer."""

        if len(epochs) != self.n_layers:
            raise e.SizeError(f"`epochs` should have {self.n_layers} values")

        current_dataset = dataset
        mse = []
        pl = []

        for i, model in enumerate(self.models):
            model_mse, model_pl = model.fit(
                current_dataset, batch_size=batch_size, epochs=epochs[i]
            )
            mse.append(model_mse)
            pl.append(model_pl)

            if i + 1 < self.n_layers:
                current_dataset = self._encode_residual_dataset(
                    current_dataset, model, batch_size
                )

        return mse, pl

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the residual representation produced by the final layer."""

        for model in self.models:
            x = self._residual_forward(model, x)

        return x
