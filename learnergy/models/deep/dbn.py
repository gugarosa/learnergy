"""Deep Belief Network."""

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
    """Stack RBMs and train them greedily, one layer at a time."""

    n_visible = _validated_property(
        "n_visible",
        lambda _, value: value > 0,
        e.ValueError,
        "`n_visible` should be > 0",
    )
    n_hidden = _validated_property("n_hidden")
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
    T = _validated_property(
        "T",
        lambda self, value: len(value) == self.n_layers,
        e.SizeError,
        "`T` should match the number of layers",
    )
    models = _validated_property("models")

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
        """Initialize a Deep Belief Network."""

        super().__init__(use_gpu=use_gpu)

        if not n_hidden or any(value <= 0 for value in n_hidden):
            raise e.ValueError("`n_hidden` should contain positive values")

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
            raise e.SizeError("`model` should not contain more entries than layers")
        model_names += ("sigmoid4deep",) * (self.n_layers - len(model_names))

        unknown = set(model_names) - MODELS.keys()
        if unknown:
            raise e.ValueError(f"unknown model type: {sorted(unknown)[0]}")

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
        """Fit each RBM layer and return its final MSE and pseudo-likelihood."""

        if len(epochs) != self.n_layers:
            raise e.SizeError(f"`epochs` should have {self.n_layers} values")

        mse = []
        pl = []

        for i, model in enumerate(self.models):
            if i == 0:
                model_mse, model_pl = model.fit(
                    dataset, batch_size=batch_size, epochs=epochs[i]
                )
            else:
                batches = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                for _ in range(epochs[i]):
                    model_mse = 0
                    model_pl = 0
                    for samples, labels in batches:
                        samples = samples.reshape(len(samples), self.n_visible).to(
                            self.device
                        )
                        with torch.no_grad():
                            for previous_model in self.models[:i]:
                                samples, _ = previous_model.hidden_sampling(samples)

                        encoded = TensorDataset(samples.detach().cpu(), labels)
                        batch_mse, batch_pl = model.fit(
                            encoded, batch_size=len(samples), epochs=1
                        )
                        model_mse += batch_mse
                        model_pl += batch_pl

                    model_mse /= len(batches)
                    model_pl /= len(batches)

            mse.append(model_mse)
            pl.append(model_pl)

        return mse, pl

    def reconstruct(
        self, dataset: torch.utils.data.Dataset
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct a dataset through the full stack."""

        batch_size = len(dataset)
        batches = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        for samples, _ in batches:
            samples = samples.reshape(batch_size, self.models[0].n_visible).to(
                self.device
            )

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
        """Return the representation produced by the final layer."""

        for model in self.models:
            x, _ = model.hidden_sampling(x)

        return x
