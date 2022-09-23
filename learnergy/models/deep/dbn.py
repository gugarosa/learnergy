"""Deep Belief Network.
"""

from typing import List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import learnergy.utils.exception as e
from learnergy.core import Dataset, Model
from learnergy.models.bernoulli import RBM, DropoutRBM, EDropoutRBM
from learnergy.models.extra import SigmoidRBM
from learnergy.models.gaussian import (
    GaussianRBM,
    GaussianReluRBM,
    GaussianSeluRBM,
    VarianceGaussianRBM,
)
from learnergy.utils import logging

logger = logging.get_logger(__name__)

MODELS = {
    "bernoulli": RBM,
    "dropout": DropoutRBM,
    "e_dropout": EDropoutRBM,
    "gaussian": GaussianRBM,
    "gaussian_relu": GaussianReluRBM,
    "gaussian_selu": GaussianSeluRBM,
    "sigmoid": SigmoidRBM,
    "variance_gaussian": VarianceGaussianRBM,
}


class DBN(Model):
    """A DBN class provides the basic implementation for Deep Belief Networks.

    References:
        G. Hinton, S. Osindero, Y. Teh. A fast learning algorithm for deep belief nets.
        Neural computation (2006).

    """

    def __init__(
        self,
        model: Optional[str] = "bernoulli",
        n_visible: Optional[int] = 128,
        n_hidden: Optional[Tuple[int, ...]] = (128,),
        steps: Optional[Tuple[int, ...]] = (1,),
        learning_rate: Optional[Tuple[float, ...]] = (0.1,),
        momentum: Optional[Tuple[float, ...]] = (0.0,),
        decay: Optional[Tuple[float, ...]] = (0.0,),
        temperature: Optional[Tuple[float, ...]] = (1.0,),
        use_gpu: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            model: Indicates which type of RBM should be used to compose the DBN.
            n_visible: Amount of visible units.
            n_hidden: Amount of hidden units per layer.
            steps: Number of Gibbs' sampling steps per layer.
            learning_rate: Learning rate per layer.
            momentum: Momentum parameter per layer.
            decay: Weight decay used for penalization per layer.
            temperature: Temperature factor per layer.
            use_gpu: Whether GPU should be used or not.

        """

        logger.info("Overriding class: Model -> DBN.")

        super(DBN, self).__init__(use_gpu=use_gpu)

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_layers = len(n_hidden)

        self.steps = steps
        self.lr = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.T = temperature

        self.models = []
        for i in range(self.n_layers):
            if i == 0:
                n_input = self.n_visible
            else:
                # Gathers the number of input units as previous number of hidden units
                n_input = self.n_hidden[i - 1]

                # After creating the first layer, we need to change the model's type to sigmoid
                model = "sigmoid"

            m = MODELS[model](
                n_input,
                self.n_hidden[i],
                self.steps[i],
                self.lr[i],
                self.momentum[i],
                self.decay[i],
                self.T[i],
                use_gpu,
            )
            self.models.append(m)

        if self.device == "cuda":
            self.cuda()

        logger.info("Class overrided.")
        logger.debug("Number of layers: %d.", self.n_layers)

    @property
    def n_visible(self) -> int:
        """Number of visible units."""

        return self._n_visible

    @n_visible.setter
    def n_visible(self, n_visible: int) -> None:
        if n_visible <= 0:
            raise e.ValueError("`n_visible` should be > 0")

        self._n_visible = n_visible

    @property
    def n_hidden(self) -> Tuple[int, ...]:
        """Tuple of hidden units."""

        return self._n_hidden

    @n_hidden.setter
    def n_hidden(self, n_hidden: Tuple[int, ...]) -> None:
        self._n_hidden = n_hidden

    @property
    def n_layers(self) -> int:
        """Number of layers."""

        return self._n_layers

    @n_layers.setter
    def n_layers(self, n_layers: int) -> None:
        if n_layers <= 0:
            raise e.ValueError("`n_layers` should be > 0")

        self._n_layers = n_layers

    @property
    def steps(self) -> Tuple[int, ...]:
        """Number of steps Gibbs' sampling steps per layer."""

        return self._steps

    @steps.setter
    def steps(self, steps: Tuple[int, ...]) -> None:
        if len(steps) != self.n_layers:
            raise e.SizeError(f"`steps` should have size equal as {self.n_layers}")

        self._steps = steps

    @property
    def lr(self) -> Tuple[float, ...]:
        """Learning rate per layer."""

        return self._lr

    @lr.setter
    def lr(self, lr: Tuple[float, ...]) -> None:
        if len(lr) != self.n_layers:
            raise e.SizeError(f"`lr` should have size equal as {self.n_layers}")

        self._lr = lr

    @property
    def momentum(self) -> Tuple[float, ...]:
        """Momentum parameter per layer."""

        return self._momentum

    @momentum.setter
    def momentum(self, momentum: Tuple[float, ...]) -> None:
        if len(momentum) != self.n_layers:
            raise e.SizeError(f"`momentum` should have size equal as {self.n_layers}")

        self._momentum = momentum

    @property
    def decay(self) -> Tuple[float, ...]:
        """Weight decay per layer."""

        return self._decay

    @decay.setter
    def decay(self, decay: Tuple[float, ...]) -> None:
        if len(decay) != self.n_layers:
            raise e.SizeError(f"`decay` should have size equal as {self.n_layers}")

        self._decay = decay

    @property
    def T(self) -> Tuple[float, ...]:
        """Temperature factor per layer."""

        return self._T

    @T.setter
    def T(self, T: Tuple[float, ...]) -> None:
        if len(T) != self.n_layers:
            raise e.SizeError(f"`T` should have size equal as {self.n_layers}")

        self._T = T

    @property
    def models(self) -> List[torch.nn.Module]:
        """List of models (RBMs)."""

        return self._models

    @models.setter
    def models(self, models: List[torch.nn.Module]) -> None:
        self._models = models

    def fit(
        self,
        dataset: Union[torch.utils.data.Dataset, Dataset],
        batch_size: Optional[int] = 128,
        epochs: Optional[Tuple[int, ...]] = (10,),
    ) -> Tuple[float, float]:
        """Fits a new DBN model.

        Args:
            dataset: A Dataset object containing the training data.
            batch_size: Amount of samples per batch.
            epochs: Number of training epochs per layer.

        Returns:
            (Tuple[float, float]): MSE (mean squared error) and log pseudo-likelihood from the training step.

        """

        if len(epochs) != self.n_layers:
            raise e.SizeError(("`epochs` should have size equal as %d", self.n_layers))

        mse, pl = [], []

        samples, targets, transform = (
            dataset.data.numpy(),
            dataset.targets.numpy(),
            dataset.transform,
        )

        for i, model in enumerate(self.models):
            logger.info("Fitting layer %d/%d ...", i + 1, self.n_layers)

            d = Dataset(samples, targets, transform)

            model_mse, model_pl = model.fit(d, batch_size, epochs[i])
            mse.append(model_mse)
            pl.append(model_pl)

            if d.transform:
                samples = d.transform(d.data)
            else:
                samples = d.data

            if self.device == "cuda":
                samples = samples.cuda()

            samples = samples.reshape(len(dataset), model.n_visible)
            targets = d.targets
            transform = None

            # Performs a forward pass over the samples to get their probabilities
            samples, _ = model.hidden_sampling(samples)

            if self.device == "cuda":
                samples = samples.cpu()
            samples = samples.detach()

        return mse, pl

    def reconstruct(
        self, dataset: torch.utils.data.Dataset
    ) -> Tuple[float, torch.Tensor]:
        """Reconstructs batches of new samples.

        Args:
            dataset (torch.utils.data.Dataset): A Dataset object containing the training data.

        Returns:
            (Tuple[float, torch.Tensor]): Reconstruction error and visible probabilities, i.e., P(v|h).

        """

        logger.info("Reconstructing new samples ...")

        mse = 0
        batch_size = len(dataset)

        batches = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        for samples, _ in tqdm(batches):
            samples = samples.reshape(batch_size, self.models[0].n_visible)
            if self.device == "cuda":
                samples = samples.cuda()

            hidden_probs = samples
            for model in self.models:
                hidden_probs = hidden_probs.reshape(batch_size, model.n_visible)
                hidden_probs, _ = model.hidden_sampling(hidden_probs)

            visible_probs = hidden_probs
            for model in reversed(self.models):
                visible_probs = visible_probs.reshape(batch_size, model.n_hidden)
                visible_probs, visible_states = model.visible_sampling(visible_probs)

            batch_mse = torch.div(
                torch.sum(torch.pow(samples - visible_states, 2)), batch_size
            )
            mse += batch_mse

        mse /= len(batches)

        logger.info("MSE: %f", mse)

        return mse, visible_probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass over the data.

        Args:
            x: An input tensor for computing the forward pass.

        Returns:
            (torch.Tensor): A tensor containing the DBN's outputs.

        """

        for model in self.models:
            x, _ = model.hidden_sampling(x)

        return x
