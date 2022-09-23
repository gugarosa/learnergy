"""Residual-based Deep Belief Networks.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

import learnergy.utils.exception as e
from learnergy.core import Dataset
from learnergy.models.deep import DBN
from learnergy.utils import logging

logger = logging.get_logger(__name__)


class ResidualDBN(DBN):
    """A ResidualDBN class provides the basic implementation for Residual-based Deep Belief Networks.

    References:
        M. Roder, et al. A Layer-Wise Information Reinforcement Approach to Improve Learning in Deep Belief Networks.
        International Conference on Artificial Intelligence and Soft Computing (2020).

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
        zetta1: Optional[float] = 1.0,
        zetta2: Optional[float] = 1.0,
        use_gpu: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            model (str): Indicates which type of RBM should be used to compose the ResidualDBN.
            n_visible (int): Amount of visible units.
            n_hidden (tuple): Amount of hidden units per layer.
            steps (tuple): Number of Gibbs' sampling steps per layer.
            learning_rate (tuple): Learning rate per layer.
            momentum (tuple): Momentum parameter per layer.
            decay (tuple): Weight decay used for penalization per layer.
            temperature (tuple): Temperature factor per layer.
            zetta1 (float): Penalization factor for original learning.
            zetta2 (float): Penalization factor for residual learning.
            use_gpu (boolean): Whether GPU should be used or not.

        """

        logger.info("Overriding class: DBN -> ResidualDBN.")

        super(ResidualDBN, self).__init__(
            model,
            n_visible,
            n_hidden,
            steps,
            learning_rate,
            momentum,
            decay,
            temperature,
            use_gpu,
        )

        self.zetta1 = zetta1
        self.zetta2 = zetta2

    @property
    def zetta1(self) -> float:
        """Penalization factor for original learning."""

        return self._zetta1

    @zetta1.setter
    def zetta1(self, zetta1: float) -> None:
        if zetta1 < 0:
            raise e.ValueError("`zetta1` should be >= 0")

        self._zetta1 = zetta1

    @property
    def zetta2(self) -> float:
        """Penalization factor for residual learning."""

        return self._zetta2

    @zetta2.setter
    def zetta2(self, zetta2: float) -> None:
        if zetta2 < 0:
            raise e.ValueError("`zetta2` should be >= 0")

        self._zetta2 = zetta2

    def calculate_residual(self, pre_activations: torch.Tensor) -> torch.Tensor:
        """Calculates the residual learning over input.

        Args:
            pre_activations (torch.Tensor): Pre-activations to be used.

        Returns:
            (torch.Tensor): The residual learning based on input pre-activations.

        """

        residual = F.relu(pre_activations)
        residual = torch.div(residual, torch.max(residual))

        return residual

    def fit(
        self,
        dataset: Union[torch.utils.data.Dataset, Dataset],
        batch_size: Optional[int] = 128,
        epochs: Optional[Tuple[int, ...]] = (10,),
    ) -> Tuple[float, float]:
        """Fits a new ResidualDBN model.

        Args:
            dataset: A Dataset object containing the training data.
            batch_size: Amount of samples per batch.
            epochs: Number of training epochs per layer.

        Returns:
            (Tuple[float, float]): MSE (mean squared error) and log pseudo-likelihood from the training step.

        """

        if len(epochs) != self.n_layers:
            raise e.SizeError(f"`epochs` should have size equal as {self.n_layers}")

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

            pre_activation = model.pre_activation(samples)

            # Aggregates the residual learning after forward pass
            samples, _ = model.hidden_sampling(samples)
            samples = torch.mul(samples, self.zetta1) + torch.mul(
                self.calculate_residual(pre_activation), self.zetta2
            )

            # Normalizes the input for the next layer
            samples = torch.div(samples, torch.max(samples))

            if self.device == "cuda":
                samples = samples.cpu()
            samples = samples.detach()

        return mse, pl

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Re-writes the forward pass for classification purposes.

        Args:
            x: An input tensor for computing the forward pass.

        Returns:
            (torch.Tensor): A tensor containing the DBN's outputs.

        """

        for model in self.models:
            pre_activation = model.pre_activation(x)

            # Aggregates the residual learning after forward pass
            x, _ = model.hidden_sampling(x)
            x = torch.mul(x, self.zetta1) + torch.mul(
                self.calculate_residual(pre_activation), self.zetta2
            )

            # Normalizes the input for the next layer
            x = torch.div(x, torch.max(x))

        return x
