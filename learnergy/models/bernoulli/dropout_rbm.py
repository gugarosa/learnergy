"""Bernoulli-Bernoulli Restricted Boltzmann Machines with Dropout and DropConnect.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import learnergy.utils.exception as e
from learnergy.models.bernoulli import RBM
from learnergy.utils import logging

logger = logging.get_logger(__name__)


class DropoutRBM(RBM):
    """A DropoutRBM class provides the basic implementation for
    Bernoulli-Bernoulli Restricted Boltzmann Machines along with a Dropout regularization.

    References:
        N. Srivastava, et al. Dropout: a simple way to prevent neural networks from overfitting.
        The journal of machine learning research (2014).

    """

    def __init__(
        self,
        n_visible: Optional[int] = 128,
        n_hidden: Optional[int] = 128,
        steps: Optional[int] = 1,
        learning_rate: Optional[float] = 0.1,
        momentum: Optional[float] = 0.0,
        decay: Optional[float] = 0.0,
        temperature: Optional[float] = 1.0,
        dropout: Optional[float] = 0.5,
        use_gpu: Optional[bool] = False,
    ) -> None:
        """Initialization method.

        Args:
            n_visible: Amount of visible units.
            n_hidden: Amount of hidden units.
            steps: Number of Gibbs' sampling steps.
            learning_rate: Learning rate.
            momentum: Momentum parameter.
            decay: Weight decay used for penalization.
            temperature: Temperature factor.
            dropout: Dropout rate.
            use_gpu: Whether GPU should be used or not.

        """

        logger.info("Overriding class: RBM -> DropoutRBM.")

        super(DropoutRBM, self).__init__(
            n_visible,
            n_hidden,
            steps,
            learning_rate,
            momentum,
            decay,
            temperature,
            use_gpu,
        )

        # Intensity of dropout
        self.p = dropout

        logger.info("Class overrided.")
        logger.debug("Additional hyperparameters: p = %s.", self.p)

    @property
    def p(self) -> float:
        """Probability of applying dropout."""

        return self._p

    @p.setter
    def p(self, p: float) -> None:
        if p < 0 or p > 1:
            raise e.ValueError("`p` should be between 0 and 1")

        self._p = p

    def hidden_sampling(
        self, v: torch.Tensor, scale: Optional[bool] = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the hidden layer sampling using a dropout mask, i.e., P(h|r,v).

        Args:
            v: A tensor incoming from the visible layer.
            scale: A boolean to decide whether temperature should be used or not.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor]): The probabilities and states of the hidden layer sampling.

        """

        # Calculating neurons' activations
        activations = F.linear(v, self.W.t(), self.b)

        # Sampling a dropout mask from Bernoulli's distribution
        mask = (
            torch.full(
                (activations.size(0), activations.size(1)),
                1 - self.p,
                dtype=torch.float,
                device=self.device,
            )
        ).bernoulli()

        # If scaling is true
        if scale:
            # Calculate probabilities with temperature
            probs = torch.mul(torch.sigmoid(torch.div(activations, self.T)), mask)

        # If scaling is false
        else:
            # Calculate probabilities as usual
            probs = torch.mul(torch.sigmoid(activations), mask)

        # Sampling current states
        states = torch.bernoulli(probs)

        return probs, states

    def reconstruct(
        self, dataset: torch.utils.data.Dataset
    ) -> Tuple[float, torch.Tensor]:
        """Reconstructs batches of new samples.

        Args:
            dataset: A Dataset object containing the testing data.

        Returns:
            (Tuple[float, torch.Tensor]): Reconstruction error and visible probabilities, i.e., P(v|h).

        """

        logger.info("Reconstructing new samples ...")

        # Resetting mse to zero
        mse = 0

        # Defining the batch size as the amount of samples in the dataset
        batch_size = len(dataset)

        # Saving dropout rate to an auxiliary variable
        p = self.p

        # Temporarily disabling dropout
        self.p = 0

        # Transforming the dataset into testing batches
        batches = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        # For every batch
        for samples, _ in tqdm(batches):
            # Flattening the samples' batch
            samples = samples.reshape(len(samples), self.n_visible)

            # Checking whether GPU is avaliable and if it should be used
            if self.device == "cuda":
                # Applies the GPU usage to the data
                samples = samples.cuda()

            # Calculating positive phase hidden probabilities and states
            _, pos_hidden_states = self.hidden_sampling(samples)

            # Calculating visible probabilities and states
            visible_probs, visible_states = self.visible_sampling(pos_hidden_states)

            # Calculating current's batch reconstruction MSE
            batch_mse = torch.div(
                torch.sum(torch.pow(samples - visible_states, 2)), batch_size
            )

            # Summing up the reconstruction's MSE
            mse += batch_mse

        # Normalizing the MSE with the number of batches
        mse /= len(batches)

        # Recovering initial dropout rate
        self.p = p

        logger.info("MSE: %f", mse)

        return mse, visible_probs


class DropConnectRBM(DropoutRBM):
    """A DropConnectRBM class provides the basic implementation for
    Bernoulli-Bernoulli Restricted Boltzmann Machines along with a DropConnect regularization.

    References:
        N. Srivastava, et al. Dropout: a simple way to prevent neural networks from overfitting.
        The journal of machine learning research (2014).

    """

    def __init__(
        self,
        n_visible: Optional[int] = 128,
        n_hidden: Optional[int] = 128,
        steps: Optional[int] = 1,
        learning_rate: Optional[float] = 0.1,
        momentum: Optional[float] = 0.0,
        decay: Optional[float] = 0.0,
        temperature: Optional[float] = 1.0,
        dropout: Optional[float] = 0.5,
        use_gpu: Optional[bool] = False,
    ) -> None:
        """Initialization method.

        Args:
            n_visible: Amount of visible units.
            n_hidden: Amount of hidden units.
            steps: Number of Gibbs' sampling steps.
            learning_rate: Learning rate.
            momentum: Momentum parameter.
            decay: Weight decay used for penalization.
            temperature: Temperature factor.
            dropout: Dropout rate.
            use_gpu: Whether GPU should be used or not.

        """

        logger.info("Overriding class: DropoutRBM -> DropConnectRBM.")

        # Override its parent class
        super(DropConnectRBM, self).__init__(
            n_visible,
            n_hidden,
            steps,
            learning_rate,
            momentum,
            decay,
            temperature,
            dropout,
            use_gpu,
        )

    def hidden_sampling(
        self, v: torch.Tensor, scale: Optional[bool] = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the hidden layer sampling using a dropconnect mask, i.e., P(h|m,v).

        Args:
            v: A tensor incoming from the visible layer.
            scale: A boolean to decide whether temperature should be used or not.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor]): The probabilities and states of the hidden layer sampling.

        """

        # Sampling a dropconnect mask from Bernoulli's distribution
        mask = (
            torch.full(
                (self.W.size(0), self.W.size(1)),
                1 - self.p,
                dtype=torch.float,
                device=self.device,
            )
        ).bernoulli()

        # Calculating neurons' activations
        activations = F.linear(v, torch.mul(self.W, mask).t(), self.b)

        # If scaling is true
        if scale:
            # Calculate probabilities with temperature
            probs = torch.sigmoid(torch.div(activations, self.T))

        # If scaling is false
        else:
            # Calculate probabilities as usual
            probs = torch.sigmoid(activations)

        # Sampling current states
        states = torch.bernoulli(probs)

        return probs, states
