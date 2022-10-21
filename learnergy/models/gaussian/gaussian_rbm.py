"""Gaussian-Bernoulli Restricted Boltzmann Machine.
"""

import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import learnergy.utils.constants as c
import learnergy.utils.exception as e
from learnergy.models.bernoulli import RBM
from learnergy.utils import logging

logger = logging.get_logger(__name__)


class GaussianRBM(RBM):
    """A GaussianRBM class provides the basic implementation for
    Gaussian-Bernoulli Restricted Boltzmann Machines (with standardization).

    Note that this classes normalize the data
    as it uses variance equals to one throughout its learning procedure.

    This is a trick to ease the calculations of the hidden and
    visible layer samplings, as well as the cost function.

    References:
        K. Cho, A. Ilin, T. Raiko.
        Improved learning of Gaussian-Bernoulli restricted Boltzmann machines.
        International conference on artificial neural networks (2011).

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
        use_gpu: Optional[bool] = False,
        normalize: Optional[bool] = True,
        input_normalize: Optional[bool] = True,
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
            use_gpu: Whether GPU should be used or not.
            normalize: Whether or not to use batch normalization.
            input_normalize: Whether or not to normalize inputs.

        """

        self._normalize = normalize
        self._input_normalize = input_normalize

        logger.info("Overriding class: RBM -> GaussianRBM.")

        super(GaussianRBM, self).__init__(
            n_visible,
            n_hidden,
            steps,
            learning_rate,
            momentum,
            decay,
            temperature,
            use_gpu,
        )

        logger.info("Class overrided.")

    @property
    def normalize(self) -> bool:
        """Whether or not to use batch normalization."""

        return self._normalize

    @normalize.setter
    def normalize(self, normalize: bool) -> None:
        self._normalize = normalize

    @property
    def input_normalize(self) -> bool:
        """Whether or not to use input normalization."""

        return self._input_normalize

    @input_normalize.setter
    def input_normalize(self, input_normalize: bool) -> None:
        self._input_normalize = input_normalize

    def energy(self, samples: torch.Tensor) -> torch.Tensor:
        """Calculates and frees the system's energy.

        Args:
            samples: Samples to be energy-freed.

        Returns:
            (torch.Tensor): The system's energy based on input samples.

        """

        activations = F.linear(samples, self.W.t(), self.b)

        # Creates a Softplus function for numerical stability
        s = nn.Softplus()

        h = torch.sum(s(activations), dim=1)
        v = 0.5 * torch.sum((samples - self.a) ** 2, dim=1)

        energy = v - h

        return energy

    def visible_sampling(
        self, h: torch.Tensor, scale: Optional[bool] = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the visible layer sampling, i.e., P(v|h).

        Args:
            h: A tensor incoming from the hidden layer.
            scale: A boolean to decide whether temperature should be used or not.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor]): The probabilities and states of the visible layer sampling.

        """

        activations = F.linear(h, self.W, self.a)

        if scale:
            states = torch.div(activations, self.T)
        else:
            states = activations

        probs = torch.sigmoid(states)

        return probs, states

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: Optional[int] = 128,
        epochs: Optional[int] = 10,
    ) -> Tuple[float, float]:
        """Fits a new GaussianRBM model.

        Args:
            dataset: A Dataset object containing the training data.
            batch_size: Amount of samples per batch.
            epochs: Number of training epochs.

        Returns:
            (Tuple[float, float]): MSE (mean squared error) and log pseudo-likelihood from the training step.

        """

        batches = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        for epoch in range(epochs):
            logger.info("Epoch %d/%d", epoch + 1, epochs)

            start = time.time()

            mse = 0
            pl = 0

            for samples, _ in tqdm(batches):
                if self.normalize:
                    samples = (
                        (samples - torch.mean(samples, 0, True))
                        / (torch.std(samples, 0, True) + 1e-6)
                    ).detach()

                samples = samples.reshape(len(samples), self.n_visible)
                if self.device == "cuda":
                    samples = samples.cuda()

                # Performs the Gibbs sampling procedure
                _, _, _, _, visible_states = self.gibbs_sampling(samples)
                visible_states = visible_states.detach()

                cost = torch.mean(self.energy(samples)) - torch.mean(
                    self.energy(visible_states)
                )

                self.optimizer.zero_grad()
                cost.backward()
                self.optimizer.step()

                batch_size = samples.size(0)

                batch_mse = torch.div(
                    torch.sum(torch.pow(samples - visible_states, 2)), batch_size
                ).detach()
                batch_pl = self.pseudo_likelihood(samples).detach()

                mse += batch_mse
                pl += batch_pl

            mse /= len(batches)
            pl /= len(batches)

            end = time.time()

            self.dump(mse=mse.item(), pl=pl.item(), time=end - start)

            logger.info("MSE: %f | log-PL: %f", mse, pl)

        return mse, pl

    def reconstruct(
        self, dataset: torch.utils.data.Dataset
    ) -> Tuple[float, torch.Tensor]:
        """Reconstructs batches of new samples.

        Args:
            dataset (torch.utils.data.Dataset): A Dataset object containing the testing data.

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
            if self.normalize:
                samples = (
                    (samples - torch.mean(samples, 0, True))
                    / (torch.std(samples, 0, True) + 1e-6)
                ).detach()

            samples = samples.reshape(len(samples), self.n_visible)
            if self.device == "cuda":
                samples = samples.cuda()

            _, pos_hidden_states = self.hidden_sampling(samples)
            visible_probs, visible_states = self.visible_sampling(pos_hidden_states)

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
            (torch.Tensor): A tensor containing the RBM's outputs.

        """

        if self.input_normalize:
            x = ((x - torch.mean(x, 0, True)) / (torch.std(x, 0, True) + 1e-6)).detach()

        x, _ = self.hidden_sampling(x)

        return x


class GaussianReluRBM(GaussianRBM):
    """A GaussianReluRBM class provides the basic implementation for
    Gaussian-ReLU Restricted Boltzmann Machines (for raw pixels values).

    Note that this class requires raw data (integer-valued)
    in order to model the image covariance into a latent ReLU layer.

    References:
        G. Hinton. A practical guide to training restricted Boltzmann machines.
        Neural networks: Tricks of the trade (2012).

    """

    def __init__(
        self,
        n_visible: Optional[int] = 128,
        n_hidden: Optional[int] = 128,
        steps: Optional[int] = 1,
        learning_rate: Optional[float] = 0.001,
        momentum: Optional[float] = 0.0,
        decay: Optional[float] = 0.0,
        temperature: Optional[float] = 1.0,
        use_gpu: Optional[bool] = False,
        normalize: Optional[bool] = True,
        input_normalize: Optional[bool] = True,
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
            use_gpu: Whether GPU should be used or not.
            normalize: Whether or not to use batch normalization.
            input_normalize: Whether or not to normalize inputs.

        """

        logger.info("Overriding class: GaussianRBM -> GaussianReluRBM.")

        # Override its parent class
        super(GaussianReluRBM, self).__init__(
            n_visible,
            n_hidden,
            steps,
            learning_rate,
            momentum,
            decay,
            temperature,
            use_gpu,
            normalize,
            input_normalize,
        )

        logger.info("Class overrided.")

    def hidden_sampling(
        self, v: torch.Tensor, scale: Optional[bool] = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the hidden layer sampling, i.e., P(h|v).

        Args:
            v: A tensor incoming from the visible layer.
            scale: A boolean to decide whether temperature should be used or not.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor]): The probabilities and states of the hidden layer sampling.

        """

        activations = F.linear(v, self.W.t(), self.b)

        if scale:
            probs = F.relu(torch.div(activations, self.T))
        else:
            probs = F.relu(activations)

        # Current states equals probabilities
        states = probs

        return probs, states


class GaussianSeluRBM(GaussianRBM):
    """A GaussianSeluRBM class provides the basic implementation for
    Gaussian-SeLU Restricted Boltzmann Machines (for raw pixels values).

    Note that this class requires raw data (integer-valued)
    in order to model the image covariance into a latent ReLU layer.

    References:
        G. Hinton. A practical guide to training restricted Boltzmann machines.
        Neural networks: Tricks of the trade (2012).

        G. Klambauer et al. Self-normalizing neural networks.
        Proceedings, NIPS (2017).
    """

    def __init__(
        self,
        n_visible: Optional[int] = 128,
        n_hidden: Optional[int] = 128,
        steps: Optional[int] = 1,
        learning_rate: Optional[float] = 0.001,
        momentum: Optional[float] = 0.0,
        decay: Optional[float] = 0.0,
        temperature: Optional[float] = 1.0,
        use_gpu: Optional[bool] = False,
        normalize: Optional[bool] = False,
        input_normalize: Optional[bool] = True,
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
            use_gpu: Whether GPU should be used or not.
            normalize: Whether or not to use batch normalization.
            input_normalize: Whether or not to normalize inputs.

        """

        logger.info("Overriding class: GaussianRBM -> GaussianSeluRBM.")

        # Override its parent class
        super(GaussianSeluRBM, self).__init__(
            n_visible,
            n_hidden,
            steps,
            learning_rate,
            momentum,
            decay,
            temperature,
            use_gpu,
            normalize,
            input_normalize,
        )

        logger.info("Class overrided.")

    def hidden_sampling(
        self, v: torch.Tensor, scale: Optional[bool] = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the hidden layer sampling, i.e., P(h|v).

        Args:
            v: A tensor incoming from the visible layer.
            scale: A boolean to decide whether temperature should be used or not.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor]): The probabilities and states of the hidden layer sampling.

        """

        activations = F.linear(v, self.W.t(), self.b)

        if scale:
            probs = F.selu(torch.div(activations, self.T))
        else:
            probs = F.selu(activations)

        # Current states equals probabilities
        states = probs

        return probs, states


class VarianceGaussianRBM(RBM):
    """A VarianceGaussianRBM class provides the basic implementation for
    Gaussian-Bernoulli Restricted Boltzmann Machines (without standardization).

    Note that this class implements a new cost function that takes in account
    a new learning parameter: variance (sigma).

    Therefore, there is no need to standardize the data, as the variance
    will be trained throughout the learning procedure.

    References:
        K. Cho, A. Ilin, T. Raiko.
        Improved learning of Gaussian-Bernoulli restricted Boltzmann machines.
        International conference on artificial neural networks (2011).

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
            use_gpu: Whether GPU should be used or not.

        """

        logger.info("Overriding class: RBM -> VarianceGaussianRBM.")

        # Override its parent class
        super(VarianceGaussianRBM, self).__init__(
            n_visible,
            n_hidden,
            steps,
            learning_rate,
            momentum,
            decay,
            temperature,
            use_gpu,
        )

        self.sigma = nn.Parameter(torch.ones(n_visible))
        self.optimizer.add_param_group({"params": self.sigma})

        if self.device == "cuda":
            self.cuda()

        logger.info("Class overrided.")

    @property
    def sigma(self) -> torch.nn.Parameter:
        """torch.nn.Parameter: Variance parameter."""

        return self._sigma

    @sigma.setter
    def sigma(self, sigma: torch.nn.Parameter) -> None:
        self._sigma = sigma

    def hidden_sampling(
        self, v: torch.Tensor, scale: Optional[bool] = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the hidden layer sampling, i.e., P(h|v).

        Args:
            v: A tensor incoming from the visible layer.
            scale: A boolean to decide whether temperature should be used or not.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor]): The probabilities and states of the hidden layer sampling.

        """

        activations = F.linear(
            torch.div(v, torch.pow(self.sigma, 2)), self.W.t(), self.b
        )

        if scale:
            probs = torch.sigmoid(torch.div(activations, self.T))
        else:
            probs = torch.sigmoid(activations)

        states = torch.bernoulli(probs)

        return probs, states

    def visible_sampling(
        self, h: torch.Tensor, scale: Optional[bool] = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the visible layer sampling, i.e., P(v|h).

        Args:
            h: A tensor incoming from the hidden layer.
            scale: A boolean to decide whether temperature should be used or not.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor]): The probabilities and states of the visible layer sampling.

        """

        activations = F.linear(h, self.W, self.a)

        if self.device == "cpu":
            # If on cpu, variance needs to have size equal to (batch_size, n_visible)
            sigma = torch.repeat_interleave(self.sigma, activations.size(0), dim=0)
        else:
            # Variance needs to have size equal to (n_visible)
            sigma = self.sigma

        states = torch.normal(activations, torch.pow(sigma, 2))

        return states, activations

    def energy(self, samples: torch.Tensor) -> torch.Tensor:
        """Calculates and frees the system's energy.

        Args:
            samples: Samples to be energy-freed.

        Returns:
            (torch.Tensor): The system's energy based on input samples.

        """

        sigma = torch.pow(self.sigma, 2)
        activations = F.linear(torch.div(samples, sigma), self.W.t(), self.b)

        # Createa a Softplus function for numerical stability
        s = nn.Softplus()

        h = torch.sum(s(activations), dim=1)
        v = torch.sum(torch.div(torch.pow(samples - self.a, 2), 2 * sigma), dim=1)

        energy = -v - h

        return energy

class GaussianRBM4deep(GaussianRBM):
    """A GaussianRBM class provides the basic implementation for
    Gaussian-Bernoulli Restricted Boltzmann Machines (with standardization).

    Note that this classes normalize the data
    as it uses variance equals to one throughout its learning procedure.

    This is a trick to ease the calculations of the hidden and
    visible layer samplings, as well as the cost function.

    References:
        K. Cho, A. Ilin, T. Raiko.
        Improved learning of Gaussian-Bernoulli restricted Boltzmann machines.
        International conference on artificial neural networks (2011).

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
        use_gpu: Optional[bool] = False,
        normalize: Optional[bool] = True,
        input_normalize: Optional[bool] = True,
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
            use_gpu: Whether GPU should be used or not.
            normalize: Whether or not to use batch normalization.
            input_normalize: Whether or not to normalize inputs.

        """

        self._normalize = normalize
        self._input_normalize = input_normalize

        logger.info("Overriding class: RBM -> GaussianRBM.")

        super(GaussianRBM4deep, self).__init__(
            n_visible,
            n_hidden,
            steps,
            learning_rate,
            momentum,
            decay,
            temperature,
            use_gpu,
        )

        logger.info("Class overrided.")

  
    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: Optional[int] = 128,
        epochs: Optional[int] = 1,
    ) -> Tuple[float, float]:
        """Fits a new GaussianRBM model.

        Args:
            dataset: A Dataset object containing the training data.
            batch_size: Amount of samples per batch.
            epochs: Number of training epochs.

        Returns:
            (Tuple[float, float]): MSE (mean squared error) and log pseudo-likelihood from the training step.

        """

        batches = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        for epoch in range(epochs):
            start = time.time()
            mse = 0
            pl = 0

            for _, (samples, _) in enumerate(batches):
                if self.normalize:
                    samples = (
                        (samples - torch.mean(samples, 0, True))
                        / (torch.std(samples, 0, True) + 1e-6)
                    ).detach()

                samples = samples.reshape(len(samples), self.n_visible)
                if self.device == "cuda":
                    samples = samples.cuda()

                # Performs the Gibbs sampling procedure
                _, _, _, _, visible_states = self.gibbs_sampling(samples)
                visible_states = visible_states.detach()

                cost = torch.mean(self.energy(samples)) - torch.mean(
                    self.energy(visible_states)
                )

                self.optimizer.zero_grad()
                cost.backward()
                self.optimizer.step()

                batch_size = samples.size(0)

                batch_mse = torch.div(
                    torch.sum(torch.pow(samples - visible_states, 2)), batch_size
                ).detach()
                batch_pl = self.pseudo_likelihood(samples).detach()

                mse += batch_mse
                pl += batch_pl

            mse /= len(batches)
            pl /= len(batches)

            end = time.time()

            self.dump(mse=mse.item(), pl=pl.item(), time=end - start)

        return mse, pl


class GaussianReluRBM4deep(GaussianRBM4deep):
    """A GaussianReluRBM class provides the basic implementation for
    Gaussian-ReLU Restricted Boltzmann Machines (for raw pixels values).

    Note that this class requires raw data (integer-valued)
    in order to model the image covariance into a latent ReLU layer.

    References:
        G. Hinton. A practical guide to training restricted Boltzmann machines.
        Neural networks: Tricks of the trade (2012).

    """

    def __init__(
        self,
        n_visible: Optional[int] = 128,
        n_hidden: Optional[int] = 128,
        steps: Optional[int] = 1,
        learning_rate: Optional[float] = 0.001,
        momentum: Optional[float] = 0.0,
        decay: Optional[float] = 0.0,
        temperature: Optional[float] = 1.0,
        use_gpu: Optional[bool] = False,
        normalize: Optional[bool] = True,
        input_normalize: Optional[bool] = True,
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
            use_gpu: Whether GPU should be used or not.
            normalize: Whether or not to use batch normalization.
            input_normalize: Whether or not to normalize inputs.

        """

        logger.info("Overriding class: GaussianRBM -> GaussianReluRBM.")

        # Override its parent class
        super(GaussianReluRBM4deep, self).__init__(
            n_visible,
            n_hidden,
            steps,
            learning_rate,
            momentum,
            decay,
            temperature,
            use_gpu,
            normalize,
            input_normalize,
        )

        logger.info("Class overrided.")

    def hidden_sampling(
        self, v: torch.Tensor, scale: Optional[bool] = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the hidden layer sampling, i.e., P(h|v).

        Args:
            v: A tensor incoming from the visible layer.
            scale: A boolean to decide whether temperature should be used or not.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor]): The probabilities and states of the hidden layer sampling.

        """

        activations = F.linear(v, self.W.t(), self.b)

        if scale:
            probs = F.relu(torch.div(activations, self.T))
        else:
            probs = F.relu(activations)

        # Current states equals probabilities
        states = probs

        return probs, states