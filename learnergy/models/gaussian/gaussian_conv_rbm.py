"""Gaussian-based Convolutional Restricted Boltzmann Machine.
"""

import time
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import learnergy.utils.constants as c
import learnergy.utils.exception as e
from learnergy.models.bernoulli import ConvRBM
from learnergy.utils import logging

logger = logging.get_logger(__name__)


class GaussianConvRBM(ConvRBM):
    """A GaussianConvRBM class provides the basic implementation for
    Gaussian-based Convolutional Restricted Boltzmann Machines.

    References:
        H. Lee, et al.
        Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations.
        Proceedings of the 26th annual international conference on machine learning (2009).

    """

    def __init__(
        self,
        visible_shape: Optional[Tuple[int, int]] = (28, 28),
        filter_shape: Optional[Tuple[int, int]] = (7, 7),
        n_filters: Optional[int] = 5,
        n_channels: Optional[int] = 1,
        steps: Optional[int] = 1,
        learning_rate: Optional[float] = 0.1,
        momentum: Optional[float] = 0.0,
        decay: Optional[float] = 0.0,
        maxpooling: Optional[bool] = False,
        pooling_kernel: Optional[int] = 2,
        use_gpu: Optional[bool] = False,
    ) -> None:
        """Initialization method.

        Args:
            visible_shape: Shape of visible units.
            filter_shape: Shape of filters.
            n_filters: Number of filters.
            n_channels: Number of channels.
            steps: Number of Gibbs' sampling steps.
            learning_rate: Learning rate.
            momentum: Momentum parameter.
            decay: Weight decay used for penalization.
            use_gpu: Whether GPU should be used or not.

        """

        logger.info("Overriding class: ConvRBM -> GaussianConvRBM.")

        super(GaussianConvRBM, self).__init__(
            visible_shape,
            filter_shape,
            n_filters,
            n_channels,
            steps,
            learning_rate,
            momentum,
            decay,
            maxpooling,
            pooling_kernel,
            use_gpu,
        )

        self.normalize = True

        # Creating a Sigmoid function to employ on sampling
        self.sig = torch.nn.Sigmoid()

        logger.info("Class overrided.")

    @property
    def normalize(self) -> bool:
        """Inner data normalization."""

        return self._normalize

    @normalize.setter
    def normalize(self, normalize: bool) -> None:
        self._normalize = normalize

    def hidden_sampling(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the hidden layer sampling, i.e., P(h|v).

        Args:
            v (torch.Tensor): A tensor incoming from the visible layer.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor]): The probabilities and states of the hidden layer sampling.

        """

        activations = F.conv2d(v, self.W, bias=self.b)
        probs = F.relu6(activations).detach()

        return probs, probs

    def visible_sampling(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the visible layer sampling, i.e., P(v|h).

        Args:
            h: A tensor incoming from the hidden layer.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor]): The probabilities and states of the visible layer sampling.

        """

        activations = F.conv_transpose2d(h, self.W, bias=self.a)

        if self.normalize:
            # Uses the previously calculated activations
            probs = activations.detach()
        else:
            #probs = F.relu6(activations).detach()
            probs = self.sig(activations).detach()

        return probs, probs

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: Optional[int] = 128,
        epochs: Optional[int] = 10,
    ) -> float:
        """Fits a new GaussianConvRBM model.

        Args:
            dataset: A Dataset object containing the training data.
            batch_size: Amount of samples per batch.
            epochs: Number of training epochs.

        Returns:
            (float): MSE (mean squared error) from the training step.

        """

        batches = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        for epoch in range(epochs):
            logger.info("Epoch %d/%d", epoch + 1, epochs)

            start = time.time()

            mse = 0            

            for samples, _ in tqdm(batches):
                samples = samples.reshape(
                    len(samples),
                    self.n_channels,
                    self.visible_shape[0],
                    self.visible_shape[1],
                )
                if self.device == "cuda":
                    samples = samples.cuda()

                if self.normalize:
                    samples = (samples - torch.mean(samples, 0, True)) / (
                        torch.std(samples, 0, True) + c.EPSILON
                    )

                # Performs the Gibbs sampling procedure
                _, _, _, _, visible_states = self.gibbs_sampling(samples)
                visible_states = visible_states.detach()

                cost = torch.mean(self.energy(samples)) - torch.mean(
                    self.energy(visible_states)
                )

                self.optimizer.zero_grad()
                cost.backward()
                self.optimizer.step()

                batch_mse = torch.div(
                    torch.sum(torch.pow(samples - visible_states, 2)), batch_size
                ).detach()
                mse += batch_mse

            mse /= len(batches)

            end = time.time()

            self.dump(mse=mse.item(), time=end - start)

            logger.info("MSE: %f", mse)

        return mse


class GaussianConvRBM4deep(ConvRBM):
    """A GaussianConvRBM class provides the basic implementation for
    Gaussian-based Convolutional Restricted Boltzmann Machines.

    References:
        H. Lee, et al.
        Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations.
        Proceedings of the 26th annual international conference on machine learning (2009).

    """

    def __init__(
        self,
        visible_shape: Optional[Tuple[int, int]] = (28, 28),
        filter_shape: Optional[Tuple[int, int]] = (7, 7),
        n_filters: Optional[int] = 5,
        n_channels: Optional[int] = 1,
        steps: Optional[int] = 1,
        learning_rate: Optional[float] = 0.1,
        momentum: Optional[float] = 0.0,
        decay: Optional[float] = 0.0,
        maxpooling: Optional[bool] = False,
        pooling_kernel: Optional[int] = 2,
        use_gpu: Optional[bool] = False,
    ) -> None:
        """Initialization method.

        Args:
            visible_shape: Shape of visible units.
            filter_shape: Shape of filters.
            n_filters: Number of filters.
            n_channels: Number of channels.
            steps: Number of Gibbs' sampling steps.
            learning_rate: Learning rate.
            momentum: Momentum parameter.
            decay: Weight decay used for penalization.
            use_gpu: Whether GPU should be used or not.

        """

        logger.info("Overriding class: ConvRBM -> GaussianConvRBM.")

        super(GaussianConvRBM4deep, self).__init__(
            visible_shape,
            filter_shape,
            n_filters,
            n_channels,
            steps,
            learning_rate,
            momentum,
            decay,
            maxpooling,
            pooling_kernel,
            use_gpu,
        )

        self.normalize = True

        # Creating a Sigmoid function to employ on sampling
        self.sig = torch.nn.Sigmoid()

        logger.info("Class overrided.")

    @property
    def normalize(self) -> bool:
        """Inner data normalization."""

        return self._normalize

    @normalize.setter
    def normalize(self, normalize: bool) -> None:
        self._normalize = normalize

    def hidden_sampling(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the hidden layer sampling, i.e., P(h|v).

        Args:
            v (torch.Tensor): A tensor incoming from the visible layer.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor]): The probabilities and states of the hidden layer sampling.

        """

        activations = F.conv2d(v, self.W, bias=self.b)
        probs = F.relu6(activations).detach()

        return probs, probs

    def visible_sampling(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the visible layer sampling, i.e., P(v|h).

        Args:
            h: A tensor incoming from the hidden layer.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor]): The probabilities and states of the visible layer sampling.

        """

        activations = F.conv_transpose2d(h, self.W, bias=self.a)

        if self.normalize:
            # Uses the previously calculated activations
            probs = activations.detach()
        else:
            #probs = F.relu6(activations).detach()
            probs = self.sig(activations).detach()

        return probs, probs

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: Optional[int] = 128,
        epochs: Optional[int] = 10,
    ) -> float:
        """Fits a new GaussianConvRBM model.

        Args:
            dataset: A Dataset object containing the training data.
            batch_size: Amount of samples per batch.
            epochs: Number of training epochs.

        Returns:
            (float): MSE (mean squared error) from the training step.

        """

        batches = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        for epoch in range(epochs):
            #logger.info("Epoch %d/%d", epoch + 1, epochs)

            start = time.time()

            mse = 0            

            for _, (samples, _) in enumerate(batches):
                samples = samples.reshape(
                    len(samples),
                    self.n_channels,
                    self.visible_shape[0],
                    self.visible_shape[1],
                )
                if self.device == "cuda":
                    samples = samples.cuda()

                if self.normalize:
                    samples = (samples - torch.mean(samples, 0, True)) / (
                        torch.std(samples, 0, True) + c.EPSILON
                    )

                # Performs the Gibbs sampling procedure
                _, _, _, _, visible_states = self.gibbs_sampling(samples)
                visible_states = visible_states.detach()

                cost = torch.mean(self.energy(samples)) - torch.mean(
                    self.energy(visible_states)
                )

                self.optimizer.zero_grad()
                cost.backward()
                self.optimizer.step()

                batch_mse = torch.div(
                    torch.sum(torch.pow(samples - visible_states, 2)), batch_size
                ).detach()
                mse += batch_mse

            mse /= len(batches)

            end = time.time()

            self.dump(mse=mse.item(), time=end - start)

            #logger.info("MSE: %f", mse)

        return mse