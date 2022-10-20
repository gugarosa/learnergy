"""Convolutional Bernoulli-Bernoulli Restricted Boltzmann Machine.
"""

import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.data import DataLoader
from tqdm import tqdm

import learnergy.utils.exception as e
from learnergy.core import Model
from learnergy.utils import logging

logger = logging.get_logger(__name__)


class ConvRBM(Model):
    """A ConvRBM class provides the basic implementation for
    Convolutional Bernoulli-Bernoulli Restricted Boltzmann Machines.

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
            maxpooling: Whether MaxPooling2D should be used or not.
            pooling_kernel: The kernel size of MaxPooling layer (when maxpooling=True).
            use_gpu: Whether GPU should be used or not.

        """

        logger.info("Overriding class: Model -> ConvRBM.")

        super(ConvRBM, self).__init__(use_gpu=use_gpu)

        self.visible_shape = visible_shape
        self.filter_shape = filter_shape
        self.hidden_shape = (
            visible_shape[0] - filter_shape[0] + 1,
            visible_shape[1] - filter_shape[1] + 1,
        )

        self.n_filters = n_filters
        self.n_channels = n_channels

        self.steps = steps
        self.lr = learning_rate
        self.momentum = momentum
        self.decay = decay

        if maxpooling:
            self.maxpol2d = nn.MaxPool2d(kernel_size=pooling_kernel, stride=2, padding=1)
            self.maxpooling = True
        else:
            self.maxpol2d = maxpooling
            self.maxpooling = False

        self.W = nn.Parameter(
            torch.randn(n_filters, n_channels, filter_shape[0], filter_shape[1]) * 0.01
        )
        self.a = nn.Parameter(torch.zeros(n_channels))
        self.b = nn.Parameter(torch.zeros(n_filters))

        self.optimizer = opt.SGD(
            self.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay
        )

        if self.device == "cuda":
            self.cuda()

        logger.info("Class overrided.")
        logger.debug(
            "Visible: %s | Filters: %d x %s | Hidden: %s | "
            "Channels: %d | Learning: CD-%d | "
            "Hyperparameters: lr = %s, momentum = %s, decay = %s | "
            "Pooling: MaxPooling2D = %s: (%s, %s).",
            self.visible_shape,
            self.n_filters,
            self.filter_shape,
            self.hidden_shape,
            self.n_channels,
            self.steps,
            self.lr,
            self.momentum,
            self.decay,
            self.maxpooling,
            pooling_kernel, pooling_kernel,
        )

    @property
    def visible_shape(self) -> Tuple[int, int]:
        """Shape of visible units."""

        return self._visible_shape

    @visible_shape.setter
    def visible_shape(self, visible_shape: Tuple[int, int]) -> None:
        self._visible_shape = visible_shape

    @property
    def filter_shape(self) -> Tuple[int, int]:
        """Shape of filters."""

        return self._filter_shape

    @filter_shape.setter
    def filter_shape(self, filter_shape: Tuple[int, int]) -> None:
        if (filter_shape[0] >= self.visible_shape[0]) or (
            filter_shape[1] >= self.visible_shape[1]
        ):
            raise e.ValueError("`filter_shape` should be smaller than `visible_shape`")

        self._filter_shape = filter_shape

    @property
    def hidden_shape(self) -> Tuple[int, int]:
        """Shape of hidden units."""

        return self._hidden_shape

    @hidden_shape.setter
    def hidden_shape(self, hidden_shape: Tuple[int, int]) -> None:
        self._hidden_shape = hidden_shape

    @property
    def n_filters(self) -> int:
        """Number of filters."""

        return self._n_filters

    @n_filters.setter
    def n_filters(self, n_filters: int) -> None:
        if n_filters <= 0:
            raise e.ValueError("`n_filters` should be > 0")

        self._n_filters = n_filters

    @property
    def n_channels(self) -> int:
        """Number of channels."""

        return self._n_channels

    @n_channels.setter
    def n_channels(self, n_channels: int) -> None:
        if n_channels <= 0:
            raise e.ValueError("`n_channels` should be > 0")

        self._n_channels = n_channels

    @property
    def steps(self) -> int:
        """Number of steps Gibbs' sampling steps."""

        return self._steps

    @steps.setter
    def steps(self, steps: int) -> None:
        if steps <= 0:
            raise e.ValueError("`steps` should be > 0")

        self._steps = steps

    @property
    def lr(self) -> float:
        """Learning rate."""

        return self._lr

    @lr.setter
    def lr(self, lr: float) -> None:
        if lr < 0:
            raise e.ValueError("`lr` should be >= 0")

        self._lr = lr

    @property
    def momentum(self) -> float:
        """Momentum parameter."""

        return self._momentum

    @momentum.setter
    def momentum(self, momentum: float) -> None:
        if momentum < 0:
            raise e.ValueError("`momentum` should be >= 0")

        self._momentum = momentum

    @property
    def decay(self) -> float:
        """Weight decay."""

        return self._decay

    @decay.setter
    def decay(self, decay: float) -> None:
        if decay < 0:
            raise e.ValueError("`decay` should be >= 0")

        self._decay = decay

    @property
    def maxpooling(self) -> bool:
        """Usage of MaxPooling."""

        return self._maxpooling

    @maxpooling.setter
    def maxpooling(self, maxpooling: bool) -> None:
        if type(maxpooling) != bool:
            raise e.ValueError("`maxpooling` should be True or False")

        self._maxpooling = maxpooling

    @property
    def W(self) -> torch.nn.Parameter:
        """Filters' matrix."""

        return self._W

    @W.setter
    def W(self, W: torch.nn.Parameter) -> None:
        self._W = W

    @property
    def a(self) -> torch.nn.Parameter:
        """Visible units bias."""

        return self._a

    @a.setter
    def a(self, a: torch.nn.Parameter) -> None:
        self._a = a

    @property
    def b(self) -> torch.nn.Parameter:
        """Hidden units bias."""

        return self._b

    @b.setter
    def b(self, b: torch.nn.Parameter) -> None:
        self._b = b

    @property
    def optimizer(self) -> torch.optim.SGD:
        """Stochastic Gradient Descent object."""

        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: torch.optim.SGD) -> None:
        self._optimizer = optimizer

    def hidden_sampling(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the hidden layer sampling, i.e., P(h|v).

        Args:
            v: A tensor incoming from the visible layer.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor]): The probabilities and states of the hidden layer sampling.

        """

        activations = F.conv2d(v, self.W, bias=self.b)
        probs = torch.sigmoid(activations)
        states = torch.bernoulli(probs)

        return probs, states

    def visible_sampling(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the visible layer sampling, i.e., P(v|h).

        Args:
            h: A tensor incoming from the hidden layer.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor]): The probabilities and states of the visible layer sampling.

        """

        activations = F.conv_transpose2d(h, self.W, bias=self.a)
        probs = torch.sigmoid(activations)
        states = torch.bernoulli(probs)

        return probs, states

    def gibbs_sampling(
        self, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs the whole Gibbs sampling procedure.

        Args:
            v: A tensor incoming from the visible layer.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): The probabilities and states of the hidden layer sampling (positive),
                the probabilities and states of the hidden layer sampling (negative)
                and the states of the visible layer sampling (negative).

        """

        # Calculating positive phase hidden probabilities and states
        pos_hidden_probs, pos_hidden_states = self.hidden_sampling(v)

        # Initially defining the negative phase
        neg_hidden_states = pos_hidden_states

        # Performing the Contrastive Divergence
        for _ in range(self.steps):
            # Calculating visible probabilities and states
            _, visible_states = self.visible_sampling(neg_hidden_states)

            # Calculating hidden probabilities and states
            neg_hidden_probs, neg_hidden_states = self.hidden_sampling(visible_states)

        return (
            pos_hidden_probs,
            pos_hidden_states,
            neg_hidden_probs,
            neg_hidden_states,
            visible_states,
        )

    def energy(self, samples: torch.Tensor) -> torch.Tensor:
        """Calculates and frees the system's energy.

        Args:
            samples: Samples to be energy-freed.

        Returns:
            (torch.Tensor): The system's energy based on input samples.

        """

        activations = F.conv2d(samples, self.W, bias=self.b)

        # Creates a Softplus function for numerical stability
        s = nn.Softplus()

        h = torch.sum(s(activations), dim=(1, 2, 3))
        v = torch.sum(samples, dim=(1, 2, 3)) * torch.mean(self.a)

        energy = -v - h

        return energy

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: Optional[int] = 128,
        epochs: Optional[int] = 10,
    ) -> float:
        """Fits a new ConvRBM model.

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

                mse += batch_mse

            mse /= len(batches)

            end = time.time()

            self.dump(mse=mse.item(), time=end - start)

            logger.info("MSE: %f", mse)

        return mse

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

        mse = 0

        batch_size = len(dataset)
        batches = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        for samples, _ in tqdm(batches):
            samples = samples.reshape(
                len(samples),
                self.n_channels,
                self.visible_shape[0],
                self.visible_shape[1],
            )
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
            (torch.Tensor): A tensor containing the Convolutional RBM's outputs.

        """

        x, _ = self.hidden_sampling(x)
        if self.maxpooling:
            x = self.maxpol2d(x)

        return x
