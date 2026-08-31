"""Convolutional Bernoulli-Bernoulli Restricted Boltzmann Machine."""

import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.data import DataLoader

import learnergy.utils.exception as e
from learnergy.core.model import Model, _validated_property


class ConvRBM(Model):
    """A ConvRBM class provides the basic implementation for
    Convolutional Bernoulli-Bernoulli Restricted Boltzmann Machines.

    References:
        H. Lee, et al.
        Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations.
        Proceedings of the 26th annual international conference on machine learning (2009).

    """

    visible_shape = _validated_property("visible_shape")
    filter_shape = _validated_property(
        "filter_shape",
        lambda self, value: all(
            filter_size < visible_size
            for filter_size, visible_size in zip(value, self.visible_shape)
        ),
        e.ValueError,
        "`filter_shape` should be smaller than `visible_shape`",
    )
    hidden_shape = _validated_property("hidden_shape")
    n_filters = _validated_property(
        "n_filters",
        lambda _, value: value > 0,
        e.ValueError,
        "`n_filters` should be > 0",
    )
    n_channels = _validated_property(
        "n_channels",
        lambda _, value: value > 0,
        e.ValueError,
        "`n_channels` should be > 0",
    )
    steps = _validated_property(
        "steps", lambda _, value: value > 0, e.ValueError, "`steps` should be > 0"
    )
    lr = _validated_property(
        "lr", lambda _, value: value >= 0, e.ValueError, "`lr` should be >= 0"
    )
    momentum = _validated_property(
        "momentum",
        lambda _, value: value >= 0,
        e.ValueError,
        "`momentum` should be >= 0",
    )
    decay = _validated_property(
        "decay", lambda _, value: value >= 0, e.ValueError, "`decay` should be >= 0"
    )
    maxpooling = _validated_property(
        "maxpooling",
        lambda _, value: isinstance(value, bool),
        e.ValueError,
        "`maxpooling` should be True or False",
    )
    W = _validated_property("W")
    a = _validated_property("a")
    b = _validated_property("b")
    optimizer = _validated_property("optimizer")

    def __init__(
        self,
        visible_shape: Tuple[int, int] = (28, 28),
        filter_shape: Tuple[int, int] = (7, 7),
        n_filters: int = 5,
        n_channels: int = 1,
        steps: int = 1,
        learning_rate: float = 0.1,
        momentum: float = 0.0,
        decay: float = 0.0,
        maxpooling: bool = False,
        pooling_kernel: int = 2,
        use_gpu: bool = False,
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
            pooling_kernel: The kernel size of MaxPooling2D layer (when maxpooling=True).
            use_gpu: Whether GPU should be used or not.

        """

        super().__init__(use_gpu=use_gpu)

        if len(visible_shape) != 2 or min(visible_shape) <= 0:
            raise e.ValueError("`visible_shape` should contain two positive values")
        if len(filter_shape) != 2 or min(filter_shape) <= 0:
            raise e.ValueError("`filter_shape` should contain two positive values")
        if pooling_kernel <= 0:
            raise e.ValueError("`pooling_kernel` should be > 0")

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

        self.maxpooling = maxpooling
        self.maxpol2d = (
            nn.MaxPool2d(kernel_size=pooling_kernel, stride=2, padding=1)
            if maxpooling
            else None
        )

        self.W = nn.Parameter(
            torch.randn(n_filters, n_channels, filter_shape[0], filter_shape[1]) * 0.01
        )
        self.a = nn.Parameter(torch.zeros(n_channels))
        self.b = nn.Parameter(torch.zeros(n_filters))

        self.to(self.device)
        self.optimizer = opt.SGD(
            self.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay
        )

    def hidden_sampling(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the hidden layer sampling, i.e., P(h|v).

        Args:
            v: A tensor incoming from the visible layer.

        Returns:
            The probabilities and states of the hidden layer sampling.

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
            The probabilities and states of the visible layer sampling.

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
            The probabilities and states of the hidden layer sampling (positive),
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
            The system's energy based on input samples.

        """

        activations = F.conv2d(samples, self.W, bias=self.b)

        # Creates a Softplus function for numerical stability
        s = nn.Softplus()

        h = torch.sum(s(activations), dim=(1, 2, 3))
        v = torch.sum(samples * self.a.view(1, -1, 1, 1), dim=(1, 2, 3))

        energy = -v - h

        return energy

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 128,
        epochs: int = 10,
    ) -> float:
        """Fits a new ConvRBM model.

        Args:
            dataset: A Dataset object containing the training data.
            batch_size: Amount of samples per batch.
            epochs: Number of training epochs.

        Returns:
            MSE (mean squared error) from the training step.

        """

        batches = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        for _ in range(epochs):
            start = time.time()

            mse = 0

            for samples, _ in batches:
                samples = samples.reshape(
                    len(samples),
                    self.n_channels,
                    self.visible_shape[0],
                    self.visible_shape[1],
                ).to(self.device)

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

        return mse

    def reconstruct(
        self, dataset: torch.utils.data.Dataset
    ) -> Tuple[float, torch.Tensor]:
        """Reconstructs batches of new samples.

        Args:
            dataset: A Dataset object containing the testing data.

        Returns:
            Reconstruction error and visible probabilities, i.e., P(v|h).

        """

        mse = 0

        batch_size = len(dataset)
        batches = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        for samples, _ in batches:
            samples = samples.reshape(
                len(samples),
                self.n_channels,
                self.visible_shape[0],
                self.visible_shape[1],
            ).to(self.device)

            _, pos_hidden_states = self.hidden_sampling(samples)
            visible_probs, visible_states = self.visible_sampling(pos_hidden_states)

            batch_mse = torch.div(
                torch.sum(torch.pow(samples - visible_states, 2)), batch_size
            )
            mse += batch_mse

        mse /= len(batches)

        return mse, visible_probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass over the data.

        Args:
            x: An input tensor for computing the forward pass.

        Returns:
            A tensor containing the Convolutional RBM's outputs.

        """

        x, _ = self.hidden_sampling(x)
        if self.maxpooling:
            x = self.maxpol2d(x)

        return x
