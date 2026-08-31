"""Bernoulli-Bernoulli Restricted Boltzmann Machine."""

import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.data import DataLoader

import learnergy.utils.exception as e
from learnergy.core.model import Model, _validated_property


class RBM(Model):
    """An RBM class provides the basic implementation for Bernoulli-Bernoulli Restricted Boltzmann Machines.

    References:
        G. Hinton. A practical guide to training restricted Boltzmann machines.
        Neural networks: Tricks of the trade (2012).

    """

    n_visible = _validated_property(
        "n_visible",
        lambda _, value: value > 0,
        e.ValueError,
        "`n_visible` should be > 0",
    )
    n_hidden = _validated_property(
        "n_hidden", lambda _, value: value > 0, e.ValueError, "`n_hidden` should be > 0"
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
    T = _validated_property(
        "T", lambda _, value: value > 0, e.ValueError, "`T` should be > 0"
    )
    W = _validated_property("W")
    a = _validated_property("a")
    b = _validated_property("b")
    optimizer = _validated_property("optimizer")

    def __init__(
        self,
        n_visible: int = 128,
        n_hidden: int = 128,
        steps: int = 1,
        learning_rate: float = 0.1,
        momentum: float = 0.0,
        decay: float = 0.0,
        temperature: float = 1.0,
        use_gpu: bool = False,
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

        super().__init__(use_gpu=use_gpu)

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        self.steps = steps
        self.lr = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.T = temperature

        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.a = nn.Parameter(torch.zeros(n_visible))
        self.b = nn.Parameter(torch.zeros(n_hidden))

        self.to(self.device)
        self.optimizer = opt.SGD(
            self.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay
        )

    def pre_activation(self, v: torch.Tensor, scale: bool = False) -> torch.Tensor:
        """Performs the pre-activation over hidden neurons, i.e., Wx' + b.

        Args:
            v: A tensor incoming from the visible layer.
            scale: A boolean to decide whether temperature should be used or not.

        Returns:
            An input for any type of activation function.

        """

        activations = F.linear(v, self.W.t(), self.b)

        if scale:
            activations = torch.div(activations, self.T)

        return activations

    def hidden_sampling(self, v: torch.Tensor, scale: bool = False) -> torch.Tensor:
        """Performs the hidden layer sampling, i.e., P(h|v).

        Args:
            v: A tensor incoming from the visible layer.
            scale: A boolean to decide whether temperature should be used or not.

        Returns:
            The probabilities and states of the hidden layer sampling.

        """

        activations = F.linear(v, self.W.t(), self.b)

        if scale:
            probs = torch.sigmoid(torch.div(activations, self.T))
        else:
            probs = torch.sigmoid(activations)

        states = torch.bernoulli(probs)

        return probs, states

    def visible_sampling(self, h: torch.Tensor, scale: bool = False) -> torch.Tensor:
        """Performs the visible layer sampling, i.e., P(v|h).

        Args:
            h: A tensor incoming from the hidden layer.
            scale: A boolean to decide whether temperature should be used or not.

        Returns:
            The probabilities and states of the visible layer sampling.

        """

        activations = F.linear(h, self.W, self.a)

        if scale:
            probs = torch.sigmoid(torch.div(activations, self.T))
        else:
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

        pos_hidden_probs, pos_hidden_states = self.hidden_sampling(v)
        neg_hidden_states = pos_hidden_states

        # Performing the Contrastive Divergence
        for _ in range(self.steps):
            _, visible_states = self.visible_sampling(neg_hidden_states, True)
            neg_hidden_probs, neg_hidden_states = self.hidden_sampling(
                visible_states, True
            )

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

        activations = F.linear(samples, self.W.t(), self.b)

        # Creates a Softplus function for numerical stability
        s = nn.Softplus()

        h = torch.sum(s(activations), dim=1)
        v = torch.mv(samples, self.a)

        energy = -v - h

        return energy

    def pseudo_likelihood(self, samples: torch.Tensor) -> torch.Tensor:
        """Calculates the logarithm of the pseudo-likelihood.

        Args:
            samples: Samples to be calculated.

        Returns:
            The logarithm of the pseudo-likelihood based on input samples.

        """

        # Calculates the energy of samples before flipping the bits
        samples_binary = torch.round(samples)
        energy = self.energy(samples_binary)

        # Samples an array of indexes to flip the bits
        indexes = torch.randint(
            0, self.n_visible, size=(samples.size(0), 1), device=self.device
        )
        bits = torch.zeros(samples.size(0), samples.size(1), device=self.device)
        bits = bits.scatter_(1, indexes, 1)

        # Calculates the energy after flipping the bits
        samples_binary = torch.where(bits == 0, samples_binary, 1 - samples_binary)
        energy1 = self.energy(samples_binary)

        # Calculate the logarithm of the pseudo-likelihood
        pl = torch.mean(
            self.n_visible
            * torch.log(
                torch.sigmoid(energy1 - energy) + torch.finfo(samples.dtype).eps
            )
        )

        return pl

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 128,
        epochs: int = 10,
    ) -> Tuple[float, float]:
        """Fits a new RBM model.

        Args:
            dataset: A Dataset object containing the training data.
            batch_size: Amount of samples per batch.
            epochs: Number of training epochs.

        Returns:
            MSE (mean squared error) and log pseudo-likelihood from the training step.

        """

        batches = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        for _ in range(epochs):
            start = time.time()

            mse, pl = 0, 0

            for samples, _ in batches:
                samples = samples.reshape(len(samples), self.n_visible).to(self.device)

                _, _, _, _, visible_states = self.gibbs_sampling(samples)
                visible_states = visible_states.detach()

                cost = torch.mean(self.energy(samples)) - torch.mean(
                    self.energy(visible_states)
                )

                self.optimizer.zero_grad()
                cost.backward()
                self.optimizer.step()

                batch_mse = torch.div(
                    torch.sum(torch.pow(samples - visible_states, 2)), samples.size(0)
                ).detach()
                batch_pl = self.pseudo_likelihood(samples).detach()

                mse += batch_mse
                pl += batch_pl

            mse /= len(batches)
            pl /= len(batches)

            end = time.time()

            self.dump(mse=mse.item(), pl=pl.item(), time=end - start)

        return mse, pl

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

        batches = DataLoader(
            dataset, batch_size=len(dataset), shuffle=False, num_workers=0
        )

        for samples, _ in batches:
            samples = samples.reshape(len(samples), self.n_visible).to(self.device)

            _, pos_hidden_states = self.hidden_sampling(samples)
            visible_probs, visible_states = self.visible_sampling(pos_hidden_states)

            batch_mse = torch.div(
                torch.sum(torch.pow(samples - visible_states, 2)), samples.size(0)
            )
            mse += batch_mse

        mse /= len(batches)

        return mse, visible_probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass over the data.

        Args:
            x: An input tensor for computing the forward pass.

        Returns:
            A tensor containing the RBM's outputs.

        """

        x, _ = self.hidden_sampling(x)

        return x
