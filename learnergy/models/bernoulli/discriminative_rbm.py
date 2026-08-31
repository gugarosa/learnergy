"""Discriminative Bernoulli-Bernoulli Restricted Boltzmann Machine."""

import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import learnergy.utils.exception as e
from learnergy.core.model import _validated_property
from learnergy.models.bernoulli.rbm import RBM


class DiscriminativeRBM(RBM):
    """A DiscriminativeRBM class provides the basic implementation for
    Discriminative Bernoulli-Bernoulli Restricted Boltzmann Machines.

    References:
        H. Larochelle and Y. Bengio. Classification using discriminative restricted Boltzmann machines.
        Proceedings of the 25th international conference on Machine learning (2008).

    """

    n_classes = _validated_property(
        "n_classes",
        lambda _, value: value > 0,
        e.ValueError,
        "`n_classes` should be > 0",
    )
    U = _validated_property("U")
    c = _validated_property("c")
    loss = _validated_property("loss")

    def __init__(
        self,
        n_visible: int = 128,
        n_hidden: int = 128,
        n_classes: int = 1,
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
            n_classes: Amount of classes.
            steps: Number of Gibbs' sampling steps.
            learning_rate: Learning rate.
            momentum: Momentum parameter.
            decay: Weight decay used for penalization.
            temperature: Temperature factor.
            use_gpu: Whether GPU should be used or not.

        """

        super().__init__(
            n_visible,
            n_hidden,
            steps,
            learning_rate,
            momentum,
            decay,
            temperature,
            use_gpu,
        )

        self.n_classes = n_classes

        self.U = nn.Parameter(torch.randn(n_classes, n_hidden) * 0.05)
        self.c = nn.Parameter(torch.zeros(n_classes))
        self.loss = nn.CrossEntropyLoss()

        self.to(self.device)
        self.optimizer.add_param_group({"params": self.U})
        self.optimizer.add_param_group({"params": self.c})

    def labels_sampling(self, samples: torch.Tensor) -> torch.Tensor:
        """Calculates labels probabilities by samplings, i.e., P(y|v).

        Args:
            samples: Samples to be labels-calculated.

        Returns:
            Labels' probabilities based on input samples.

        """

        probs = torch.zeros(samples.size(0), self.n_classes, device=self.device)
        activations = F.linear(samples, self.W.t(), self.b)

        for i in range(self.n_classes):
            # Calculates the logit-probability for the particular class
            probs[:, i] = self.c[i] + torch.sum(
                F.softplus(activations + self.U[i, :]), dim=1
            )

        preds = torch.argmax(probs.detach(), 1)

        return probs, preds

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 128,
        epochs: int = 10,
    ) -> Tuple[float, float]:
        """Fits a new DRBM model.

        Args:
            dataset: A Dataset object containing the training data.
            batch_size: Amount of samples per batch.
            epochs: Number of training epochs.

        Returns:
            Loss and accuracy from the training step.

        """

        batches = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        for _ in range(epochs):
            start = time.time()

            loss = 0
            acc = 0

            for samples, labels in batches:
                samples = samples.reshape(len(samples), self.n_visible).to(self.device)
                labels = labels.to(self.device)

                probs, _ = self.labels_sampling(samples)
                cost = self.loss(probs, labels)

                self.optimizer.zero_grad()
                cost.backward()
                self.optimizer.step()

                _, preds = self.labels_sampling(samples)

                batch_acc = (preds == labels).float().mean()

                loss += cost.detach()
                acc += batch_acc

            loss /= len(batches)
            acc /= len(batches)

            end = time.time()

            self.dump(loss=loss.item(), acc=acc.item(), time=end - start)

        return loss, acc

    def predict(
        self, dataset: torch.utils.data.Dataset
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """Predicts batches of new samples.

        Args:
            dataset: A Dataset object containing the testing data.

        Returns:
            Accuracy, prediction probabilities and labels, i.e., P(y|v).

        """

        acc = 0
        batch_size = len(dataset)

        batches = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        for samples, labels in batches:
            samples = samples.reshape(len(samples), self.n_visible).to(self.device)
            labels = labels.to(self.device)

            probs, preds = self.labels_sampling(samples)

            batch_acc = (preds == labels).float().mean()
            acc += batch_acc

        acc /= len(batches)

        return acc, probs, preds


class HybridDiscriminativeRBM(DiscriminativeRBM):
    """A HybridDiscriminativeRBM class provides the basic implementation for
    Hybrid Discriminative Bernoulli-Bernoulli Restricted Boltzmann Machines.

    References:
        H. Larochelle and Y. Bengio. Classification using discriminative restricted Boltzmann machines.
        Proceedings of the 25th international conference on Machine learning (2008).

    """

    alpha = _validated_property(
        "alpha", lambda _, value: value >= 0, e.ValueError, "`alpha` should be >= 0"
    )

    def __init__(
        self,
        n_visible: int = 128,
        n_hidden: int = 128,
        n_classes: int = 1,
        steps: int = 1,
        learning_rate: float = 0.1,
        alpha: float = 0.01,
        momentum: float = 0.0,
        decay: float = 0.0,
        temperature: float = 1.0,
        use_gpu: bool = False,
    ) -> None:
        """Initialization method.

        Args:
            n_visible: Amount of visible units.
            n_hidden: Amount of hidden units.
            n_classes: Amount of classes.
            steps: Number of Gibbs' sampling steps.
            learning_rate: Learning rate.
            alpha: Amount of penalization to the generative loss.
            momentum: Momentum parameter.
            decay: Weight decay used for penalization.
            temperature: Temperature factor.
            use_gpu: Whether GPU should be used or not.

        """

        super().__init__(
            n_visible,
            n_hidden,
            n_classes,
            steps,
            learning_rate,
            momentum,
            decay,
            temperature,
            use_gpu,
        )

        self.alpha = alpha

    def hidden_sampling(
        self, v: torch.Tensor, y: torch.Tensor, scale: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the hidden layer sampling, i.e., P(h|y,v).

        Args:
            v: A tensor incoming from the visible layer.
            y: A tensor incoming from the class layer.
            scale: A boolean to decide whether temperature should be used or not.

        Returns:
            The probabilities and states of the hidden layer sampling.

        """

        activations = F.linear(v, self.W.t(), self.b) + torch.matmul(y, self.U)

        if scale:
            probs = torch.sigmoid(torch.div(activations, self.T))
        else:
            probs = torch.sigmoid(activations)

        states = torch.bernoulli(probs)

        return probs, states

    def class_sampling(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the class layer sampling, i.e., P(y|h).

        Args:
            h: A tensor incoming from the hidden layer.

        Returns:
            The probabilities and states of the class layer sampling.

        """

        activations = torch.exp(F.linear(h, self.U, self.c))
        probs = torch.div(activations, torch.sum(activations, dim=1).unsqueeze(1))
        states = torch.nn.functional.one_hot(
            torch.argmax(probs, dim=1), num_classes=self.n_classes
        ).float()

        return probs, states

    def gibbs_sampling(
        self, v: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs the whole Gibbs sampling procedure.

        Args:
            v: A tensor incoming from the visible layer.
            y: A tensor incoming from the class layer.

        Returns:
            The probabilities and states of the hidden layer sampling (positive),
                the probabilities and states of the hidden layer sampling (negative)
                and the states of the visible layer sampling (negative).

        """

        y = torch.nn.functional.one_hot(y, num_classes=self.n_classes).float()

        pos_hidden_probs, pos_hidden_states = self.hidden_sampling(v, y)
        neg_hidden_states = pos_hidden_states

        # Performing the Contrastive Divergence
        for _ in range(self.steps):
            _, visible_states = self.visible_sampling(neg_hidden_states, True)
            _, class_states = self.class_sampling(neg_hidden_states)

            neg_hidden_probs, neg_hidden_states = self.hidden_sampling(
                visible_states, class_states, True
            )

        return (
            pos_hidden_probs,
            pos_hidden_states,
            neg_hidden_probs,
            neg_hidden_states,
            visible_states,
        )

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 128,
        epochs: int = 10,
    ) -> Tuple[float, float]:
        """Fits a new DRBM model.

        Args:
            dataset: A Dataset object containing the training data.
            batch_size: Amount of samples per batch.
            epochs: Number of training epochs.

        Returns:
            Loss and accuracy from the training step.

        """

        batches = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        for _ in range(epochs):
            start = time.time()

            d_loss, g_loss, loss, acc = 0, 0, 0, 0

            for samples, labels in batches:
                samples = samples.reshape(len(samples), self.n_visible).to(self.device)
                labels = labels.to(self.device)

                _, _, _, _, visible_states = self.gibbs_sampling(samples, labels)
                visible_states = visible_states.detach()
                disc_probs, _ = self.labels_sampling(samples)

                d_cost = self.loss(disc_probs, labels)
                g_cost = -self.pseudo_likelihood(samples)
                cost = d_cost + self.alpha * g_cost

                self.optimizer.zero_grad()
                cost.backward()
                self.optimizer.step()

                _, preds = self.labels_sampling(samples)

                batch_acc = (preds == labels).float().mean()

                d_loss += d_cost
                g_loss += g_cost
                loss += cost.detach()
                acc += batch_acc

            d_loss /= len(batches)
            g_loss /= len(batches)
            loss /= len(batches)
            acc /= len(batches)

            end = time.time()

            self.dump(
                d_loss=d_loss.item(),
                g_loss=g_loss.item(),
                loss=loss.item(),
                acc=acc.item(),
                time=end - start,
            )

        return loss, acc
