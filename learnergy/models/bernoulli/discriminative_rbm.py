"""Discriminative Bernoulli-Bernoulli Restricted Boltzmann Machine.
"""

import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import learnergy.utils.exception as e
from learnergy.models.bernoulli import RBM
from learnergy.utils import logging

logger = logging.get_logger(__name__)


class DiscriminativeRBM(RBM):
    """A DiscriminativeRBM class provides the basic implementation for
    Discriminative Bernoulli-Bernoulli Restricted Boltzmann Machines.

    References:
        H. Larochelle and Y. Bengio. Classification using discriminative restricted Boltzmann machines.
        Proceedings of the 25th international conference on Machine learning (2008).

    """

    def __init__(
        self,
        n_visible: Optional[int] = 128,
        n_hidden: Optional[int] = 128,
        n_classes: Optional[int] = 1,
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
            n_classes: Amount of classes.
            steps: Number of Gibbs' sampling steps.
            learning_rate: Learning rate.
            momentum: Momentum parameter.
            decay: Weight decay used for penalization.
            temperature: Temperature factor.
            use_gpu: Whether GPU should be used or not.

        """

        logger.info("Overriding class: RBM -> DiscriminativeRBM.")

        super(DiscriminativeRBM, self).__init__(
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

        self.optimizer.add_param_group({"params": self.U})
        self.optimizer.add_param_group({"params": self.c})

        if self.device == "cuda":
            self.cuda()

        logger.info("Class overrided.")

    @property
    def n_classes(self) -> int:
        """Number of classes."""

        return self._n_classes

    @n_classes.setter
    def n_classes(self, n_classes: int) -> None:
        if n_classes <= 0:
            raise e.ValueError("`n_classes` should be > 0")

        self._n_classes = n_classes

    @property
    def U(self) -> torch.nn.Parameter:
        """Class weights' matrix."""

        return self._U

    @U.setter
    def U(self, U: torch.nn.Parameter) -> None:
        self._U = U

    @property
    def c(self) -> torch.nn.Parameter:
        """Class units bias."""

        return self._c

    @c.setter
    def c(self, c: torch.nn.Parameter) -> None:
        self._c = c

    @property
    def loss(self) -> torch.nn.CrossEntropyLoss:
        """Cross-Entropy loss function."""

        return self._loss

    @loss.setter
    def loss(self, loss: torch.nn.CrossEntropyLoss) -> None:
        self._loss = loss

    def labels_sampling(self, samples: torch.Tensor) -> torch.Tensor:
        """Calculates labels probabilities by samplings, i.e., P(y|v).

        Args:
            samples: Samples to be labels-calculated.

        Returns:
            (torch.Tensor): Labels' probabilities based on input samples.

        """

        probs = torch.zeros(samples.size(0), self.n_classes, device=self.device)
        activations = F.linear(samples, self.W.t(), self.b)

        # Creates a Softplus function for numerical stability
        s = nn.Softplus()

        for i in range(self.n_classes):
            # Calculates the logit-probability for the particular class
            probs[:, i] = self.c[i] + torch.sum(s(activations + self.U[i, :]), dim=1)

        preds = torch.argmax(probs.detach(), 1)

        return probs, preds

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: Optional[int] = 128,
        epochs: Optional[int] = 10,
    ) -> Tuple[float, float]:
        """Fits a new DRBM model.

        Args:
            dataset: A Dataset object containing the training data.
            batch_size: Amount of samples per batch.
            epochs: Number of training epochs.

        Returns:
            (Tuple[float, float]): Loss and accuracy from the training step.

        """

        batches = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        for epoch in range(epochs):
            logger.info("Epoch %d/%d", epoch + 1, epochs)

            start = time.time()

            loss = 0
            acc = 0

            for samples, labels in tqdm(batches):
                samples = samples.reshape(len(samples), self.n_visible)
                if self.device == "cuda":
                    samples = samples.cuda()
                    labels = labels.cuda()

                probs, _ = self.labels_sampling(samples)
                cost = self.loss(probs, labels)

                self.optimizer.zero_grad()
                cost.backward()
                self.optimizer.step()

                _, preds = self.labels_sampling(samples)

                batch_size = samples.size(0)
                batch_acc = torch.mean(
                    (torch.sum(preds == labels).float()) / batch_size
                )

                loss += cost.detach()
                acc += batch_acc

            loss /= len(batches)
            acc /= len(batches)

            end = time.time()

            self.dump(loss=loss.item(), acc=acc.item(), time=end - start)

            logger.info("Loss: %f | Accuracy: %f", loss, acc)

        return loss, acc

    def predict(
        self, dataset: torch.utils.data.Dataset
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """Predicts batches of new samples.

        Args:
            dataset: A Dataset object containing the testing data.

        Returns:
            (Tuple[float, torch.Tensor, torch.Tensor]): Accuracy, prediction probabilities and labels, i.e., P(y|v).

        """

        logger.info("Predicting new samples ...")

        acc = 0
        batch_size = len(dataset)

        batches = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        for samples, labels in tqdm(batches):
            samples = samples.reshape(len(samples), self.n_visible)
            if self.device == "cuda":
                samples = samples.cuda()
                labels = labels.cuda()

            probs, preds = self.labels_sampling(samples)

            batch_acc = torch.mean((torch.sum(preds == labels).float()) / batch_size)
            acc += batch_acc

        acc /= len(batches)

        logger.info("Accuracy: %f", acc)

        return acc, probs, preds


class HybridDiscriminativeRBM(DiscriminativeRBM):
    """A HybridDiscriminativeRBM class provides the basic implementation for
    Hybrid Discriminative Bernoulli-Bernoulli Restricted Boltzmann Machines.

    References:
        H. Larochelle and Y. Bengio. Classification using discriminative restricted Boltzmann machines.
        Proceedings of the 25th international conference on Machine learning (2008).

    """

    def __init__(
        self,
        n_visible: Optional[int] = 128,
        n_hidden: Optional[int] = 128,
        n_classes: Optional[int] = 1,
        steps: Optional[int] = 1,
        learning_rate: Optional[float] = 0.1,
        alpha: Optional[float] = 0.01,
        momentum: Optional[float] = 0.0,
        decay: Optional[float] = 0.0,
        temperature: Optional[float] = 1.0,
        use_gpu: Optional[bool] = False,
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

        logger.info("Overriding class: DiscriminativeRBM -> HybridDiscriminativeRBM.")

        # Override its parent class
        super(HybridDiscriminativeRBM, self).__init__(
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

    @property
    def alpha(self) -> float:
        """Generative loss penalization."""

        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if alpha < 0:
            raise e.ValueError("`alpha` should be >= 0")

        self._alpha = alpha

    def hidden_sampling(
        self, v: torch.Tensor, y: torch.Tensor, scale: Optional[bool] = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the hidden layer sampling, i.e., P(h|y,v).

        Args:
            v: A tensor incoming from the visible layer.
            y: A tensor incoming from the class layer.
            scale: A boolean to decide whether temperature should be used or not.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor]): The probabilities and states of the hidden layer sampling.

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
            (Tuple[torch.Tensor, torch.Tensor]): The probabilities and states of the class layer sampling.

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
            (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): The probabilities and states of the hidden layer sampling (positive),
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
        batch_size: Optional[int] = 128,
        epochs: Optional[int] = 10,
    ) -> Tuple[float, float]:
        """Fits a new DRBM model.

        Args:
            dataset: A Dataset object containing the training data.
            batch_size: Amount of samples per batch.
            epochs: Number of training epochs.

        Returns:
            (Tuple[float, float]): Loss and accuracy from the training step.

        """

        batches = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        for epoch in range(epochs):
            logger.info("Epoch %d/%d", epoch + 1, epochs)

            start = time.time()

            d_loss, g_loss, loss, acc = 0, 0, 0, 0

            for samples, labels in tqdm(batches):
                samples = samples.reshape(len(samples), self.n_visible)
                if self.device == "cuda":
                    samples = samples.cuda()
                    labels = labels.cuda()

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

                batch_size = samples.size(0)
                batch_acc = torch.mean(
                    (torch.sum(preds == labels).float()) / batch_size
                )

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

            logger.info(
                "Loss(D): %f | Loss(G): %f | Loss: %f | Accuracy: %f",
                d_loss,
                g_loss,
                loss,
                acc,
            )

        return loss, acc
