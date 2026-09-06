# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Discriminative Bernoulli-Bernoulli Restricted Boltzmann Machines.

The discriminative model computes one unnormalized class logit per sample and class. The hybrid model combines
that discriminative objective with a pseudo-likelihood generative objective and Contrastive Divergence sampling.

References:
    H. Larochelle and Y. Bengio. Classification using discriminative restricted Boltzmann machines.
    Proceedings of the 25th international conference on Machine learning (2008).

"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import learnergy.utils.exception as e
from learnergy.core.model import _validated_property
from learnergy.models.bernoulli.rbm import RBM


class DiscriminativeRBM(RBM):
    """Implement a discriminative Bernoulli-Bernoulli Restricted Boltzmann Machine."""

    n_classes = _validated_property(
        "n_classes",
        lambda _, value: value > 0,
        e.ValueError,
        "`n_classes` should be > 0.",
        doc="Number of target classes.",
    )
    U = _validated_property("U", doc="Class-to-hidden weight matrix.")
    c = _validated_property("c", doc="Class-bias vector.")
    loss = _validated_property("loss", doc="Cross-entropy classification loss.")

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
        """Initialize a discriminative Bernoulli-Bernoulli RBM.

        Args:
            n_visible: Number of visible units.
            n_hidden: Number of hidden units.
            n_classes: Number of target classes.
            steps: Number of Contrastive Divergence sampling steps.
            learning_rate: Learning rate used by stochastic gradient descent.
            momentum: Momentum used by stochastic gradient descent.
            decay: Weight decay used by stochastic gradient descent.
            temperature: Positive temperature applied during scaled sampling.
            use_gpu: Whether to use CUDA when it is available.

        Raises:
            ValueError: If a unit count, class count, step count, optimizer value, or temperature is invalid.

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

    def labels_sampling(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute class logits and predicted indices for visible samples.

        Args:
            samples: Visible tensor shaped ``(batch_size, n_visible)``.

        Returns:
            Logits shaped ``(batch_size, n_classes)`` and predicted indices shaped ``(batch_size,)``, in that order.

        Notes:
            Logits retain gradients and predicted indices are computed from a detached argmax.

        """

        logits = torch.zeros(samples.size(0), self.n_classes, device=self.device)
        activations = F.linear(samples, self.W.t(), self.b)

        for i in range(self.n_classes):
            logits[:, i] = self.c[i] + torch.sum(F.softplus(activations + self.U[i, :]), dim=1)

        preds = torch.argmax(logits.detach(), 1)

        return logits, preds

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 128,
        epochs: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update classifier parameters and metric history with shuffled mini-batches.

        Args:
            dataset: Dataset yielding visible samples and integer class labels.
            batch_size: Maximum number of samples per training batch.
            epochs: Number of complete training passes.

        Returns:
            Final-epoch detached scalar cross-entropy loss and mean batch accuracy tensors in that order.

        """

        batches = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        for _ in range(epochs):
            start = time.time()

            loss = 0
            acc = 0

            for samples, labels in batches:
                samples = samples.reshape(len(samples), self.n_visible).to(self.device)
                labels = labels.to(self.device)

                logits, _ = self.labels_sampling(samples)
                cost = self.loss(logits, labels)

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

    def predict(self, dataset: torch.utils.data.Dataset) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict labels for an entire dataset in one unshuffled batch.

        Args:
            dataset: Dataset yielding visible samples and integer class labels.

        Returns:
            Scalar accuracy, unnormalized class logits, and predicted class indices in dataset order.

        Notes:
            Logits have shape ``(len(dataset), n_classes)`` and retain gradients.
            Predicted indices have shape ``(len(dataset),)``.

        """

        acc = 0
        batch_size = len(dataset)

        batches = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        for samples, labels in batches:
            samples = samples.reshape(len(samples), self.n_visible).to(self.device)
            labels = labels.to(self.device)

            logits, preds = self.labels_sampling(samples)

            batch_acc = (preds == labels).float().mean()
            acc += batch_acc

        acc /= len(batches)

        return acc, logits, preds


class HybridDiscriminativeRBM(DiscriminativeRBM):
    """Implement a hybrid discriminative and generative Bernoulli-Bernoulli RBM."""

    alpha = _validated_property(
        "alpha",
        lambda _, value: value >= 0,
        e.ValueError,
        "`alpha` should be >= 0.",
        doc="Weight applied to the generative loss.",
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
        """Initialize a hybrid discriminative and generative RBM.

        Args:
            n_visible: Number of visible units.
            n_hidden: Number of hidden units.
            n_classes: Number of target classes.
            steps: Number of Contrastive Divergence sampling steps.
            learning_rate: Learning rate used by stochastic gradient descent.
            alpha: Nonnegative weight applied to the generative loss.
            momentum: Momentum used by stochastic gradient descent.
            decay: Weight decay used by stochastic gradient descent.
            temperature: Positive temperature applied during scaled sampling.
            use_gpu: Whether to use CUDA when it is available.

        Raises:
            ValueError: If a unit count, class count, step count, optimizer value, temperature, or alpha is invalid.

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample hidden units conditioned on visible units and class encodings.

        Args:
            v: Visible tensor shaped ``(batch_size, n_visible)``.
            y: One-hot class tensor shaped ``(batch_size, n_classes)``.
            scale: Whether to divide activations by the sampling temperature.

        Returns:
            Hidden probabilities and Bernoulli states, each shaped ``(batch_size, n_hidden)``, in that order.

        """

        activations = F.linear(v, self.W.t(), self.b) + torch.matmul(y, self.U)

        if scale:
            probs = torch.sigmoid(torch.div(activations, self.T))
        else:
            probs = torch.sigmoid(activations)

        states = torch.bernoulli(probs)

        return probs, states

    def class_sampling(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select class states conditioned on hidden units.

        Args:
            h: Hidden tensor shaped ``(batch_size, n_hidden)``.

        Returns:
            Class probabilities and one-hot argmax states shaped ``(batch_size, n_classes)``, in that order.

        """

        probs = F.softmax(F.linear(h, self.U, self.c), dim=1)
        states = torch.nn.functional.one_hot(torch.argmax(probs, dim=1), num_classes=self.n_classes).float()

        return probs, states

    def gibbs_sampling(
        self, v: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run Contrastive Divergence sampling from visible samples and class indices.

        Args:
            v: Visible tensor shaped ``(batch_size, n_visible)``.
            y: Integer class indices shaped ``(batch_size,)``.

        Returns:
            Positive hidden probabilities and states, negative hidden probabilities and states, and visible states.

        """

        y = torch.nn.functional.one_hot(y, num_classes=self.n_classes).float()

        pos_hidden_probs, pos_hidden_states = self.hidden_sampling(v, y)
        neg_hidden_states = pos_hidden_states

        for _ in range(self.steps):
            _, visible_states = self.visible_sampling(neg_hidden_states, True)
            _, class_states = self.class_sampling(neg_hidden_states)

            neg_hidden_probs, neg_hidden_states = self.hidden_sampling(visible_states, class_states, True)

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update hybrid-model parameters and metric history with shuffled mini-batches.

        Args:
            dataset: Dataset yielding visible samples and integer class labels.
            batch_size: Maximum number of samples per training batch.
            epochs: Number of complete training passes.

        Returns:
            Final-epoch detached scalar combined loss and mean batch accuracy tensors in that order.

        """

        batches = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        for _ in range(epochs):
            start = time.time()

            d_loss, g_loss, loss, acc = 0, 0, 0, 0

            for samples, labels in batches:
                samples = samples.reshape(len(samples), self.n_visible).to(self.device)
                labels = labels.to(self.device)

                _, _, _, _, visible_states = self.gibbs_sampling(samples, labels)
                visible_states = visible_states.detach()
                disc_logits, _ = self.labels_sampling(samples)

                d_cost = self.loss(disc_logits, labels)
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
