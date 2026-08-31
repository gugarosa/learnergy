"""Gaussian convolutional Restricted Boltzmann Machine."""

import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from learnergy.core.model import _validated_property
from learnergy.models.bernoulli.conv_rbm import ConvRBM


class GaussianConvRBM(ConvRBM):
    """Convolutional RBM with Gaussian visible units."""

    normalize = _validated_property("normalize")

    def __init__(
        self,
        visible_shape: tuple[int, int] = (28, 28),
        filter_shape: tuple[int, int] = (7, 7),
        n_filters: int = 5,
        n_channels: int = 1,
        steps: int = 1,
        learning_rate: float = 0.1,
        momentum: float = 0.0,
        decay: float = 0.0,
        maxpooling: bool = False,
        pooling_kernel: int = 2,
        use_gpu: bool = False,
        normalize: bool = True,
    ) -> None:
        """Initialize a Gaussian convolutional RBM."""

        super().__init__(
            visible_shape=visible_shape,
            filter_shape=filter_shape,
            n_filters=n_filters,
            n_channels=n_channels,
            steps=steps,
            learning_rate=learning_rate,
            momentum=momentum,
            decay=decay,
            maxpooling=maxpooling,
            pooling_kernel=pooling_kernel,
            use_gpu=use_gpu,
        )
        self.normalize = normalize

    def hidden_sampling(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute hidden probabilities and activations."""

        activations = F.conv2d(v, self.W, bias=self.b)
        return F.relu6(activations).detach(), activations

    def visible_sampling(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute visible probabilities and activations."""

        activations = F.conv_transpose2d(h, self.W, bias=self.a)
        probs = activations if self.normalize else torch.sigmoid(activations)
        return probs.detach(), activations

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 128,
        epochs: int = 10,
        log: bool = False,
    ) -> torch.Tensor:
        """Fit the model and return the final reconstruction error."""

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

                if self.normalize:
                    eps = torch.finfo(samples.dtype).eps
                    samples = (samples - samples.mean(0, True)) / (
                        samples.std(0, True) + eps
                    )

                _, _, _, _, visible_states = self.gibbs_sampling(samples)
                visible_states = visible_states.detach()

                cost = self.energy(samples).mean() - self.energy(visible_states).mean()
                self.optimizer.zero_grad()
                cost.backward()
                self.optimizer.step()

                mse += ((samples - visible_states) ** 2).sum() / samples.size(0)

            mse /= len(batches)
            self.dump(mse=mse.item(), time=time.time() - start)

        return mse

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return hidden activations, optionally pooled."""

        if self.normalize:
            eps = torch.finfo(x.dtype).eps
            x = (x - x.mean(0, True)) / (x.std(0, True) + eps)

        x, _ = self.hidden_sampling(x)
        if self.maxpooling:
            x = self.maxpol2d(x)

        return x


class GaussianConvRBM4Deep(GaussianConvRBM):
    """Gaussian convolutional RBM used in deeper ConvDBN layers."""

    def visible_sampling(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        activations = F.conv_transpose2d(h, self.W, bias=self.a)
        probs = (
            activations.detach() if self.normalize else F.relu6(activations).detach()
        )
        return probs, activations
