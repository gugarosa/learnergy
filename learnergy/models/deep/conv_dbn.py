"""Convolutional Deep Belief Network.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import learnergy.utils.exception as e
from learnergy.core import Dataset, Model
from learnergy.models.bernoulli import ConvRBM
from learnergy.models.gaussian import GaussianConvRBM, GaussianConvRBM4deep
from learnergy.utils import logging

logger = logging.get_logger(__name__)

MODELS = {"bernoulli": ConvRBM, "gaussian": GaussianConvRBM, "gaussiandeep": GaussianConvRBM4deep}


class ConvDBN(Model):
    """A ConvDBN class provides the basic implementation for Convolutional DBNs.

    References:
        H. Lee, et al.
        Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations.
        Proceedings of the 26th annual international conference on machine learning (2009).

    """

    def __init__(
        self,
        model: Optional[str] = "bernoulli",
        visible_shape: Optional[Tuple[int, int]] = (28, 28),
        filter_shape: Optional[Tuple[Tuple[int, int], ...]] = ((7, 7),),
        n_filters: Optional[Tuple[int, ...]] = (16,),
        n_channels: Optional[int] = 1,
        steps: Optional[Tuple[int, ...]] = (1,),
        learning_rate: Optional[Tuple[float, ...]] = (0.1,),
        momentum: Optional[Tuple[float, ...]] = (0.0,),
        decay: Optional[Tuple[float, ...]] = (0.0,),
        maxpooling: Optional[Tuple[bool,...]] = (False, False),
        pooling_kernel: Optional[Tuple[int, ...]] = (2, 2),
        use_gpu: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            model: Indicates which type of ConvRBM should be used to compose the DBN.
            visible_shape: Shape of visible units.
            filter_shape: Shape of filters per layer.
            n_filters: Number of filters per layer.
            n_channels: Number of channels.
            steps: Number of Gibbs' sampling steps per layer.
            learning_rate: Learning rate per layer.
            momentum: Momentum parameter per layer.
            decay: Weight decay used for penalization per layer.
            maxpooling: Whether MaxPooling2D should be used or not.
            pooling_kernel: The kernel size of each square-sized MaxPooling layer (when maxpooling=True).
            use_gpu: Whether GPU should be used or not.

        """

        logger.info("Overriding class: Model -> ConvDBN.")

        super(ConvDBN, self).__init__(use_gpu=use_gpu)

        self.visible_shape = visible_shape
        self.filter_shape = filter_shape

        self.n_filters = n_filters
        self.n_channels = n_channels
        self.n_layers = len(n_filters)

        self.steps = steps
        self.lr = learning_rate
        self.momentum = momentum
        self.decay = decay        

        self.maxpooling = maxpooling
        self.maxpol2d = []

        for i, mx in enumerate(maxpooling):
            if mx:
                self.maxpol2d.append(nn.MaxPool2d(kernel_size=pooling_kernel[i], stride=2, padding=1))
            else:
                self.maxpol2d.append(None)

        self.models = []
        for i in range(self.n_layers):
            
            if i > 0 and model=='gaussian':
                m = MODELS['gaussiandeep'](
                visible_shape,
                self.filter_shape[i],
                self.n_filters[i],
                n_channels,
                self.steps[i],
                self.lr[i],
                self.momentum[i],
                self.decay[i],
                self.maxpooling[i],
                pooling_kernel[i],
                use_gpu,
            )
            else:
                m = MODELS[model](
                visible_shape,
                self.filter_shape[i],
                self.n_filters[i],
                n_channels,
                self.steps[i],
                self.lr[i],
                self.momentum[i],
                self.decay[i],
                self.maxpooling[i],
                pooling_kernel[i],
                use_gpu,
            )

            visible_shape = (
                visible_shape[0] - self.filter_shape[i][0] + 1,
                visible_shape[1] - self.filter_shape[i][1] + 1,
            )
            if self.maxpooling[i]:
                # TODO: Needs to be adjusted to when pooling_kernel != 2 
                visible_shape = ((m.hidden_shape[0]//2) + 1, (m.hidden_shape[1]//2) + 1)
                

            n_channels = self.n_filters[i]

            self.models.append(m)

        if self.device == "cuda":
            self.cuda()

        logger.info("Class overrided.")

    @property
    def visible_shape(self) -> Tuple[int, int]:
        """Shape of visible units."""

        return self._visible_shape

    @visible_shape.setter
    def visible_shape(self, visible_shape: Tuple[int, int]) -> None:
        self._visible_shape = visible_shape

    @property
    def filter_shape(self) -> Tuple[Tuple[int, int], ...]:
        """Shape of filters."""

        return self._filter_shape

    @filter_shape.setter
    def filter_shape(self, filter_shape: Tuple[Tuple[int, int], ...]) -> None:
        self._filter_shape = filter_shape

    @property
    def n_filters(self) -> Tuple[int, ...]:
        """Number of filters."""

        return self._n_filters

    @n_filters.setter
    def n_filters(self, n_filters: Tuple[int, ...]) -> None:
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
    def n_layers(self) -> int:
        """Number of layers."""

        return self._n_layers

    @n_layers.setter
    def n_layers(self, n_layers: int) -> None:
        if n_layers <= 0:
            raise e.ValueError("`n_layers` should be > 0")

        self._n_layers = n_layers

    @property
    def steps(self) -> Tuple[int, ...]:
        """Number of steps Gibbs' sampling steps per layer."""

        return self._steps

    @steps.setter
    def steps(self, steps: Tuple[int, ...]) -> None:
        if len(steps) != self.n_layers:
            raise e.SizeError(f"`steps` should have size equal as {self.n_layers}")

        self._steps = steps

    @property
    def lr(self) -> Tuple[float, ...]:
        """Learning rate per layer."""

        return self._lr

    @lr.setter
    def lr(self, lr: Tuple[float, ...]) -> None:
        if len(lr) != self.n_layers:
            raise e.SizeError(f"`lr` should have size equal as {self.n_layers}")

        self._lr = lr

    @property
    def momentum(self) -> Tuple[float, ...]:
        """Momentum parameter per layer."""

        return self._momentum

    @momentum.setter
    def momentum(self, momentum: Tuple[float, ...]) -> None:
        if len(momentum) != self.n_layers:
            raise e.SizeError(f"`momentum` should have size equal as {self.n_layers}")

        self._momentum = momentum

    @property
    def decay(self) -> Tuple[float, ...]:
        """Weight decay per layer."""

        return self._decay

    @decay.setter
    def decay(self, decay: Tuple[float, ...]) -> None:
        if len(decay) != self.n_layers:
            raise e.SizeError(f"`decay` should have size equal as {self.n_layers}")

        self._decay = decay

    @property
    def maxpooling(self) -> Tuple[bool, ...]:
        """Usage of MaxPooling."""

        return self._maxpooling

    @maxpooling.setter
    def maxpooling(self, maxpooling: Tuple[bool, ...]) -> None:
        if len(maxpooling) != self.n_layers:
            raise e.ValueError("`maxpooling` should be a Tuple of True or False")

        self._maxpooling = maxpooling

    @property
    def models(self) -> List[torch.nn.Module]:
        """List of models (RBMs)."""

        return self._models

    @models.setter
    def models(self, models: List[torch.nn.Module]) -> None:
        self._models = models

    def fit(
        self,
        dataset: Union[torch.utils.data.Dataset, Dataset],
        batch_size: Optional[int] = 128,
        epochs: Optional[Tuple[int, ...]] = (10, 10),
    ) -> float:
        """Fits a new ConvDBN model.

        Args:
            dataset: A Dataset object containing the training data.
            batch_size: Amount of samples per batch.
            epochs: Number of training epochs per layer.

        Returns:
            (float): MSE (mean squared error) from the training step.

        """

        if len(epochs) != self.n_layers:
            raise e.SizeError(("`epochs` should have size equal as %d", self.n_layers))

        mse = []

        try: 
            samples, targets, transform = (
                    dataset.data.numpy(),
                    dataset.targets.numpy(),
                    dataset.transform,
            )
            d = Dataset(samples, targets, transform)
        except:
            try:
                samples, targets, transform = (
                        dataset.data,
                        dataset.targets,
                        dataset.transform,
                )
                d = Dataset(samples, targets, transform)
            except:
                d = dataset
        batches = DataLoader(d, batch_size=batch_size, shuffle=True)

        for i, model in enumerate(self.models):
            logger.info("Fitting layer %d/%d ...", i + 1, self.n_layers)

            if i ==0:
                model_mse = model.fit(d, batch_size, epochs[i])
                mse.append(model_mse)
            else:                
                # creating the training phase for deeper models
                for ep in range(epochs[i]):
                    logger.info("Epoch %d/%d", ep + 1, epochs[i])
                    model_mse = 0
                    for step, (samples, y) in enumerate(batches):

                        if self.device == "cuda":
                            samples = samples.cuda()

                        for ii in range(i):
                            samples, _ = self.models[ii].hidden_sampling(samples)
                            if self.maxpooling[ii]:
                                samples = self.maxpol2d[ii](samples)

                        # Creating the dataset to ''mini-fit'' the i-th model                        
                        ds = Dataset(samples, y, None, show_log=False)
                        # Fiting the model with the batch
                        model_mse += model.fit(ds, samples.size(0), 1)

                    model_mse/=len(batches)
                    logger.info("MSE: %f", model_mse)
                mse.append(model_mse)                

        return mse

    def reconstruct(
        self, dataset: torch.utils.data.Dataset
    ) -> Tuple[float, torch.Tensor]:
        """Reconstructs batches of new samples.

        Args:
            dataset (torch.utils.data.Dataset): A Dataset object containing the training data.

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

            hidden_probs = samples
            for model in self.models:
                hidden_probs, _ = model.hidden_sampling(hidden_probs)

            visible_probs = hidden_probs
            for model in reversed(self.models):
                visible_probs, visible_states = model.visible_sampling(visible_probs)

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
            (torch.Tensor): A tensor containing the ConvDBN's outputs.

        """

        i = 0
        for model in self.models:
            x, _ = model.hidden_sampling(x)
            if self.maxpooling[i]:
                x = self.maxpol2d[i](x)
            i+=1

        return x
