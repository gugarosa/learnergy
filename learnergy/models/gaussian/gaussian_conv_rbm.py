"""Gaussian-based Convolutional Restricted Boltzmann Machine.
"""

import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import learnergy.utils.constants as c
import learnergy.utils.exception as e
import learnergy.utils.logging as l
from learnergy.models.bernoulli import ConvRBM

logger = l.get_logger(__name__)


class GaussianConvRBM(ConvRBM):
    """A GaussianConvRBM class provides the basic implementation for
    Gaussian-based Convolutional Restricted Boltzmann Machines.

    References:
        H. Lee, et al.
        Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations.
        Proceedings of the 26th annual international conference on machine learning (2009).

    """

    def __init__(self, visible_shape=(28, 28), filter_shape=(7, 7), n_filters=5, n_channels=1,
                 steps=1, learning_rate=0.1, momentum=0, decay=0, use_gpu=False):
        """Initialization method.

        Args:
            visible_shape (tuple): Shape of visible units.
            filter_shape (tuple): Shape of filters.
            n_filters (int): Number of filters.
            n_channels (int): Number of channels.
            steps (int): Number of Gibbs' sampling steps.
            learning_rate (float): Learning rate.
            momentum (float): Momentum parameter.
            decay (float): Weight decay used for penalization.
            use_gpu (boolean): Whether GPU should be used or not.

        """

        # Inner data normalization
        self.normalize = True

        logger.info('Overriding class: ConvRBM -> GaussianConvRBM.')

        # Override its parent class
        super(GaussianConvRBM, self).__init__(visible_shape, filter_shape, n_filters, n_channels,
                                              steps, learning_rate, momentum, decay, use_gpu)

        logger.info('Class overrided.')

    @property
    def normalize(self):
        """bool: Inner data normalization.

        """

        return self._normalize

    @normalize.setter
    def normalize(self, normalize):
        if not isinstance(normalize, bool):
            raise e.TypeError('`normalize` should be a boolean')

        self._normalize = normalize

    def hidden_sampling(self, v):
        """Performs the hidden layer sampling, i.e., P(h|v).

        Args:
            v (torch.Tensor): A tensor incoming from the visible layer.

        Returns:
            The probabilities and states of the hidden layer sampling.

        """
        # Calculating neurons' activations
        activations = F.conv2d(v, self.W, bias=self.b)

        # Calculate probabilities
        probs = F.relu6(activations).detach()

        return probs, probs

    def visible_sampling(self, h):
        """Performs the visible layer sampling, i.e., P(v|h).

        Args:
            h (torch.Tensor): A tensor incoming from the hidden layer.

        Returns:
            The probabilities and states of the visible layer sampling.

        """

        # Calculating neurons' activations
        activations = F.conv_transpose2d(h, self.W, bias=self.a)

        # Checks it is supposed to perform the normalization
        if self.normalize:
            # Uses the previously calculated activations
            probs = activations.detach()

        # If it is not supposed to normalize
        else:
            # Applies a non-linear function
            probs = F.relu6(activations).detach()

        return probs, probs

    def fit(self, dataset, batch_size=128, epochs=10):
        """Fits a new RBM model.

        Args:
            dataset (torch.utils.data.Dataset): A Dataset object containing the training data.
            batch_size (int): Amount of samples per batch.
            epochs (int): Number of training epochs.

        Returns:
            MSE (mean squared error) from the training step.

        """

        # Transforming the dataset into training batches
        batches = DataLoader(dataset, batch_size=batch_size,
                             shuffle=True, num_workers=1)

        # For every epoch
        for epoch in range(epochs):
            logger.info('Epoch %d/%d', epoch+1, epochs)

            # Calculating the time of the epoch's starting
            start = time.time()

            # Resetting epoch's MSE to zero
            mse = 0

            # For every batch
            for samples, _ in tqdm(batches):
                # Guarantee the samples' batch
                samples = samples.reshape(len(samples), self.n_channels, self.visible_shape[0], self.visible_shape[1])

                # Checking whether GPU is avaliable and if it should be used
                if self.device == 'cuda':
                    # Applies the GPU usage to the data
                    samples = samples.cuda()

                # If it is supposed to use normalization
                if self.normalize:
                    # Performs the normalization
                    samples = ((samples - torch.mean(samples, 0, True)) /
                               (torch.std(samples, 0, True) + c.EPSILON))

                # Performs the Gibbs sampling procedure
                _, _, _, _, visible_states = self.gibbs_sampling(samples)

                # Detaching the visible states from GPU for further computation
                visible_states = visible_states.detach()

                # Calculates the loss for further gradients' computation
                cost = torch.mean(self.energy(samples)) - \
                    torch.mean(self.energy(visible_states))

                # Initializing the gradient
                self.optimizer.zero_grad()

                # Computing the gradients
                cost.backward()

                # Updating the parameters
                self.optimizer.step()

                # Calculating current's batch MSE
                batch_mse = torch.div(
                    torch.sum(torch.pow(samples - visible_states, 2)), batch_size).detach()

                # Summing up to epochs' MSE
                mse += batch_mse

            # Normalizing the MSE with the number of batches
            mse /= len(batches)

            # Calculating the time of the epoch's ending
            end = time.time()

            # Dumps the desired variables to the model's history
            self.dump(mse=mse.item(), time=end-start)

            logger.info('MSE: %f', mse)

        return mse
