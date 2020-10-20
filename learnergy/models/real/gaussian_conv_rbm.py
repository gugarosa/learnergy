"""Gaussian-based Convolutional Restricted Boltzmann Machine.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import learnergy.utils.logging as l
from learnergy.models.binary import ConvRBM

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

        logger.info('Overriding class: ConvRBM -> GaussianConvRBM.')

        # Override its parent class
        super(GaussianConvRBM, self).__init__(visible_shape, filter_shape, n_filters, n_channels,
                                              steps, learning_rate, momentum, decay, use_gpu)

        logger.info('Class overrided.')

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

        # Calculate probabilities
        probs = torch.clamp(F.relu(activations), 0, 1).detach()

        return probs, probs

    def propagate(self, dataset):
        """Reconstructs batches of new samples.

        Args:
            dataset (torch.utils.data.Dataset): A Dataset object containing the testing data.

        Returns:
            Reconstruction error and visible probabilities, i.e., P(v|h).

        """

        logger.info('Propagating through hidden map ...')


        # Defining the new dataset for convolutions
        ds = torch.ones((len(dataset), self.n_filters, self.hidden_shape[0], self.hidden_shape[1]), dtype=torch.float, device=self.device)

        batch_size = 200

        # Transforming the dataset into training batches
        batches = DataLoader(dataset, batch_size=batch_size,
                             shuffle=False, drop_last=False, num_workers=0)

        # For every batch
        j = 0
        for samples, _ in tqdm(batches):
            # Flattening the samples' batch
            samples = samples.reshape(
                len(samples), self.n_channels, self.visible_shape[0], self.visible_shape[1])

            # Checking whether GPU is avaliable and if it should be used
            if self.device == 'cuda':
                # Applies the GPU usage to the data
                samples = samples.cuda()

            # Calculating positive phase hidden probabilities and states
            _, ds[j:(j + batch_size), :, :, :] = self.hidden_sampling(samples)
            j += batch_size

        return ds.detach()