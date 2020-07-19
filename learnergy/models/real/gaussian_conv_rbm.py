"""Gaussian-based Convolutional Restricted Boltzmann Machine.
"""

import torch
import torch.nn.functional as F

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
        probs = F.relu6(activations)

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
        probs = torch.clamp(F.relu(activations), 0, 1)

        return probs, probs
