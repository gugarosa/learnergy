import torch
import torch.nn.functional as F

import learnergy.utils.exception as e
import learnergy.utils.logging as l
from learnergy.models.rbm import RBM

logger = l.get_logger(__name__)


class GaussianRBM(RBM):
    """A GaussianRBM class provides the basic implementation for Gaussian-Gaussian Restricted Boltzmann Machines.

    References:
        

    """

    def __init__(self, n_visible=128, n_hidden=128, steps=1, learning_rate=0.1, momentum=0, decay=0, temperature=1, use_gpu=False):
        """Initialization method.

        Args:
            n_visible (int): Amount of visible units.
            n_hidden (int): Amount of hidden units.
            steps (int): Number of Gibbs' sampling steps.
            learning_rate (float): Learning rate.
            momentum (float): Momentum parameter.
            decay (float): Weight decay used for penalization.
            temperature (float): Temperature factor.
            use_gpu (boolean): Whether GPU should be used or not.

        """

        logger.info('Overriding class: RBM -> GaussianRBM.')

        # Override its parent class
        super(GaussianRBM, self).__init__(n_visible=n_visible, n_hidden=n_hidden, steps=steps,
                                         learning_rate=learning_rate, momentum=momentum, decay=decay, temperature=temperature, use_gpu=use_gpu)

        logger.info('Class overrided.')

    def visible_sampling(self, h, scale=False):
        """Performs the visible layer sampling, i.e., P(v|h).

        Args:
            h (tensor): A tensor incoming from the hidden layer.
            scale (bool): A boolean to decide whether temperature should be used or not.

        Returns:
            The states and probabilities of the visible layer sampling.

        """

        # Calculating neurons' activations
        activations = F.linear(h, self.W, self.a)

        # If scaling is true
        if scale:
            # Calculate probabilities with temperature
            states = activations / self.T

        # If scaling is false
        else:
            # Calculate probabilities as usual
            states = activations

        return None, states
