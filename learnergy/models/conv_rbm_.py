import torch
import torch.nn as nn
import torch.nn.functional as F

import learnergy.utils.exception as e
import learnergy.utils.logging as l
from learnergy.models import RBM

logger = l.get_logger(__name__)


class ConvRBM(RBM):
    """A ConvRBM class provides the basic implementation for Convolutional Restricted Boltzmann Machines.

    References:
        H. Lee, et al. Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations.
        Proceedings of the 26th annual international conference on machine learning (2009).

    """

    def __init__(self, n_visible=128, n_hidden=128, steps=1, learning_rate=0.1,
                 momentum=0, decay=0, temperature=1, use_gpu=False):
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

        logger.info('Overriding class: RBM -> ConvRBM.')

        # Override its parent class
        super(ConvRBM, self).__init__(n_visible, n_hidden, steps, learning_rate,
                                          momentum, decay, temperature, use_gpu)

        # Updating optimizer's parameters with `sigma`
        self.optimizer.add_param_group({'params': self.sigma})

        # Re-checks if current device is CUDA-based due to new parameter
        if self.device == 'cuda':
            # If yes, re-uses CUDA in the whole class
            self.cuda()

        logger.info('Class overrided.')
