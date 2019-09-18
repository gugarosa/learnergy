import torch

import recogners.utils.logging as l
from recogners.models.rbm import RBM

logger = l.get_logger(__name__)


class EDropoutRBM(RBM):
    """An E-DropoutRBM class provides the basic implementation for Restricted Boltzmann Machines
    along with a energy-based Dropout regularization.

    References:
        Still to publish ...

    """

    def __init__(self, n_visible=128, n_hidden=128, steps=1, learning_rate=0.1, momentum=0, decay=0, temperature=1):
        """Initialization method.

        Args:
            n_visible (int): Amount of visible units.
            n_hidden (int): Amount of hidden units.
            steps (int): Number of Gibbs' sampling steps.
            learning_rate (float): Learning rate.
            momentum (float): Momentum parameter.
            decay (float): Weight decay used for penalization.
            temperature (float): Temperature factor.

        """

        logger.info('Overriding class: RBM -> EDropoutRBM.')

        # Override its parent class
        super(EDropoutRBM, self).__init__(n_visible=n_visible, n_hidden=n_hidden, steps=steps,
                                         learning_rate=learning_rate, momentum=momentum,
                                         decay=decay, temperature=temperature)

        logger.info('Class overrided.')
