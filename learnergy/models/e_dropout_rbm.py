import torch

import learnergy.utils.logging as l
from learnergy.models.rbm import RBM

logger = l.get_logger(__name__)


class EDropoutRBM(RBM):
    """An E-DropoutRBM class provides the basic implementation for Restricted Boltzmann Machines
    along with a energy-based Dropout regularization.

    References:
        Publication pending...

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

        # Hidden units importance level
        self._I = torch.zeros(1, n_hidden)

        logger.info('Class overrided.')

    @property
    def I(self):
        """tensor: Hidden units Importance level [1 x n_hidden].

        """

        return self._I

    @I.setter
    def I(self, I):
        self._I = I

    def hidden_sampling(self, v, scale=False):
        """Performs the hidden layer sampling using a dropout mask, i.e., P(h|r,v).

        Args:
            v (tensor): A tensor incoming from the visible layer.
            scale (bool): A boolean to decide whether temperature should be used or not.

        Returns:
            The probabilities and states of the hidden layer sampling.

        """

        # Calculating neurons' activations
        activations = torch.mm(v, self.W) + self.b

        # Calculating the Importance level
        self.I = (activations / torch.mean(self.energy(v)))

        # Re-scaling it with its maximum value
        self.I = self.I / torch.max(self.I)

        # Calculating its mean over the batches
        self.I = torch.mean(self.I, axis=0)

        # Sampling the e-dropout mask
        mask = (self.I < torch.rand(self.n_hidden)).double()

        # If scaling is true
        if scale:
            # Calculate probabilities with temperature
            probs = torch.sigmoid(activations / self.T) * mask

        # If scaling is false
        else:
            # Calculate probabilities
            probs = torch.sigmoid(activations) * mask

        # Sampling current states
        states = (probs > torch.rand(self.n_hidden)).double()

        return probs, states
