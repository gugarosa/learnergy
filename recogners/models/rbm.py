import torch

import recogners.utils.logging as l

logger = l.get_logger(__name__)


class RBM:
    """An RBM class provides the basic implementation for Restricted Boltzmann Machines.

    References:
        G. Hinton. A practical guide to training restricted Boltzmann machines. Neural networks: Tricks of the trade (2012).

    """

    def __init__(self, n_visible=128, n_hidden=128, learning_rate=0.1, steps=1, temperature=1):
        """Initialization method.

        Args:
            n_visible (int): Amount of visible units.
            n_hidden (int): Amount of hidden units.
            learning_rate (float): Learning rate.
            steps (int): Number of Gibbs' sampling steps.
            temperature (float): Temperature factor.

        """

        logger.info('Creating model: RBM.')

        # Amount of visible units
        self._n_visible = n_visible

        # Amount of hidden units
        self._n_hidden = n_hidden

        # Learning rate
        self._lr = learning_rate

        # Number of steps Gibbs' sampling steps
        self._steps = steps

        # Temperature factor
        self._T = temperature

        # Weights matrix
        self._W = torch.randn(n_visible, n_hidden)

        # Visible units bias
        self._a = torch.zeros(n_visible)

        # Hidden units bias
        self._b = torch.zeros(n_hidden)

        logger.info('Model created.')
        logger.debug(
            f'Size: ({self._n_visible}, {self._n_hidden}) | Hyperparameters: lr = {self._lr}, steps = {self._steps}, T = {self._T}.')

    @property
    def n_visible(self):
        """int: Amount of visible units.

        """

        return self._n_visible

    @property
    def n_hidden(self):
        """int: Amount of hidden units.

        """

        return self._n_hidden

    @property
    def lr(self):
        """float: The model's learning rate.

        """

        return self._lr

    @property
    def steps(self):
        """int: Number of Gibbs' sampling steps.

        """

        return self._steps

    @property
    def T(self):
        """float: Temperature factor.

        """

        return self._T

    @property
    def W(self):
        """tensor: Weights matrix [n_visible x n_hidden].

        """

        return self._W

    @W.setter
    def W(self, W):
        self._W = W

    @property
    def a(self):
        """tensor: Visible units bias [1 x n_visible].

        """

        return self._a

    @a.setter
    def a(self, a):
        self._a = a

    @property
    def b(self):
        """tensor: Hidden units bias [1 x n_hidden].

        """

        return self._b

    @b.setter
    def b(self, b):
        self._b = b

    def fit(self, batches, epochs=10):
        """
        """

        #
        for e in range(epochs):
            logger.info(f'Epoch {e+1}/{epochs}')
            #
            for samples, _ in batches:
                pass
