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

        # Setting default tensor type to Double
        torch.set_default_tensor_type(torch.FloatTensor)

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
        self._W = torch.randn(n_visible, n_hidden) * 0.01

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

    def hidden_sampling(self, v):
        """
        """
        
        #
        activations = torch.mm(v, self.W) + self.b

        #
        probs = torch.sigmoid(activations)

        return probs

    def visible_sampling(self, h):
        """
        """
        
        #
        activations = torch.mm(h, self.W.t()) + self.a

        #
        probs = torch.sigmoid(activations)

        return probs



    def fit(self, batches, epochs=10):
        """
        """

        # For every epoch
        for e in range(epochs):
            logger.info(f'Epoch {e+1}/{epochs}')
            
            #
            error = 0

            # For every batch
            for i, (samples, _) in enumerate(batches):
                #
                samples = samples.view(len(samples), self.n_visible)

                # Calculating positive phase hidden probabilities
                pos_hidden_probs = self.hidden_sampling(samples)

                #
                pos_hidden_states = (pos_hidden_probs > torch.rand(self.n_hidden)).float()

                #
                pos_gradient = torch.mm(samples.t(), pos_hidden_probs)

                #
                visible_probs = self.visible_sampling(pos_hidden_states)

                #
                visible_states = (visible_probs > torch.rand(self.n_visible)).float()

                for _ in range(self.steps):
                    #
                    hidden_probs = self.hidden_sampling(visible_states)

                    #
                    hidden_states = (hidden_probs > torch.rand(self.n_hidden)).float()

                    #
                    visible_probs = self.visible_sampling(hidden_states)

                    #
                    visible_states = (visible_probs > torch.rand(self.n_visible)).float()

                #
                neg_gradient = torch.mm(visible_probs.t(), hidden_probs)

                #
                batch_size = samples.size(0)

                #
                self.W += self.lr * (pos_gradient - neg_gradient) / batch_size

                #
                self.a += self.lr * torch.sum((samples - visible_probs), dim=0) / batch_size

                #
                self.b += self.lr * torch.sum((pos_hidden_probs - hidden_probs), dim=0) / batch_size

                #
                batch_error = torch.sum((samples - visible_states) ** 2) / batch_size

                #
                error += batch_error

            #
            error /= i

            logger.info(f'Reconstruction error: {error}')


