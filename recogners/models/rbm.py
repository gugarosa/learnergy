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

    def hidden_sampling(self, v):
        """Performs the hidden layer sampling, i.e., P(h|v).

        Args:
            v (tensor): A tensor incoming from the visible layer.

        Returns:
            The probabilities and states of the hidden layer sampling.

        """

        # Calculating neurons' activations
        activations = torch.mm(v, self.W) + self.b

        # Transforming into probabilites
        probs = torch.sigmoid(activations)

        # Sampling current states
        states = (probs > torch.rand(self.n_hidden)).float()

        return probs, states

    def visible_sampling(self, h):
        """Performs the visible layer sampling, i.e., P(v|h).

        Args:
            h (tensor): A tensor incoming from the hidden layer.

        Returns:
            The probabilities and states of the visible layer sampling.

        """

        # Calculating neurons' activations
        activations = torch.mm(h, self.W.t()) + self.a

        # Transforming into probabilites
        probs = torch.sigmoid(activations)

        # Sampling current states
        states = (probs > torch.rand(self.n_visible)).float()

        return probs, states

    def fit(self, batches, epochs=10):
        """Fits a new RBM model.

        Args:
            batches (DataLoader): A DataLoader object containing the training batches.
            epochs (int): Number of training epochs.

        """

        # For every epoch
        for e in range(epochs):
            logger.info(f'Epoch {e+1}/{epochs}')

            # Resetting epoch's error to zero
            error = 0

            # For every batch
            for i, (samples, _) in enumerate(batches):
                # Flattening the samples' batch
                samples = samples.view(len(samples), self.n_visible)

                # Calculating positive phase hidden probabilities and states
                pos_hidden_probs, pos_hidden_states = self.hidden_sampling(
                    samples)

                # Calculating visible probabilities and states
                visible_probs, visible_states = self.visible_sampling(
                    pos_hidden_states)

                # Performing the Contrastive Divergence
                for _ in range(self.steps):
                    # Calculating negative phase hidden probabilities and states
                    neg_hidden_probs, neg_hidden_states = self.hidden_sampling(
                        visible_states)

                    # Calculating visible probabilities and states
                    visible_probs, visible_states = self.visible_sampling(
                        neg_hidden_states)

                # Building the positive gradient
                pos_gradient = torch.mm(samples.t(), pos_hidden_probs)

                # Building the negative gradient
                neg_gradient = torch.mm(visible_probs.t(), neg_hidden_probs)

                # Gathering the size of the batch
                batch_size = samples.size(0)

                # Updating weights matrix
                self.W += self.lr * (pos_gradient - neg_gradient) / batch_size

                # Updating visible units biases
                self.a += self.lr * \
                    torch.sum((samples - visible_probs), dim=0) / batch_size

                # Updating hidden units biases
                self.b += self.lr * \
                    torch.sum((pos_hidden_probs - neg_hidden_probs),
                              dim=0) / batch_size

                # Calculating current's batch error
                batch_error = torch.sum(
                    (samples - visible_states) ** 2) / batch_size

                # Summing up to epochs' error
                error += batch_error

            # Normalizing the error with the number of batches
            error /= i

            logger.info(f'Reconstruction error: {error}')
