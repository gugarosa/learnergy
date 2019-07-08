import torch

import recogners.utils.logging as l
from recogners.models.rbm import RBM

logger = l.get_logger(__name__)


class DropoutRBM(RBM):
    """A DropoutRBM class provides the basic implementation for Restricted Boltzmann Machines
    along with a Dropout regularization.

    References:
        N. Srivastava, et al. Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research (2014).

    """

    def __init__(self, n_visible=128, n_hidden=128, steps=1, learning_rate=0.1, momentum=0, decay=0, temperature=1, dropout=0.5):
        """Initialization method.

        Args:
            n_visible (int): Amount of visible units.
            n_hidden (int): Amount of hidden units.
            steps (int): Number of Gibbs' sampling steps.
            learning_rate (float): Learning rate.
            momentum (float): Momentum parameter.
            decay (float): Weight decay used for penalization.
            temperature (float): Temperature factor.
            dropout (float): Dropout rate.

        """

        logger.info('Overriding class: RBM -> DropoutRBM.')

        # Override its parent class
        super(DropoutRBM, self).__init__(n_visible=n_visible, n_hidden=n_hidden, steps=steps,
                                         learning_rate=learning_rate, momentum=momentum,
                                         decay=decay, temperature=temperature)

        # Intensity of dropout
        self._p = dropout

        logger.info('Class overrided.')
        logger.debug(f'Additional hyperparameters: p = {self.p}')

    @property
    def p(self):
        """float: Intensity of dropout rate.

        """

        return self._p

    @p.setter
    def p(self, p):
        self._p = p

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

        # Sampling a dropout mask from Bernoulli's distribution
        mask = (torch.full((activations.size(0), activations.size(1)),
                           1 - self.p)).bernoulli()

        # If scaling is true
        if scale:
            # Calculate probabilities with temperature and dropout mask
            probs = torch.sigmoid(activations / self.T) * mask

        # If scaling is false
        else:
            # Calculate probabilities using a dropout mask
            probs = torch.sigmoid(activations) * mask

        # Sampling current states
        states = (probs > torch.rand(self.n_hidden)).double()

        return probs, states

    def reconstruct(self, batches):
        """Reconstruct batches of new samples.

        Args:
            batches (DataLoader): A DataLoader object containing batches to be reconstructed.

        Returns:
            Reconstruction error and visible probabilities, i.e., P(v|h).

        """

        logger.info(f'Reconstructing new samples ...')

        # Resetting error to zero
        error = 0

        # Scaling up weights matrix with dropout rate
        # self.W *= (1 - self.p)

        # Saving dropout rate to an auxiliary variable
        p = self.p

        # Temporarily disabling dropout
        self.p = 0

        # For every batch
        for samples, _ in batches:
            # Flattening the samples' batch
            samples = samples.view(len(samples), self.n_visible).double()

            # Calculating positive phase hidden probabilities and states
            pos_hidden_probs, pos_hidden_states = self.hidden_sampling(
                samples)

            # Calculating visible probabilities and states
            visible_probs, visible_states = self.visible_sampling(
                pos_hidden_states)

            # Gathering the size of the batch
            batch_size = samples.size(0)

            # Calculating current's batch reconstruction error
            batch_error = torch.sum(
                (samples - visible_states) ** 2) / batch_size

            # Summing up to reconstruction's error
            error += batch_error

        # Normalizing the error with the number of batches
        error /= len(batches)

        # Recovering initial dropout rate
        self.p = p

        logger.info(f'Error: {error}')

        return error, visible_probs
