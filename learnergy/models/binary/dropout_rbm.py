import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import learnergy.utils.exception as e
import learnergy.utils.logging as l
from learnergy.models.binary import RBM

logger = l.get_logger(__name__)


class DropoutRBM(RBM):
    """A DropoutRBM class provides the basic implementation for Bernoulli-Bernoulli Restricted Boltzmann Machines
    along with a Dropout regularization.

    References:
        N. Srivastava, et al. Dropout: a simple way to prevent neural networks from overfitting.
        The journal of machine learning research (2014).

    """

    def __init__(self, n_visible=128, n_hidden=128, steps=1, learning_rate=0.1,
                 momentum=0, decay=0, temperature=1, dropout=0.5, use_gpu=False):
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
            use_gpu (boolean): Whether GPU should be used or not.

        """

        logger.info('Overriding class: RBM -> DropoutRBM.')

        # Override its parent class
        super(DropoutRBM, self).__init__(n_visible, n_hidden, steps, learning_rate,
                                         momentum, decay, temperature, use_gpu)

        # Intensity of dropout
        self.p = dropout

        logger.info('Class overrided.')
        logger.debug(f'Additional hyperparameters: p = {self.p}.')

    @property
    def p(self):
        """float: Probability of applying dropout.

        """

        return self._p

    @p.setter
    def p(self, p):
        if not (isinstance(p, float) or isinstance(p, int)):
            raise e.TypeError('`p` should be a float or integer')
        if p < 0 or p > 1:
            raise e.ValueError('`p` should be between 0 and 1')

        self._p = p

    def hidden_sampling(self, v, scale=False):
        """Performs the hidden layer sampling using a dropout mask, i.e., P(h|r,v).

        Args:
            v (torch.Tensor): A tensor incoming from the visible layer.
            scale (bool): A boolean to decide whether temperature should be used or not.

        Returns:
            The probabilities and states of the hidden layer sampling.

        """

        # Calculating neurons' activations
        activations = F.linear(v, self.W.t(), self.b)

        # Sampling a dropout mask from Bernoulli's distribution
        mask = (torch.full((activations.size(0), activations.size(1)),
                           1 - self.p, dtype=torch.float, device=self.device)).bernoulli()

        # If scaling is true
        if scale:
            # Calculate probabilities with temperature
            probs = torch.mul(torch.sigmoid(
                torch.div(activations, self.T)), mask)

        # If scaling is false
        else:
            # Calculate probabilities as usual
            probs = torch.mul(torch.sigmoid(activations), mask)

        # Sampling current states
        states = torch.bernoulli(probs)

        return probs, states

    def reconstruct(self, dataset):
        """Reconstructs batches of new samples.

        Args:
            dataset (torch.utils.data.Dataset): A Dataset object containing the training data.

        Returns:
            Reconstruction error and visible probabilities, i.e., P(v|h).

        """

        logger.info(f'Reconstructing new samples ...')

        # Resetting mse to zero
        mse = 0

        # Defining the batch size as the amount of samples in the dataset
        batch_size = len(dataset)

        self.W = nn.Parameter(self.W * self.p)

        # Saving dropout rate to an auxiliary variable
        p = self.p

        # Temporarily disabling dropout
        self.p = 0

        # Transforming the dataset into testing batches
        batches = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)

        # For every batch
        for samples, _ in tqdm(batches):
            # Flattening the samples' batch
            samples = samples.reshape(len(samples), self.n_visible)

            # Checking whether GPU is avaliable and if it should be used
            if self.device == 'cuda':
                # Applies the GPU usage to the data
                samples = samples.cuda()

            # Calculating positive phase hidden probabilities and states
            pos_hidden_probs, pos_hidden_states = self.hidden_sampling(samples)

            # Calculating visible probabilities and states
            visible_probs, visible_states = self.visible_sampling(
                pos_hidden_states)

            # Calculating current's batch reconstruction MSE
            batch_mse = torch.div(
                torch.sum(torch.pow(samples - visible_states, 2)), batch_size)

            # Summing up the reconstruction's MSE
            mse += batch_mse

        # Normalizing the MSE with the number of batches
        mse /= len(batches)

        # Recovering initial dropout rate
        self.p = p

        logger.info(f'MSE: {mse}')

        return mse, visible_probs
