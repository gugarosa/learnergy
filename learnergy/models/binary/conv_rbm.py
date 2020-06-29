import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.data import DataLoader
from tqdm import tqdm

import learnergy.utils.constants as c
import learnergy.utils.exception as e
import learnergy.utils.logging as l
from learnergy.core import Model

logger = l.get_logger(__name__)


class ConvRBM(Model):
    """A ConvRBM class provides the basic implementation for Bernoulli-Bernoulli Restricted Boltzmann Machines.

    References:
        G. Hinton. A practical guide to training restricted Boltzmann machines.
        Neural networks: Tricks of the trade (2012).

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

        logger.info('Overriding class: Model -> ConvRBM.')

        # Override its parent class
        super(ConvRBM, self).__init__(use_gpu=use_gpu)

        # Shape of visible units
        self.visible_shape = visible_shape

        # Shape of filters
        self.filter_shape = filter_shape

        # Shape of hidden units
        self.hidden_shape = (visible_shape[0] - filter_shape[0] + 1, visible_shape[1] - filter_shape[1] + 1)

        # Number of filters
        self.n_filters = n_filters

        # Number of channels
        self.n_channels = n_channels

        # Number of steps Gibbs' sampling steps
        self.steps = steps

        # Learning rate
        self.lr = learning_rate

        # Momentum parameter
        self.momentum = momentum

        # Weight decay
        self.decay = decay

        # Filters' matrix
        self.W = nn.Parameter(torch.randn(n_filters, n_channels, filter_shape[0], filter_shape[1]) * 0.01)

        # Visible units bias
        self.a = nn.Parameter(torch.zeros(1))

        # Hidden units bias
        self.b = nn.Parameter(torch.zeros(n_filters))

        # Creating the optimizer object
        self.optimizer = opt.SGD(self.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)

        # Checks if current device is CUDA-based
        if self.device == 'cuda':
            # If yes, uses CUDA in the whole class
            self.cuda()

        logger.info('Class overrided.')
        logger.debug(
            f'Visible: {self.visible_shape} | Filters: {self.n_filters} x {self.filter_shape} | Hidden: {self.hidden_shape} | '
            f'Channels: {self.n_channels} | Learning: CD-{self.steps} | '
            f'Hyperparameters: lr = {self.lr}, momentum = {self.momentum}, decay = {self.decay}.')

    @property
    def visible_shape(self):
        """tuple: Shape of visible units.

        """

        return self._visible_shape

    @visible_shape.setter
    def visible_shape(self, visible_shape):
        if not isinstance(visible_shape, tuple):
            raise e.TypeError('`visible_shape` should be a tuple')

        self._visible_shape = visible_shape

    @property
    def filter_shape(self):
        """tuple: Shape of filters.

        """

        return self._filter_shape

    @filter_shape.setter
    def filter_shape(self, filter_shape):
        if not isinstance(filter_shape, tuple):
            raise e.TypeError('`filter_shape` should be a tuple')
        if (filter_shape[0] >= self.visible_shape[0]) or (filter_shape[1] >= self.visible_shape[1]):
            raise e.ValueError('`filter_shape` should be smaller than `visible_shape`')

        self._filter_shape = filter_shape

    @property
    def hidden_shape(self):
        """tuple: Shape of hidden units.

        """

        return self._hidden_shape

    @hidden_shape.setter
    def hidden_shape(self, hidden_shape):
        if not isinstance(hidden_shape, tuple):
            raise e.TypeError('`hidden_shape` should be a tuple')

        self._hidden_shape = hidden_shape

    @property
    def n_filters(self):
        """int: Number of filters.

        """

        return self._n_filters

    @n_filters.setter
    def n_filters(self, n_filters):
        if not isinstance(n_filters, int):
            raise e.TypeError('`n_filters` should be an integer')
        if n_filters <= 0:
            raise e.ValueError('`n_filters` should be > 0')

        self._n_filters = n_filters

    @property
    def n_channels(self):
        """int: Number of channels.

        """

        return self._n_channels

    @n_channels.setter
    def n_channels(self, n_channels):
        if not isinstance(n_channels, int):
            raise e.TypeError('`n_channels` should be an integer')
        if n_channels <= 0:
            raise e.ValueError('`n_channels` should be > 0')

        self._n_channels = n_channels

    @property
    def steps(self):
        """int: Number of steps Gibbs' sampling steps.

        """

        return self._steps

    @steps.setter
    def steps(self, steps):
        if not isinstance(steps, int):
            raise e.TypeError('`steps` should be an integer')
        if steps <= 0:
            raise e.ValueError('`steps` should be > 0')

        self._steps = steps

    @property
    def lr(self):
        """float: Learning rate.

        """

        return self._lr

    @lr.setter
    def lr(self, lr):
        if not (isinstance(lr, float) or isinstance(lr, int)):
            raise e.TypeError('`lr` should be a float or integer')
        if lr < 0:
            raise e.ValueError('`lr` should be >= 0')

        self._lr = lr

    @property
    def momentum(self):
        """float: Momentum parameter.

        """

        return self._momentum

    @momentum.setter
    def momentum(self, momentum):
        if not (isinstance(momentum, float) or isinstance(momentum, int)):
            raise e.TypeError('`momentum` should be a float or integer')
        if momentum < 0:
            raise e.ValueError('`momentum` should be >= 0')

        self._momentum = momentum

    @property
    def decay(self):
        """float: Weight decay.

        """

        return self._decay

    @decay.setter
    def decay(self, decay):
        if not (isinstance(decay, float) or isinstance(decay, int)):
            raise e.TypeError('`decay` should be a float or integer')
        if decay < 0:
            raise e.ValueError('`decay` should be >= 0')

        self._decay = decay

    @property
    def W(self):
        """torch.nn.Parameter: Filters' matrix.

        """

        return self._W

    @W.setter
    def W(self, W):
        if not isinstance(W, nn.Parameter):
            raise e.TypeError('`W` should be a PyTorch parameter')

        self._W = W

    @property
    def a(self):
        """torch.nn.Parameter: Visible units bias.

        """

        return self._a

    @a.setter
    def a(self, a):
        if not isinstance(a, nn.Parameter):
            raise e.TypeError('`a` should be a PyTorch parameter')

        self._a = a

    @property
    def b(self):
        """torch.nn.Parameter: Hidden units bias.

        """

        return self._b

    @b.setter
    def b(self, b):
        if not isinstance(b, nn.Parameter):
            raise e.TypeError('`b` should be a PyTorch parameter')

        self._b = b

    @property
    def optimizer(self):
        """torch.optim.SGD: Stochastic Gradient Descent object.

        """

        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        if not isinstance(optimizer, opt.SGD):
            raise e.TypeError('`optimizer` should be a SGD')

        self._optimizer = optimizer

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
        probs = torch.sigmoid(activations)

        # Sampling current states
        states = torch.bernoulli(probs)

        return probs, states

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
        probs = torch.sigmoid(activations)

        # Sampling current states
        states = torch.bernoulli(probs)

        return probs, states

    def gibbs_sampling(self, v):
        """Performs the whole Gibbs sampling procedure.

        Args:
            v (torch.Tensor): A tensor incoming from the visible layer.

        Returns:
            The probabilities and states of the hidden layer sampling (positive),
            the probabilities and states of the hidden layer sampling (negative)
            and the states of the visible layer sampling (negative). 

        """

        # Calculating positive phase hidden probabilities and states
        pos_hidden_probs, pos_hidden_states = self.hidden_sampling(v)

        # Initially defining the negative phase
        neg_hidden_states = pos_hidden_states

        # Performing the Contrastive Divergence
        for _ in range(self.steps):
            # Calculating visible probabilities and states
            visible_probs, visible_states = self.visible_sampling(neg_hidden_states)

            # Calculating hidden probabilities and states
            neg_hidden_probs, neg_hidden_states = self.hidden_sampling(visible_states)

        return pos_hidden_probs, pos_hidden_states, neg_hidden_probs, neg_hidden_states, visible_states

    def energy(self, samples):
        """Calculates and frees the system's energy.

        Args:
            samples (torch.Tensor): Samples to be energy-freed.

        Returns:
            The system's energy based on input samples.

        """

        # Calculate samples' activations
        activations = F.conv2d(samples, self.W, bias=self.b)

        # Creating a Softplus function for numerical stability
        s = nn.Softplus()

        # Calculate the hidden term
        h = torch.sum(s(activations), dim=(1, 2, 3))

        # Calculate the visible term
        v = torch.sum(samples, dim=(1, 2, 3)) * self.a

        # Finally, gathers the system's energy
        energy = -v - h

        return energy

    def fit(self, dataset, batch_size=128, epochs=10):
        """Fits a new RBM model.

        Args:
            dataset (torch.utils.data.Dataset): A Dataset object containing the training data.
            batch_size (int): Amount of samples per batch.
            epochs (int): Number of training epochs.

        Returns:
            MSE (mean squared error) and log pseudo-likelihood from the training step.

        """

        # Transforming the dataset into training batches
        batches = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

        # For every epoch
        for e in range(epochs):
            logger.info(f'Epoch {e+1}/{epochs}')

            # Calculating the time of the epoch's starting
            start = time.time()

            # Resetting epoch's MSE to zero
            mse = 0

            # For every batch
            for samples, _ in tqdm(batches):
                # Flattening the samples' batch
                samples = samples.reshape(len(samples), self.n_channels, self.visible_shape[0], self.visible_shape[1])

                # Checking whether GPU is avaliable and if it should be used
                if self.device == 'cuda':
                    # Applies the GPU usage to the data
                    samples = samples.cuda()

                # Performs the Gibbs sampling procedure
                _, _, _, _, visible_states = self.gibbs_sampling(samples)

                # Detaching the visible states from GPU for further computation
                visible_states = visible_states.detach()

                # Calculates the loss for further gradients' computation
                cost = torch.mean(self.energy(samples)) - torch.mean(self.energy(visible_states))

                # Initializing the gradient
                self.optimizer.zero_grad()

                # Computing the gradients
                cost.backward()

                # Updating the parameters
                self.optimizer.step()

                # Gathering the size of the batch
                batch_size = samples.size(0)

                # Calculating current's batch MSE
                batch_mse = torch.div(torch.sum(torch.pow(samples - visible_states, 2)), batch_size).detach()

                # Summing up to epochs' MSE
                mse += batch_mse

            # Normalizing the MSE with the number of batches
            mse /= len(batches)

            # Calculating the time of the epoch's ending
            end = time.time()

            # Dumps the desired variables to the model's history
            self.dump(mse=mse.item(), time=end-start)

            logger.info(f'MSE: {mse}')

        return mse

    def reconstruct(self, dataset):
        """Reconstructs batches of new samples.

        Args:
            dataset (torch.utils.data.Dataset): A Dataset object containing the testing data.

        Returns:
            Reconstruction error and visible probabilities, i.e., P(v|h).

        """

        logger.info(f'Reconstructing new samples ...')

        # Resetting MSE to zero
        mse = 0

        # Defining the batch size as the amount of samples in the dataset
        batch_size = len(dataset)

        # Transforming the dataset into training batches
        batches = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)

        # For every batch
        for samples, _ in tqdm(batches):
            # Flattening the samples' batch
            samples = samples.reshape(len(samples), self.n_channels, self.visible_shape[0], self.visible_shape[1])

            # Checking whether GPU is avaliable and if it should be used
            if self.device == 'cuda':
                # Applies the GPU usage to the data
                samples = samples.cuda()

            # Calculating positive phase hidden probabilities and states
            pos_hidden_probs, pos_hidden_states = self.hidden_sampling(samples)

            # Calculating visible probabilities and states
            visible_probs, visible_states = self.visible_sampling(pos_hidden_states)

            # Calculating current's batch reconstruction MSE
            batch_mse = torch.div(torch.sum(torch.pow(samples - visible_states, 2)), batch_size)

            # Summing up the reconstruction's MSE
            mse += batch_mse

        # Normalizing the MSE with the number of batches
        mse /= len(batches)

        logger.info(f'MSE: {mse}')

        return mse, visible_probs
