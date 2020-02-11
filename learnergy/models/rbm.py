import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

import learnergy.utils.exception as e
import learnergy.utils.logging as l
from learnergy.core.model import Model

logger = l.get_logger(__name__)


class RBM(Model):
    """An RBM class provides the basic implementation for Bernoulli-Bernoulli Restricted Boltzmann Machines.

    References:
        G. Hinton. A practical guide to training restricted Boltzmann machines. Neural networks: Tricks of the trade (2012).

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

        logger.info('Overriding class: Model -> RBM.')

        # Override its parent class
        super(RBM, self).__init__(use_gpu=use_gpu)

        # Amount of visible units
        self.n_visible = n_visible

        # Amount of hidden units
        self.n_hidden = n_hidden

        # Number of steps Gibbs' sampling steps
        self.steps = steps

        # Learning rate
        self.lr = learning_rate

        # Momentum parameter
        self.momentum = momentum

        # Weight decay
        self.decay = decay

        # Temperature factor
        self.T = temperature

        # Weights matrix
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)

        # Visible units bias
        self.a = nn.Parameter(torch.zeros(n_visible))

        # Hidden units bias
        self.b = nn.Parameter(torch.zeros(n_hidden))

        # Creating the optimizer object
        self.optimizer = opt.SGD(
            self.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)

        # Checks if current device is CUDA-based
        if self.device == 'cuda':
            # If yes, uses CUDA in the whole class
            self.cuda()

        logger.info('Class overrided.')
        logger.debug(
            f'Size: ({self.n_visible}, {self.n_hidden}) | Learning: CD-{self.steps} | Hyperparameters: lr = {self.lr}, momentum = {self.momentum}, decay = {self.decay}, T = {self.T}.')

    @property
    def n_visible(self):
        """int: Number of visible units.

        """

        return self._n_visible

    @n_visible.setter
    def n_visible(self, n_visible):
        if not isinstance(n_visible, int):
            raise e.TypeError('`n_visible` should be an integer')
        if n_visible <= 0:
            raise e.ValueError('`n_visible` should be > 0')

        self._n_visible = n_visible

    @property
    def n_hidden(self):
        """int: Number of hidden units.

        """

        return self._n_hidden

    @n_hidden.setter
    def n_hidden(self, n_hidden):
        if not isinstance(n_hidden, int):
            raise e.TypeError('`n_hidden` should be an integer')
        if n_hidden <= 0:
            raise e.ValueError('`n_hidden` should be > 0')

        self._n_hidden = n_hidden

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
    def T(self):
        """float: Temperature factor.

        """

        return self._T

    @T.setter
    def T(self, T):
        if not (isinstance(T, float) or isinstance(T, int)):
            raise e.TypeError('`T` should be a float or integer')
        if T < 0 or T > 1:
            raise e.ValueError('`T` should be between 0 and 1')

        self._T = T

    @property
    def W(self):
        """torch.nn.Parameter: Weights' matrix.

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

    def hidden_sampling(self, v, scale=False):
        """Performs the hidden layer sampling, i.e., P(h|v).

        Args:
            v (tensor): A tensor incoming from the visible layer.
            scale (bool): A boolean to decide whether temperature should be used or not.

        Returns:
            The probabilities and states of the hidden layer sampling.

        """

        # Calculating neurons' activations
        activations = F.linear(v, self.W.t(), self.b)

        # If scaling is true
        if scale:
            # Calculate probabilities with temperature
            probs = torch.sigmoid(activations / self.T)

        # If scaling is false
        else:
            # Calculate probabilities as usual
            probs = torch.sigmoid(activations)

        # Sampling current states
        states = torch.bernoulli(probs)

        return probs, states

    def visible_sampling(self, h, scale=False):
        """Performs the visible layer sampling, i.e., P(v|h).

        Args:
            h (tensor): A tensor incoming from the hidden layer.
            scale (bool): A boolean to decide whether temperature should be used or not.

        Returns:
            The probabilities and states of the visible layer sampling.

        """

        # Calculating neurons' activations
        activations = F.linear(h, self.W, self.a)

        # If scaling is true
        if scale:
            # Calculate probabilities with temperature
            probs = torch.sigmoid(activations / self.T)

        # If scaling is false
        else:
            # Calculate probabilities as usual
            probs = torch.sigmoid(activations)

        # Sampling current states
        states = torch.bernoulli(probs)

        return probs, states

    def gibbs_sampling(self, v):
        """Performs the whole Gibbs sampling procedure.

        Args:
            v (tensor): A tensor incoming from the visible layer.

        Returns:
            The probabilities and states of the hidden layer sampling (positive),
            the probabilities and states of the hidden layer sampling (negative)
            and the states of the visible layer sampling (negative). 

        """

        # Calculating positive phase hidden probabilities and states
        pos_hidden_probs, pos_hidden_states = self.hidden_sampling(v)

        # Calculating visible probabilities and states
        visible_probs, visible_states = self.visible_sampling(
            pos_hidden_states)

        # Performing the Contrastive Divergence
        for _ in range(self.steps):
            # Calculating negative phase hidden probabilities and states
            neg_hidden_probs, neg_hidden_states = self.hidden_sampling(
                visible_states, scale=True)

            # Calculating visible probabilities and states
            visible_probs, visible_states = self.visible_sampling(
                neg_hidden_states, scale=True)

        return pos_hidden_probs, pos_hidden_states, neg_hidden_probs, neg_hidden_states, visible_states

    def energy(self, samples):
        """Calculates and frees the system's energy.

        Args:
            samples (tensor): Samples to be energy-freed.

        Returns:
            The system's energy based on input samples.

        """

        # Calculate samples' activations
        activations = F.linear(samples, self.W.t(), self.b)

        # Calculate the hidden term
        h = torch.sum(torch.log(1 + torch.exp(activations)), dim=1)

        # Calculate the visible term
        v = torch.mv(samples, self.a)

        # Finally, gathers the system's energy
        energy = -v - h

        return energy

    def pseudo_likelihood(self, samples):
        """Calculates the logarithm of the pseudo-likelihood.

        Args:
            samples (tensor): Samples to be calculated.

        Returns:
            The logarithm of the pseudo-likelihood based on input samples.

        """

        # Gathering a new array to hold the rounded samples
        samples_binary = torch.round(samples)

        # Calculates the energy of samples before flipping the bits
        e = self.energy(samples_binary)

        # Samples an array of indexes to flip the bits
        indexes = torch.randint(0, self.n_visible, size=(
            samples.size(0), 1), device=self.device)

        # Creates an empty vector for filling the indexes
        bits = torch.zeros(samples.size(
            0), samples.size(1), device=self.device)

        # Fills the sampled indexes with 1
        bits = bits.scatter_(1, indexes, 1)

        # Actually flips the bits
        samples_binary = torch.where(
            bits == 0, samples_binary, 1 - samples_binary)

        # Calculates the energy after flipping the bits
        e1 = self.energy(samples_binary)

        # Calculate the logarithm of the pseudo-likelihood
        pl = torch.mean(self.n_visible * torch.log(torch.sigmoid(e1 - e)))

        return pl

    def fit(self, batches, epochs=10):
        """Fits a new RBM model.

        Args:
            batches (DataLoader): A DataLoader object containing the training batches.
            epochs (int): Number of training epochs.

        Returns:
            MSE (mean squared error), log pseudo-likelihood and time from the training step.

        """

        # For every epoch
        for e in range(epochs):
            logger.info(f'Epoch {e+1}/{epochs}')

            # Calculating the time of the epoch's starting
            start = time.time()

            # Resetting epoch's MSE and pseudo-likelihood to zero
            mse = 0
            pl = 0

            # For every batch
            for samples, _ in batches:
                # Flattening the samples' batch
                samples = samples.view(len(samples), self.n_visible).float()

                # Checking whether GPU is avaliable and if it should be used
                if self.device == 'cuda':
                    # Applies the GPU usage to the data
                    samples = samples.cuda()

                # Performs the Gibbs sampling procedure
                _, _, _, _, visible_states = self.gibbs_sampling(samples)

                # Detaching the visible states from GPU for further computation
                visible_states = visible_states.detach()

                # Calculates the loss for further gradients' computation
                cost = torch.mean(self.energy(samples)) - \
                    torch.mean(self.energy(visible_states))

                # Initializing the gradient
                self.optimizer.zero_grad()

                # Computing the gradients
                cost.backward()

                # Updating the parameters
                self.optimizer.step()

                # Gathering the size of the batch
                batch_size = samples.size(0)

                # Calculating current's batch MSE
                batch_mse = torch.sum(
                    (samples - visible_states) ** 2) / batch_size

                # Calculating the current's batch logarithm pseudo-likelihood
                batch_pl = self.pseudo_likelihood(samples)

                # Summing up to epochs' MSE and pseudo-likelihood
                mse += batch_mse
                pl += batch_pl

            # Normalizing the MSE and pseudo-likelihood with the number of batches
            mse /= len(batches)
            pl /= len(batches)

            # Calculating the time of the epoch's ending
            end = time.time()

            # Dumps the desired variables to the model's history
            self.dump(mse=mse.item(), pl=pl.item(), time=end-start)

            logger.info(f'MSE: {mse} | log-PL: {pl}')

        return mse, pl

    def reconstruct(self, batches):
        """Reconstruct batches of new samples.

        Args:
            batches (DataLoader): A DataLoader object containing batches to be reconstructed.

        Returns:
            Reconstruction error and visible probabilities, i.e., P(v|h).

        """

        logger.info(f'Reconstructing new samples ...')

        # Resetting MSE to zero
        mse = 0

        # For every batch
        for samples, _ in batches:
            # Flattening the samples' batch
            samples = samples.view(len(samples), self.n_visible).float()

            # Checking whether GPU is avaliable and if it should be used
            if self.device == 'cuda':
                # Applies the GPU usage to the data
                samples = samples.cuda()

            # Calculating positive phase hidden probabilities and states
            pos_hidden_probs, pos_hidden_states = self.hidden_sampling(samples)

            # Calculating visible probabilities and states
            visible_probs, visible_states = self.visible_sampling(
                pos_hidden_states)

            # Gathering the size of the batch
            batch_size = samples.size(0)

            # Calculating current's batch reconstruction MSE
            batch_mse = torch.sum((samples - visible_states) ** 2) / batch_size

            # Summing up to reconstruction's MSE
            mse += batch_mse

        # Normalizing the MSE with the number of batches
        mse /= len(batches)

        logger.info(f'MSE: {mse}')

        return mse, visible_probs
