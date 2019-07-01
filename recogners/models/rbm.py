import torch

import recogners.utils.logging as l

logger = l.get_logger(__name__)


class RBM:
    """An RBM class provides the basic implementation for Restricted Boltzmann Machines.

    References:
        G. Hinton. A practical guide to training restricted Boltzmann machines. Neural networks: Tricks of the trade (2012).

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

        logger.info('Creating model: RBM.')

        # Setting default tensor type to Double
        torch.set_default_tensor_type(torch.DoubleTensor)

        # Amount of visible units
        self._n_visible = n_visible

        # Amount of hidden units
        self._n_hidden = n_hidden

        # Number of steps Gibbs' sampling steps
        self._steps = steps

        # Learning rate
        self._lr = learning_rate

        # Momentum parameter
        self._momentum = momentum

        # Weight decay
        self._decay = decay

        # Temperature factor
        self._T = temperature

        # Weights matrix
        self._W = torch.randn(n_visible, n_hidden) * 0.01

        # Visible units bias
        self._a = torch.zeros(1, n_visible)

        # Hidden units bias
        self._b = torch.zeros(1, n_hidden)

        logger.info('Model created.')
        logger.debug(
            f'Size: ({self._n_visible}, {self._n_hidden}) | Learning: CD-{self.steps} | Hyperparameters: lr = {self._lr}, momentum = {self._momentum}, decay = {self._decay}, T = {self._T}.')

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
    def steps(self):
        """int: Number of Gibbs' sampling steps.

        """

        return self._steps

    @property
    def lr(self):
        """float: The model's learning rate.

        """

        return self._lr

    @property
    def momentum(self):
        """float: Momentum parameter.

        """

        return self._momentum

    @property
    def decay(self):
        """float: Weight decay used for penalization.

        """

        return self._decay

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

    def hidden_sampling(self, v, scale=False):
        """Performs the hidden layer sampling, i.e., P(h|v).

        Args:
            v (tensor): A tensor incoming from the visible layer.
            scale (bool): A boolean to decide whether temperature should be used or not.

        Returns:
            The probabilities and states of the hidden layer sampling.

        """

        # Calculating neurons' activations
        activations = torch.mm(v, self.W) + self.b

        # If scaling is true
        if scale:
            # Calculate probabilities with temperature
            probs = torch.sigmoid(activations / self.T)

        # If scaling is false
        else:
            # Calculate probabilities as usual
            probs = torch.sigmoid(activations)

        # Sampling current states
        states = (probs > torch.rand(self.n_hidden)).double()

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
        activations = torch.mm(h, self.W.t()) + self.a

        # If scaling is true
        if scale:
            # Calculate probabilities with temperature
            probs = torch.sigmoid(activations / self.T)

        # If scaling is false
        else:
            # Calculate probabilities as usual
            probs = torch.sigmoid(activations)

        # Sampling current states
        states = (probs > torch.rand(self.n_visible)).double()

        return probs, states

    def energy(self, samples):
        """Calculates and frees the system's energy.

        Args:
            samples (tensor): Samples to be energy-freed.

        Returns:
            The system's energy based on input samples.

        """

        # Calculate samples' activations
        activations = torch.mm(samples, self.W) + self.b

        # Calculate the visible term
        v = torch.mm(samples, self.a.t())

        # Calculate the hidden term
        h = torch.sum(torch.log(1 + torch.exp(activations)), dim=1)

        # Finally, gathers the system's energy
        energy = -h - v

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
        bits = torch.randint(0, self.n_visible, size=(samples.size(0), 1))

        # Iterate through all samples in the batch
        for i in range(samples.size(0)):
            # Flips the bit on corresponding index
            samples_binary[i][bits[i]] = 1 - samples_binary[i][bits[i]]

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

        """

        # Creating weights, visible and hidden biases momentums
        w_momentum = torch.zeros(self.n_visible, self.n_hidden)
        a_momentum = torch.zeros(self.n_visible)
        b_momentum = torch.zeros(self.n_hidden)

        # For every epoch
        for e in range(epochs):
            logger.info(f'Epoch {e+1}/{epochs}')

            # Resetting epoch's error and pseudo-likelihood to zero
            error = 0
            pl = 0

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

                # Performing the Contrastive Divergence
                for _ in range(self.steps):
                    # Calculating negative phase hidden probabilities and states
                    neg_hidden_probs, neg_hidden_states = self.hidden_sampling(
                        visible_states, scale=True)

                    # Calculating visible probabilities and states
                    visible_probs, visible_states = self.visible_sampling(
                        neg_hidden_states, scale=True)

                # Building the positive and negative gradients
                pos_gradient = torch.mm(samples.t(), pos_hidden_probs)
                neg_gradient = torch.mm(visible_probs.t(), neg_hidden_probs)

                # Gathering the size of the batch
                batch_size = samples.size(0)

                # Calculating weights, visible and hidden biases momentums
                w_momentum = (w_momentum * self.momentum) + \
                    (self.lr * (pos_gradient - neg_gradient) / batch_size)

                a_momentum = (a_momentum * self.momentum) + \
                    (self.lr * torch.sum((samples - visible_probs), dim=0) / batch_size)

                b_momentum = (b_momentum * self.momentum) + \
                    (self.lr * torch.sum((pos_hidden_probs - neg_hidden_probs), dim=0) / batch_size)

                # Updating weights matrix, visible and hidden biases
                self.W += w_momentum - (self.W * self.decay)
                self.a += a_momentum
                self.b += b_momentum

                # Calculating current's batch error
                batch_error = torch.sum((samples - visible_states) ** 2) / batch_size

                # Calculating the logarithm of current's batch pseudo-likelihood
                batch_pl = self.pseudo_likelihood(samples)

                # Summing up to epochs' error and pseudo-likelihood
                error += batch_error
                pl += batch_pl

            # Normalizing the error and pseudo-likelihood with the number of batches
            error /= len(batches)
            pl /= len(batches)

            logger.info(f'Error: {error} | log-PL: {pl}')

    def reconstruct(self, batches):
        """Reconstruct batches of new samples.

        Args:
            batches (DataLoader): A DataLoader object containing batches to be reconstructed.

        """

        logger.info(f'Reconstructing new samples ...')

        # Resetting error to zero
        error = 0

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
            batch_error = torch.sum((samples - visible_states) ** 2) / batch_size

            # Summing up to reconstruction's error
            error += batch_error

        # Normalizing the error with the number of batches
        error /= len(batches)

        logger.info(f'Error: {error}')

        return visible_probs
