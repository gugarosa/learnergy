import torch
import torch.nn as nn
import torch.nn.functional as F

import learnergy.utils.exception as e
import learnergy.utils.logging as l
from learnergy.models.binary import RBM

logger = l.get_logger(__name__)


class GaussianRBM(RBM):
    """A GaussianRBM class provides the basic implementation for Gaussian-Bernoulli Restricted Boltzmann Machines (with standardization).

    Note that this classes requires standardization of data as it uses variance equals to one throughout its learning procedure.
    This is a trick to ease the calculations of the hidden and visible layer samplings, as well as the cost function.

    References:
        K. Cho, A. Ilin, T. Raiko. Improved learning of Gaussian-Bernoulli restricted Boltzmann machines.
        International conference on artificial neural networks (2011).

    """

    def __init__(self, n_visible=128, n_hidden=128, steps=1, learning_rate=0.1,
                 momentum=0, decay=0, temperature=1, use_gpu=False):
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

        logger.info('Overriding class: RBM -> GaussianRBM.')

        # Override its parent class
        super(GaussianRBM, self).__init__(n_visible, n_hidden, steps, learning_rate,
                                          momentum, decay, temperature, use_gpu)

        logger.info('Class overrided.')
        
    def energy(self, samples):
        """Calculates and frees the system's energy.

        Args:
            samples (torch.Tensor): Samples to be energy-freed.

        Returns:
            The system's energy based on input samples.

        """

        # Calculate samples' activations
        activations = F.linear(samples, self.W.t(), self.b)

        # Creating a Softplus function for numerical stability
        s = nn.Softplus()

        # Calculate the hidden term
        h = torch.sum(s(activations), dim=1)

        # Calculate the visible term
        v = 0.5 * torch.sum((samples - self.a) ** 2, dim=1)

        # Finally, gathers the system's energy
        energy = v - h

        return energy     

    def visible_sampling(self, h, scale=False):
        """Performs the visible layer sampling, i.e., P(v|h).

        Args:
            h (torch.Tensor): A tensor incoming from the hidden layer.
            scale (bool): A boolean to decide whether temperature should be used or not.

        Returns:
            The probabilities and states of the visible layer sampling.

        """

        # Calculating neurons' activations
        activations = F.linear(h, self.W, self.a)

        # If scaling is true
        if scale:
            # Scale with temperature
            states = torch.div(activations, self.T)

        # If scaling is false
        else:
            # Gathers the states as usual
            states = activations

        return states, activations


class GaussianReluRBM(GaussianRBM):
    """A GaussianReluRBM class provides the basic implementation for Gaussian-ReLU Restricted Boltzmann Machines (for raw pixels values).

    Note that this class requires raw data (integer-valued) in order to model the image covariance into a latent ReLU layer.

    References:
        G. Hinton. A practical guide to training restricted Boltzmann machines.
        Neural networks: Tricks of the trade (2012).

    """

    def __init__(self, n_visible=128, n_hidden=128, steps=1, learning_rate=0.001,
                 momentum=0, decay=0, temperature=1, use_gpu=False):
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

        logger.info('Overriding class: GaussianRBM -> GaussianReluRBM.')

        # Override its parent class
        super(GaussianReluRBM, self).__init__(n_visible, n_hidden, steps, learning_rate,
                                              momentum, decay, temperature, use_gpu)

        logger.info('Class overrided.')


    def hidden_sampling(self, v, scale=False):
        """Performs the hidden layer sampling, i.e., P(h|v).

        Args:
            v (torch.Tensor): A tensor incoming from the visible layer.
            scale (bool): A boolean to decide whether temperature should be used or not.

        Returns:
            The probabilities and states of the hidden layer sampling.

        """

        # Calculating neurons' activations
        activations = F.linear(v, self.W.t(), self.b)

        # If scaling is true
        if scale:
            # Calculate probabilities with temperature
            probs = F.relu(torch.div(activations, self.T))

        # If scaling is false
        else:
            # Calculate probabilities as usual
            probs = F.relu(activations)

        # Current states equals probabilities
        states = probs

        return probs, states


class VarianceGaussianRBM(RBM):
    """A VarianceGaussianRBM class provides the basic implementation for Gaussian-Bernoulli Restricted Boltzmann Machines (without standardization).

    Note that this class implements a new cost function that takes in account a new learning parameter: variance (sigma). Therefore,
    there is no need to standardize the data, as the variance will be trained throughout the learning procedure.

    References:
        K. Cho, A. Ilin, T. Raiko. Improved learning of Gaussian-Bernoulli restricted Boltzmann machines.
        International conference on artificial neural networks (2011).

    """

    def __init__(self, n_visible=128, n_hidden=128, steps=1, learning_rate=0.1,
                 momentum=0, decay=0, temperature=1, use_gpu=False):
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

        logger.info('Overriding class: RBM -> VarianceGaussianRBM.')

        # Override its parent class
        super(VarianceGaussianRBM, self).__init__(n_visible, n_hidden, steps, learning_rate,
                                                  momentum, decay, temperature, use_gpu)

        # Variance parameter
        self.sigma = nn.Parameter(torch.ones(n_visible))

        # Updating optimizer's parameters with `sigma`
        self.optimizer.add_param_group({'params': self.sigma})

        # Re-checks if current device is CUDA-based due to new parameter
        if self.device == 'cuda':
            # If yes, re-uses CUDA in the whole class
            self.cuda()

        logger.info('Class overrided.')

    @property
    def sigma(self):
        """torch.nn.Parameter: Variance parameter.

        """

        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        if not isinstance(sigma, nn.Parameter):
            raise e.TypeError('`sigma` should be a PyTorch parameter')

        self._sigma = sigma

    def hidden_sampling(self, v, scale=False):
        """Performs the hidden layer sampling, i.e., P(h|v).

        Args:
            v (torch.Tensor): A tensor incoming from the visible layer.
            scale (bool): A boolean to decide whether temperature should be used or not.

        Returns:
            The probabilities and states of the hidden layer sampling.

        """

        # Calculating neurons' activations
        activations = F.linear(
            torch.div(v, torch.pow(self.sigma, 2)), self.W.t(), self.b)

        # If scaling is true
        if scale:
            # Calculate probabilities with temperature
            probs = torch.sigmoid(torch.div(activations, self.T))

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
            h (torch.Tensor): A tensor incoming from the hidden layer.
            scale (bool): A boolean to decide whether temperature should be used or not.

        Returns:
            The probabilities and states of the visible layer sampling.

        """

        # Calculating neurons' activations
        activations = F.linear(h, self.W, self.a)

        # Checks if device is CPU-based
        if self.device == 'cpu':
            # If yes, variance needs to have size equal to (batch_size, n_visible)
            sigma = torch.repeat_interleave(self.sigma, activations.size(0), dim=0)

        # If it is GPU-based
        else:
            # Variance needs to have size equal to (n_visible)
            sigma = self.sigma

        # Sampling current states from a Gaussian distribution
        states = torch.normal(activations, torch.pow(sigma, 2))

        return states, activations

    def energy(self, samples):
        """Calculates and frees the system's energy.

        Args:
            samples (torch.Tensor): Samples to be energy-freed.

        Returns:
            The system's energy based on input samples.

        """

        # Calculating the potency of variance
        sigma = torch.pow(self.sigma, 2)

        # Calculate samples' activations
        activations = F.linear(torch.div(samples, sigma), self.W.t(), self.b)

        # Creating a Softplus function for numerical stability
        s = nn.Softplus()

        # Calculate the hidden term
        h = torch.sum(s(activations), dim=1)

        # Calculate the visible term
        # Note that this might be improved
        v = torch.sum(torch.div(torch.pow(samples - self.a, 2), 2 * sigma), dim=1)

        # Finally, gathers the system's energy
        energy = -v - h

        return energy
