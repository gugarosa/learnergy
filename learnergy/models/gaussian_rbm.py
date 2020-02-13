import torch
import torch.nn as nn
import torch.nn.functional as F

import learnergy.utils.exception as e
import learnergy.utils.logging as l
from learnergy.models.rbm import RBM

logger = l.get_logger(__name__)


class GaussianRBM(RBM):
    """A GaussianRBM class provides the basic implementation for Gaussian-Bernoulli Restricted Boltzmann Machines.

    References:


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

        logger.info('Overriding class: RBM -> GaussianRBM.')

        # Override its parent class
        super(GaussianRBM, self).__init__(n_visible=n_visible, n_hidden=n_hidden, steps=steps,
                                          learning_rate=learning_rate, momentum=momentum, decay=decay, temperature=temperature, use_gpu=use_gpu)

        # Variance parameter
        self.sigma = nn.Parameter(torch.ones(n_visible))

        # Updating optimizer's parameters with `sigma`
        self.optimizer.add_param_group({'params': self.sigma})

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
            v (tensor): A tensor incoming from the visible layer.
            scale (bool): A boolean to decide whether temperature should be used or not.

        Returns:
            The probabilities and states of the hidden layer sampling.

        """

        # Calculating neurons' activations
        activations = F.linear(
            v / torch.pow(self.sigma, 2), self.W.t(), self.b)

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

        activations = F.linear(h, self.W, self.a)
        # activations = self.a + self.sigma * torch.mm(h, self.W.t())

        # print(activations.size())
        # print(self.sigma.size())
#
        sigma = torch.repeat_interleave(self.sigma, activations.size(0), dim=0)

        # print(self.sigma)

        # print(sigma.size())
#
        states = torch.normal(activations, torch.pow(sigma, 2))

        # Calculating neurons' activations
        # activations = F.linear(h, self.W, self.a)

        # # If scaling is true
        # if scale:
        #     # Calculate probabilities with temperature
        #     probs = torch.sigmoid(0.5 * activations / self.T)

        # # If scaling is false
        # else:
        #     # Calculate probabilities as usual
        #     probs = torch.sigmoid(0.5 * activations)

        # # Sampling current states
        # states = torch.bernoulli(probs)

        return states, activations

    def energy(self, samples):
        """Calculates and frees the system's energy.

        Args:
            samples (tensor): Samples to be energy-freed.

        Returns:
            The system's energy based on input samples.

        """

        # Calculate samples' activations
        activations = F.linear(
            samples / torch.pow(self.sigma, 2), self.W.t(), self.b)

        # print(activations.size())

        # Creating a Softplus function for numerical stability
        s = nn.Softplus()

        # Calculate the hidden term
        h = torch.sum(s(activations), dim=1)

        # Calculate the visible term
        # v = torch.mv(samples, self.a)
        # v = ((samples - self.a) ** 2 / (2 * self.sigma ** 2))

        a = self.a.expand(1, 784)

        v = torch.sum(torch.mm(samples / torch.pow(2 * self.sigma, 2), samples.t()), dim=1) - torch.mv(samples, self.a / torch.pow(self.sigma, 2)) + torch.mm(a / torch.pow(2 * self.sigma, 2), a.t())

        # print(v.size())
        # print(v_a.size())

        # v = torch.sum(v, dim=1)

        # Finally, gathers the system's energy
        # energy = -v - h
        energy = -v - h

        return energy
