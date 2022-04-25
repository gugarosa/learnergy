"""Gaussian-Bernoulli Restricted Boltzmann Machine.
"""

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
from learnergy.models.bernoulli import RBM

from torch.utils.data import DataLoader

logger = l.get_logger(__name__)


class GaussianRBM(RBM):
    """A GaussianRBM class provides the basic implementation for
    Gaussian-Bernoulli Restricted Boltzmann Machines (with standardization).

    Note that this classes normalize the data
    as it uses variance equals to one throughout its learning procedure.

    This is a trick to ease the calculations of the hidden and
    visible layer samplings, as well as the cost function.

    References:
        K. Cho, A. Ilin, T. Raiko.
        Improved learning of Gaussian-Bernoulli restricted Boltzmann machines.
        International conference on artificial neural networks (2011).

    """

    def __init__(self, n_visible=128, n_hidden=128, steps=1, learning_rate=0.1,
                 momentum=0, decay=0, temperature=1, use_gpu=False, normalize=True, input_normalize=True):
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
            normalize (boolean): Whether or not to use batch normalization
            input_normalize (boolean): Whether or not to normalize inputs

        """

        self._normalize = normalize
        self._input_normalize = input_normalize

        logger.info('Overriding class: RBM -> GaussianRBM.')

        super(GaussianRBM, self).__init__(n_visible, n_hidden, steps, learning_rate,
                                          momentum, decay, temperature, use_gpu)

        logger.info('Class overrided.')

 
    @property
    def normalize(self):
        """boolean: Whether or not to use batch normalization.

        """

        return self._normalize

    @normalize.setter
    def normalize(self, normalize):

        self._normalize = normalize

    @property
    def input_normalize(self):
        """boolean: Whether or not to use input normalization.

        """

        return self._input_normalize

    @input_normalize.setter
    def input_normalize(self, input_normalize):

        self._input_normalize = input_normalize


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

        # Passes states through a Sigmoid function 
        probs = torch.sigmoid(states)

        return probs, states

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
        batches = DataLoader(dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)

        # For every epoch
        for epoch in range(epochs):
            logger.info('Epoch %d/%d', epoch+1, epochs)

            # Calculating the time of the epoch's starting
            start = time.time()

            # Resetting epoch's MSE and pseudo-likelihood to zero
            mse = 0
            pl = 0

            # For every batch
            for samples, _ in tqdm(batches):
                if self.normalize:
                    # Normalizing the samples' batch
                    samples = ((samples - torch.mean(samples, 0, True)) / (torch.std(samples, 0, True) + 1e-6)).detach()

                # Flattening the samples' batch    
                samples = samples.reshape(len(samples), self.n_visible)

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
                batch_mse = torch.div(
                    torch.sum(torch.pow(samples - visible_states, 2)), batch_size).detach()

                # Calculating the current's batch logarithm pseudo-likelihood
                batch_pl = self.pseudo_likelihood(samples).detach()

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

            logger.info('MSE: %f | log-PL: %f', mse, pl)

        return mse, pl

    def forward(self, x):
        """Performs a forward pass over the data.

        Args:
            x (torch.Tensor): An input tensor for computing the forward pass.

        Returns:
            A tensor containing the RBM's outputs.

        """

        if self.input_normalize:
            # Normalizing the samples'
            x = ((x - torch.mean(x, 0, True)) / (torch.std(x, 0, True) + 1e-6)).detach()

        # Calculates the outputs of the model
        x, _ = self.hidden_sampling(x)

        return x

    def reconstruct(self, dataset):
        """Reconstructs batches of new samples.

        Args:
            dataset (torch.utils.data.Dataset): A Dataset object containing the testing data.

        Returns:
            Reconstruction error and visible probabilities, i.e., P(v|h).

        """

        logger.info('Reconstructing new samples ...')

        # Resetting MSE to zero
        mse = 0

        # Defining the batch size as the amount of samples in the dataset
        batch_size = len(dataset)

        # Transforming the dataset into training batches
        batches = DataLoader(dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)

        # For every batch
        for samples, _ in tqdm(batches):

            if self.normalize:
                # Normalizing the samples' batch
                samples = ((samples - torch.mean(samples, 0, True)) / (torch.std(samples, 0, True) + 1e-6)).detach()

            # Flattening the samples' batch
            samples = samples.reshape(len(samples), self.n_visible)

            # Checking whether GPU is avaliable and if it should be used
            if self.device == 'cuda':
                # Applies the GPU usage to the data
                samples = samples.cuda()

            # Calculating positive phase hidden probabilities and states
            _, pos_hidden_states = self.hidden_sampling(samples)

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

        logger.info('MSE: %f', mse)

        return mse, visible_probs



class GaussianReluRBM(GaussianRBM):
    """A GaussianReluRBM class provides the basic implementation for
    Gaussian-ReLU Restricted Boltzmann Machines (for raw pixels values).

    Note that this class requires raw data (integer-valued)
    in order to model the image covariance into a latent ReLU layer.

    References:
        G. Hinton. A practical guide to training restricted Boltzmann machines.
        Neural networks: Tricks of the trade (2012).

    """

    def __init__(self, n_visible=128, n_hidden=128, steps=1, learning_rate=0.001,
                 momentum=0, decay=0, temperature=1, use_gpu=False, normalize=True, input_normalize=True):
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
            normalize (boolean): Whether or not to use batch normalization
            input_normalize (boolean): Whether or not to normalize inputs

        """

        logger.info('Overriding class: GaussianRBM -> GaussianReluRBM.')

        # Override its parent class
        super(GaussianReluRBM, self).__init__(n_visible, n_hidden, steps, learning_rate,
                                              momentum, decay, temperature, use_gpu, normalize, input_normalize)

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


class GaussianSeluRBM(GaussianRBM):
    """A GaussianSeluRBM class provides the basic implementation for
    Gaussian-SeLU Restricted Boltzmann Machines (for raw pixels values).

    Note that this class requires raw data (integer-valued)
    in order to model the image covariance into a latent ReLU layer.

    References:
        G. Hinton. A practical guide to training restricted Boltzmann machines.
        Neural networks: Tricks of the trade (2012).

        G. Klambauer et al. Self-normalizing neural networks.
        Proceedings, NIPS (2017).
    """

    def __init__(self, n_visible=128, n_hidden=128, steps=1, learning_rate=0.001,
                 momentum=0, decay=0, temperature=1, use_gpu=False, normalize=False, input_normalize=True):
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
            normalize (boolean): Whether or not to use batch normalization
            input_normalize (boolean): Whether or not to normalize inputs

        """

        logger.info('Overriding class: GaussianRBM -> GaussianSeluRBM.')

        # Override its parent class
        super(GaussianSeluRBM, self).__init__(n_visible, n_hidden, steps, learning_rate,
                                              momentum, decay, temperature, use_gpu, normalize, input_normalize)

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
            probs = F.selu(torch.div(activations, self.T))

        # If scaling is false
        else:
            # Calculate probabilities as usual
            probs = F.selu(activations)

        # Current states equals probabilities
        states = probs

        return probs, states



class VarianceGaussianRBM(RBM):
    """A VarianceGaussianRBM class provides the basic implementation for
    Gaussian-Bernoulli Restricted Boltzmann Machines (without standardization).

    Note that this class implements a new cost function that takes in account
    a new learning parameter: variance (sigma).

    Therefore, there is no need to standardize the data, as the variance
    will be trained throughout the learning procedure.

    References:
        K. Cho, A. Ilin, T. Raiko.
        Improved learning of Gaussian-Bernoulli restricted Boltzmann machines.
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
            sigma = torch.repeat_interleave(
                self.sigma, activations.size(0), dim=0)

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
        v = torch.sum(
            torch.div(torch.pow(samples - self.a, 2), 2 * sigma), dim=1)

        # Finally, gathers the system's energy
        energy = -v - h

        return energy
