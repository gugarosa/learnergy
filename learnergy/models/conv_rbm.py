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

    def __init__(self, visible_shape=(28, 28), filter_shape=(7, 7), n_filters=5, n_channels=1, steps=1, learning_rate=0.1):
        """Initialization method.

        Args:


        """

        logger.info('Overriding class: Model -> ConvRBM.')

        # Override its parent class
        super(ConvRBM, self).__init__()

        # Amount of visible units
        self.visible_shape = visible_shape

        # Amount of hidden units
        self.filter_shape = filter_shape

        #
        self.hidden_shape = (visible_shape[0] - filter_shape[0] + 1, visible_shape[1] - filter_shape[1] + 1)

        #
        self.n_filters = n_filters

        #
        self.n_channels = n_channels

        #
        self.steps = steps

        # Weights matrix
        self.W = nn.Parameter(torch.randn(n_filters, n_channels, filter_shape[0], filter_shape[1]) * 0.01)

        # Visible units bias
        self.a = nn.Parameter(torch.zeros(1))

        # Hidden units bias
        self.b = nn.Parameter(torch.zeros(n_filters))

        # Creating the optimizer object
        self.optimizer = opt.SGD(self.parameters(), lr=learning_rate)

        logger.info('Class overrided.')

    def hidden_sampling(self, v, scale=False):
        """Performs the hidden layer sampling, i.e., P(h|v).

        Args:
            v (torch.Tensor): A tensor incoming from the visible layer.
            scale (bool): A boolean to decide whether temperature should be used or not.

        Returns:
            The probabilities and states of the hidden layer sampling.

        """

        print(v.size(), self.W.size(), self.b.size())

        # Calculating neurons' activations
        activations = F.conv2d(v, self.W, bias=self.b)

        print(activations.size())

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

        print(h.size(), self.W.permute(1, 0, 2, 3).size(), self.a.size())

        # Calculating neurons' activations
        activations = F.conv2d(h, self.W.permute(1, 0, 2, 3), bias=self.a)

        print(activations.size())

        # Calculate probabilities as usual
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
            visible_probs, visible_states = self.visible_sampling(
                neg_hidden_states, True)

            # Calculating hidden probabilities and states
            neg_hidden_probs, neg_hidden_states = self.hidden_sampling(
                visible_states, True)

        return pos_hidden_probs, pos_hidden_states, neg_hidden_probs, neg_hidden_states, visible_states

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
        v = torch.mv(samples, self.a)

        # Finally, gathers the system's energy
        energy = -v - h

        return energy

    def pseudo_likelihood(self, samples):
        """Calculates the logarithm of the pseudo-likelihood.

        Args:
            samples (torch.Tensor): Samples to be calculated.

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
        pl = torch.mean(self.n_visible *
                        torch.log(torch.sigmoid(e1 - e) + c.EPSILON))

        return pl
    
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

            # Resetting epoch's MSE and pseudo-likelihood to zero
            mse = 0
            pl = 0

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

            logger.info(f'MSE: {mse} | log-PL: {pl}')

        return mse, pl

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

        logger.info(f'MSE: {mse}')

        return mse, visible_probs

    def forward(self, x):
        """Performs a forward pass over the data.

        Args:
            x (torch.Tensor): An input tensor for computing the forward pass.

        Returns:
            A tensor containing the RBM's outputs.

        """

        # Calculates the outputs of the model
        x, _ = self.hidden_sampling(x)

        return x
