"""Continuous-based Convolutional Deep Belief Network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.data import DataLoader
from tqdm import tqdm

import learnergy.utils.logging as l
from learnergy.models.binary import ConvRBM
from learnergy.models.real import GaussianConvRBM
from learnergy.core import Dataset, Model

logger = l.get_logger(__name__)

MODELS = {
    'conv_rbm': ConvRBM,
    'cont_conv_rbm': GaussianConvRBM
}


class CDBN(Model):
    """A Continuous ConvDBN class provides the basic implementation for
    Continuous-based input Convolutional DBNs.

    References:
        H. Lee, et al.
        Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations.
        Proceedings of the 26th annual international conference on machine learning (2009).

    """

    def __init__(self, visible_shape=(28, 28), filter_shape=[(7, 7), (7, 7)], n_filters=[16, 16], n_channels=1,
                 n_layers=2, steps=1, learning_rate=(0.1,), momentum=(0,), decay=(0,), use_gpu=False):
        """Initialization method.

        Args:
            visible_shape (tuple): Shape of visible units.
            filter_shape (list of tuple): Shape of filters for each CRBM.
            n_filters (list of int): Number of filters for each CRBM.
            n_channels (int): Number of channels.
            n_layers (int): Number of layers
            steps (int): Number of Gibbs' sampling steps.
            learning_rate (float): Learning rate.
            momentum (float): Momentum parameter.
            decay (float): Weight decay used for penalization.
            use_gpu (boolean): Whether GPU should be used or not.

        """

        logger.info('Overriding class: Model -> CDBN.')

        # Override its parent class
        super(CDBN, self).__init__(use_gpu=use_gpu)

        # Shape of visible units
        self.visible_shape = visible_shape

        # Shape of filters
        self.filter_shape = filter_shape

        # Number of filters
        self.n_filters = n_filters

        # Number of channels
        self.n_channels = n_channels

        # Number of layers
        self.n_layers = n_layers

        # Number of steps Gibbs' sampling steps
        self.steps = steps

        # Learning rate
        self.lr = learning_rate

        # Momentum parameter
        self.momentum = momentum

        # Weight decay
        self.decay = decay

        # List of models (RBMs)
        self.models = []

        # For every possible layer
        for i in range(self.n_layers):
            model = 'cont_conv_rbm'

            # Shape of hidden units
            self.hidden_shape = (
                visible_shape[0] - filter_shape[i][0] + 1,
                visible_shape[1] - filter_shape[i][1] + 1)

            # Creates an CRBM
            m = MODELS[model](visible_shape=visible_shape, filter_shape=filter_shape[i], n_filters=n_filters[i],
                              n_channels=n_channels, steps=1, learning_rate=learning_rate[i],
                              momentum=momentum[i], decay=decay[i], use_gpu=use_gpu)

            # The new visible input stands for the hidden output incoming from the previous RBM
            visible_shape = (visible_shape[0] - filter_shape[i][0] + 1, visible_shape[1] - filter_shape[i][1] + 1)
            n_channels = n_filters[i]

            # Appends the model to the list
            self.models.append(m)

        # Checks if current device is CUDA-based
        if self.device == 'cuda':
            # If yes, uses CUDA in the whole class
            self.cuda()

        logger.info('Class overrided.')

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
        probs = F.relu6(activations)

        return probs, probs

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
        probs = torch.clamp(F.relu(activations), 0, 1)
        # probs = torch.sigmoid(activations)

        return probs, probs

    def fit(self, dataset, batch_size=128, epochs=(10, 10)):
        """Fits a new CDBN model.
        Args:
            dataset (torch.utils.data.Dataset | Dataset): A Dataset object containing the training data.
            batch_size (int): Amount of samples per batch.
            epochs (tuple): Number of training epochs per layer.
        Returns:
            MSE (mean squared error) from the training step.
        """

        # Initializing MSE and pseudo-likelihood as lists
        mse, pl = [], []

        # Initializing the dataset's variables
        try:
            samples, targets, transform = dataset.data.numpy(), dataset.targets.numpy(), dataset.transform
        except:
            # If the dataset is not a numpy array
            import numpy as np
            samples, targets, transform = dataset.data, dataset.targets, dataset.transform
            samples = np.array(samples)
            targets = np.array(targets)

        # For every possible model (CRBM)
        for i, model in enumerate(self.models):
            logger.info('Fitting layer %d/%d ...', i + 1, self.n_layers)

            # Creating the dataset
            d = Dataset(samples, targets, transform)

            # Fits the RBM
            model_mse = model.fit(d, batch_size, epochs[i])

            # Appending the metrics
            mse.append(model_mse)
            # pl.append(model_pl)

            # If is not the last model
            if i < len(self.models) - 1:
                # If the dataset has a transform
                if d.transform:
                    # Applies the transform over the samples
                    samples = torch.tensor(samples, dtype=torch.float).detach()
                # If there is no transform
                else:
                    # Just gather the samples
                    samples = d.data

                # Gathers the targets
                targets = d.targets

                # Gathers the transform callable from current dataset
                transform = None

                # Performs a forward pass over the samples to get their probabilities
                samples = model.propagate(d)

                # Checking whether GPU is being used
                if self.device == 'cuda':
                    # If yes, get samples back to the CPU
                    samples = samples.cpu()

            # Detaches the variable from the computing graph
            samples = samples.detach()

        return mse  # , pl

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
            # Flattening the samples' batch
            samples = samples.reshape(
                len(samples), self.n_channels, self.visible_shape[0], self.visible_shape[1])

            # Checking whether GPU is avaliable and if it should be used
            if self.device == 'cuda':
                # Applies the GPU usage to the data
                samples = samples.cuda()

            # Applying the initial hidden probabilities as the samples
            hidden_probs = samples

            # For every possible model (CRBM)
            for model in self.models:
                # Performing a hidden layer sampling
                hidden_probs, _ = model.hidden_sampling(hidden_probs)

            # Applying the initial visible probabilities as the hidden probabilities
            visible_probs = hidden_probs

            # For every possible model (CRBM)
            for model in reversed(self.models):
                # Performing a visible layer sampling
                visible_probs, visible_states = model.visible_sampling(
                    visible_probs)

            # Calculating current's batch reconstruction MSE
            batch_mse = torch.div(
                torch.sum(torch.pow(samples - visible_states, 2)), batch_size)

            # Summing up to reconstruction's MSE
            mse += batch_mse

        # Normalizing the MSE with the number of batches
        mse /= len(batches)

        logger.info('MSE: %f', mse)

        return mse, visible_probs

    def forward(self, x):
        """Performs a forward pass over the data.

        Args:
            x (torch.Tensor): An input tensor for computing the forward pass.

        Returns:
            A tensor containing the Convolutional RBM's outputs.

        """

        # For every possible model
        for model in self.models:
            # Calculates the outputs of the model
            x, _ = model.hidden_sampling(x)

        return x
