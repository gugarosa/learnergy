"""Convolutional Deep Belief Network.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import learnergy.utils.exception as e
import learnergy.utils.logging as l
from learnergy.core import Dataset, Model
from learnergy.models.bernoulli import ConvRBM
from learnergy.models.gaussian import GaussianConvRBM

logger = l.get_logger(__name__)

MODELS = {
    'bernoulli': ConvRBM,
    'gaussian': GaussianConvRBM
}


class ConvDBN(Model):
    """A ConvDBN class provides the basic implementation for Convolutional DBNs.

    References:
        H. Lee, et al.
        Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations.
        Proceedings of the 26th annual international conference on machine learning (2009).

    """

    def __init__(self, model='bernoulli', visible_shape=(28, 28), filter_shape=((7, 7),), n_filters=(16,),
                 n_channels=1, steps=(1,), learning_rate=(0.1,), momentum=(0,), decay=(0,), use_gpu=False):
        """Initialization method.

        Args:
            model (str): Indicates which type of ConvRBM should be used to compose the DBN.
            visible_shape (tuple): Shape of visible units.
            filter_shape (tuple of tuples): Shape of filters per layer.
            n_filters (tuple): Number of filters per layer.
            n_channels (int): Number of channels.
            steps (tuple): Number of Gibbs' sampling steps per layer.
            learning_rate (tuple): Learning rate per layer.
            momentum (tuple): Momentum parameter per layer.
            decay (tuple): Weight decay used for penalization per layer.
            use_gpu (boolean): Whether GPU should be used or not.

        """

        logger.info('Overriding class: Model -> ConvDBN.')

        # Override its parent class
        super(ConvDBN, self).__init__(use_gpu=use_gpu)

        # Shape of visible units
        self.visible_shape = visible_shape

        # Shape of filters
        self.filter_shape = filter_shape

        # Number of filters
        self.n_filters = n_filters

        # Number of channels
        self.n_channels = n_channels

        # Number of layers
        self.n_layers = len(n_filters)

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
            # Creates an CRBM
            m = MODELS[model](visible_shape, self.filter_shape[i], self.n_filters[i],
                              n_channels, self.steps[i], self.lr[i], self.momentum[i], self.decay[i], use_gpu)

            # Re-defines the visible shape
            visible_shape = (visible_shape[0] - self.filter_shape[i][0] + 1,
                             visible_shape[1] - self.filter_shape[i][1] + 1)

            # Also defines the new number of channels
            n_channels = self.n_filters[i]

            # Appends the model to the list
            self.models.append(m)

        # Checks if current device is CUDA-based
        if self.device == 'cuda':
            # If yes, uses CUDA in the whole class
            self.cuda()

        logger.info('Class overrided.')

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
            raise e.TypeError('`filter_shape` should be a tuple of tuples')

        self._filter_shape = filter_shape

    @property
    def n_filters(self):
        """tuple: Number of filters.

        """

        return self._n_filters

    @n_filters.setter
    def n_filters(self, n_filters):
        if not isinstance(n_filters, tuple):
            raise e.TypeError('`n_filters` should be a tuple')

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
    def n_layers(self):
        """int: Number of layers.

        """

        return self._n_layers

    @n_layers.setter
    def n_layers(self, n_layers):
        if not isinstance(n_layers, int):
            raise e.TypeError('`n_layers` should be an integer')
        if n_layers <= 0:
            raise e.ValueError('`n_layers` should be > 0')

        self._n_layers = n_layers

    @property
    def steps(self):
        """tuple: Number of steps Gibbs' sampling steps per layer.

        """

        return self._steps

    @steps.setter
    def steps(self, steps):
        if not isinstance(steps, tuple):
            raise e.TypeError('`steps` should be a tuple')
        if len(steps) != self.n_layers:
            raise e.SizeError(
                f'`steps` should have size equal as {self.n_layers}')

        self._steps = steps

    @property
    def lr(self):
        """tuple: Learning rate per layer.

        """

        return self._lr

    @lr.setter
    def lr(self, lr):
        if not isinstance(lr, tuple):
            raise e.TypeError('`lr` should be a tuple')
        if len(lr) != self.n_layers:
            raise e.SizeError(
                f'`lr` should have size equal as {self.n_layers}')

        self._lr = lr

    @property
    def momentum(self):
        """tuple: Momentum parameter per layer.

        """

        return self._momentum

    @momentum.setter
    def momentum(self, momentum):
        if not isinstance(momentum, tuple):
            raise e.TypeError('`momentum` should be a tuple')
        if len(momentum) != self.n_layers:
            raise e.SizeError(
                f'`momentum` should have size equal as {self.n_layers}')

        self._momentum = momentum

    @property
    def decay(self):
        """tuple: Weight decay per layer.

        """

        return self._decay

    @decay.setter
    def decay(self, decay):
        if not isinstance(decay, tuple):
            raise e.TypeError('`decay` should be a tuple')
        if len(decay) != self.n_layers:
            raise e.SizeError(
                f'`decay` should have size equal as {self.n_layers}')

        self._decay = decay

    @property
    def models(self):
        """list: List of models (RBMs).

        """

        return self._models

    @models.setter
    def models(self, models):
        if not isinstance(models, list):
            raise e.TypeError('`models` should be a list')

        self._models = models

    def fit(self, dataset, batch_size=128, epochs=(10, 10)):
        """Fits a new ConvDBN model.

        Args:
            dataset (torch.utils.data.Dataset | Dataset): A Dataset object containing the training data.
            batch_size (int): Amount of samples per batch.
            epochs (tuple): Number of training epochs per layer.

        Returns:
            MSE (mean squared error) from the training step.

        """

        # Checking if the length of number of epochs' list is correct
        if len(epochs) != self.n_layers:
            # If not, raises an error
            raise e.SizeError(('`epochs` should have size equal as %d', self.n_layers))

        # Initializing MSE as a list
        mse = []

        # Initializing the dataset's variables
        samples, targets, transform = dataset.data.numpy(), dataset.targets.numpy(), dataset.transform

        # For every possible model (ConvRBM)
        for i, model in enumerate(self.models):
            logger.info('Fitting layer %d/%d ...', i + 1, self.n_layers)

            # Creating the dataset
            d = Dataset(samples, targets, transform)

            # Fits the RBM
            model_mse = model.fit(d, batch_size, epochs[i])

            # Appending the metrics
            mse.append(model_mse)

            # If the dataset has a transform
            if d.transform:
                # Applies the transform over the samples
                samples = d.transform(d.data)

            # If there is no transform
            else:
                # Just gather the samples
                samples = d.data

            # Checking whether GPU is avaliable and if it should be used
            if self.device == 'cuda':
                # Applies the GPU usage to the data
                samples = samples.cuda()

            # Reshape the samples into an appropriate shape
            samples = samples.reshape(len(dataset), model.n_channels, model.visible_shape[0], model.visible_shape[1])

            # Gathers the targets
            targets = d.targets

            # Gathers the transform callable from current dataset
            transform = None

            # Performs a forward pass over the samples to get their probabilities
            samples, _ = model.hidden_sampling(samples)

            # Checking whether GPU is being used
            if self.device == 'cuda':
                # If yes, get samples back to the CPU
                samples = samples.cpu()

            # Detaches the variable from the computing graph
            samples = samples.detach()

        return mse

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
        batches = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)

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
                visible_probs, visible_states = model.visible_sampling(visible_probs)

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
            A tensor containing the ConvDBN's outputs.

        """

        # For every possible model
        for model in self.models:
            # Calculates the outputs of the model
            x, _ = model.hidden_sampling(x)

        return x
