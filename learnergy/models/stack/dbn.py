import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import learnergy.utils.constants as c
import learnergy.utils.exception as e
import learnergy.utils.logging as l
from learnergy.core import Dataset, Model
from learnergy.models.binary import RBM, DropoutRBM, EDropoutRBM
from learnergy.models.real import (GaussianRBM, GaussianReluRBM, SigmoidRBM,
                                   VarianceGaussianRBM)

logger = l.get_logger(__name__)

MODELS = {
    'bernoulli': RBM,
    'dropout': DropoutRBM,
    'e_dropout': EDropoutRBM,
    'gaussian': GaussianRBM,
    'gaussian_relu': GaussianReluRBM,
    'sigmoid': SigmoidRBM,
    'variance_gaussian': VarianceGaussianRBM
}


class DBN(Model):
    """A DBN class provides the basic implementation for Deep Belief Networks.

    References:
        G. Hinton, S. Osindero, Y. Teh. A fast learning algorithm for deep belief nets.
        Neural computation (2006).

    """

    def __init__(self, model='bernoulli', n_visible=128, n_hidden=[128], steps=[1],
                 learning_rate=[0.1], momentum=[0], decay=[0], temperature=[1], use_gpu=False):
        """Initialization method.

        Args:
            model (str): Indicates which type of RBM should be used to compose the DBN.
            n_visible (int): Amount of visible units.
            n_hidden (list): Amount of hidden units per layer.
            steps (list): Number of Gibbs' sampling steps per layer.
            learning_rate (list): Learning rate per layer.
            momentum (list): Momentum parameter per layer.
            decay (list): Weight decay used for penalization per layer.
            temperature (list): Temperature factor per layer.
            use_gpu (boolean): Whether GPU should be used or not.

        """

        logger.info('Overriding class: Model -> DBN.')

        # Override its parent class
        super(DBN, self).__init__(use_gpu=use_gpu)

        # Amount of visible units
        self.n_visible = n_visible

        # Amount of hidden units per layer
        self.n_hidden = n_hidden

        # Number of layers
        self.n_layers = len(n_hidden)

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

        # List of models (RBMs)
        self.models = []

        # self.fc = torch.nn.Linear(self.n_hidden[self.n_layers-1], 10)

        # For every possible layer
        for i in range(self.n_layers):
            # If it is the first layer
            if i == 0:
                # Gathers the number of input units as number of visible units
                n_input = self.n_visible

            # If it is not the first layer
            else:
                # Gathers the number of input units as previous number of hidden units
                n_input = self.n_hidden[i-1]

                # After creating the first layer, we need to change the model's type to sigmoid
                model = 'sigmoid'

            # Creates an RBM
            m = MODELS[model](n_input, self.n_hidden[i], self.steps[i],
                              self.lr[i], self.momentum[i], self.decay[i], self.T[i], use_gpu)

            # Appends the model to the list
            self.models.append(m)

        # Checks if current device is CUDA-based
        if self.device == 'cuda':
            # If yes, uses CUDA in the whole class
            self.cuda()

        logger.info('Class overrided.')
        logger.debug(f'Number of layers: {self.n_layers}.')

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
        """list: List of hidden units.

        """

        return self._n_hidden

    @n_hidden.setter
    def n_hidden(self, n_hidden):
        if not isinstance(n_hidden, list):
            raise e.TypeError('`n_hidden` should be a list')

        self._n_hidden = n_hidden

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
        """list: Number of steps Gibbs' sampling steps per layer.

        """

        return self._steps

    @steps.setter
    def steps(self, steps):
        if not isinstance(steps, list):
            raise e.TypeError('`steps` should be a list')
        if len(steps) != self.n_layers:
            raise e.SizeError(
                f'`steps` should have size equal as {self.n_layers}')

        self._steps = steps

    @property
    def lr(self):
        """list: Learning rate per layer.

        """

        return self._lr

    @lr.setter
    def lr(self, lr):
        if not isinstance(lr, list):
            raise e.TypeError('`lr` should be a list')
        if len(lr) != self.n_layers:
            raise e.SizeError(
                f'`lr` should have size equal as {self.n_layers}')

        self._lr = lr

    @property
    def momentum(self):
        """list: Momentum parameter per layer.

        """

        return self._momentum

    @momentum.setter
    def momentum(self, momentum):
        if not isinstance(momentum, list):
            raise e.TypeError('`momentum` should be a list')
        if len(momentum) != self.n_layers:
            raise e.SizeError(
                f'`momentum` should have size equal as {self.n_layers}')

        self._momentum = momentum

    @property
    def decay(self):
        """list: Weight decay per layer.

        """

        return self._decay

    @decay.setter
    def decay(self, decay):
        if not isinstance(decay, list):
            raise e.TypeError('`decay` should be a list')
        if len(decay) != self.n_layers:
            raise e.SizeError(
                f'`decay` should have size equal as {self.n_layers}')

        self._decay = decay

    @property
    def T(self):
        """list: Temperature factor per layer.

        """

        return self._T

    @T.setter
    def T(self, T):
        if not isinstance(T, list):
            raise e.TypeError('`T` should be a list')
        if len(T) != self.n_layers:
            raise e.SizeError(f'`T` should have size equal as {self.n_layers}')

        self._T = T

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

    def fit(self, dataset, batch_size=128, epochs=[10]):
        """Fits a new DBN model.

        Args:
            dataset (torch.utils.data.Dataset | Dataset): A Dataset object containing the training data.
            batch_size (int): Amount of samples per batch.
            epochs (list): Number of training epochs per layer.

        Returns:
            MSE (mean squared error) and log pseudo-likelihood from the training step.

        """

        # Checking if the length of number of epochs' list is correct
        if len(epochs) != self.n_layers:
            # If not, raises an error
            raise e.SizeError(
                f'`epochs` should have size equal as {self.n_layers}')

        # Initializing MSE and pseudo-likelihood as lists
        mse, pl = [], []

        # Initializing the dataset's variables
        samples, targets, transform = dataset.data.numpy(), dataset.targets.numpy(), dataset.transform

        # For every possible model (RBM)
        for i, model in enumerate(self.models):
            logger.info(f'Fitting layer {i+1}/{self.n_layers} ...')

            # Creating the dataset
            d = Dataset(samples, targets, transform)

            # Fits the RBM
            model_mse, model_pl = model.fit(d, batch_size, epochs[i])

            # Appending the metrics
            mse.append(model_mse)
            pl.append(model_pl)

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
            samples = samples.reshape(len(dataset), model.n_visible)

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

        return mse, pl

    def reconstruct(self, dataset):
        """Reconstructs batches of new samples.

        Args:
            dataset (torch.utils.data.Dataset): A Dataset object containing the training data.

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
            samples = samples.reshape(batch_size, self.models[0].n_visible)

            # Checking whether GPU is avaliable and if it should be used
            if self.device == 'cuda':
                # Applies the GPU usage to the data
                samples = samples.cuda()

            # Applying the initial hidden probabilities as the samples
            hidden_probs = samples

            # For every possible model (RBM)
            for i, model in enumerate(self.models):
                # Flattening the hidden probabilities
                hidden_probs = hidden_probs.reshape(batch_size, model.n_visible)

                # Performing a hidden layer sampling
                hidden_probs, hidden_states = model.hidden_sampling(hidden_probs)

            # Applying the initial visible probabilities as the hidden probabilities
            visible_probs = hidden_probs

            # For every possible model (RBM)
            for i, model in enumerate(reversed(self.models)):
                # Flattening the visible probabilities
                visible_probs = visible_probs.reshape(batch_size, model.n_hidden)

                # Performing a visible layer sampling
                visible_probs, visible_states = model.visible_sampling(visible_probs)

            # Calculating current's batch reconstruction MSE
            batch_mse = torch.div(
                torch.sum(torch.pow(samples - visible_states, 2)), batch_size)

            # Summing up to reconstruction's MSE
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
            A tensor containing the DBN's outputs.
       
        """

        # For every possible model
        for model in self.models:
            # Calculates the outputs of current model
            x, _ = model.hidden_sampling(x)

        return x
