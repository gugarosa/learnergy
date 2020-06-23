import torch
import torch.nn.functional as F

import learnergy.utils.exception as e
import learnergy.utils.logging as l
from learnergy.core import Dataset
from learnergy.models.stack import DBN

logger = l.get_logger(__name__)


class ResidualDBN(DBN):
    """A ResidualDBN class provides the basic implementation for Residual-based Deep Belief Networks.

    References:
        M. Roder, et al. A Layer-Wise Information Reinforcement Approach to Improve Learning in Deep Belief Networks.
        International Conference on Artificial Intelligence and Soft Computing (2020).

    """

    def __init__(self, model='bernoulli', n_visible=128, n_hidden=[128], steps=[1],
                 learning_rate=[0.1], momentum=[0], decay=[0], temperature=[1],
                 zetta1=1, zetta2=1, use_gpu=False):
        """Initialization method.

        Args:
            model (str): Indicates which type of RBM should be used to compose the ResidualDBN.
            n_visible (int): Amount of visible units.
            n_hidden (list): Amount of hidden units per layer.
            steps (list): Number of Gibbs' sampling steps per layer.
            learning_rate (list): Learning rate per layer.
            momentum (list): Momentum parameter per layer.
            decay (list): Weight decay used for penalization per layer.
            temperature (list): Temperature factor per layer.
            zetta1 (float): Penalization factor for original learning.
            zetta2 (float): Penalization factor for residual learning.
            use_gpu (boolean): Whether GPU should be used or not.

        """

        logger.info('Overriding class: DBN -> ResidualDBN.')

        # Override its parent class
        super(ResidualDBN, self).__init__(model, n_visible, n_hidden, steps, learning_rate,
                                          momentum, decay, temperature, use_gpu)

        # Defining a property for holding the original learning's penalization
        self.zetta1 = zetta1

        # Defining a property for holding the residual learning's penalization
        self.zetta2 = zetta2

    @property
    def zetta1(self):
        """float: Penalization factor for original learning.

        """

        return self._zetta1

    @zetta1.setter
    def zetta1(self, zetta1):
        if not (isinstance(zetta1, float) or isinstance(zetta1, int)):
            raise e.TypeError('`zetta1` should be a float or integer')
        if zetta1 < 0:
            raise e.ValueError('`zetta1` should be >= 0')

        self._zetta1 = zetta1

    @property
    def zetta2(self):
        """float: Penalization factor for residual learning.

        """

        return self._zetta2

    @zetta2.setter
    def zetta2(self, zetta2):
        if not (isinstance(zetta2, float) or isinstance(zetta2, int)):
            raise e.TypeError('`zetta2` should be a float or integer')
        if zetta2 < 0:
            raise e.ValueError('`zetta2` should be >= 0')

        self._zetta2 = zetta2

    def calculate_residual(self, pre_activations):
        """Calculates the residual learning over input.

        Args:
            pre_activations (torch.Tensor): Pre-activations to be used.

        Returns:
            The residual learning based on input pre-activations.

        """

        # Calculating the residual values
        residual = F.relu(pre_activations)

        # Normalizing the values
        residual = torch.div(residual, torch.max(residual))

        return residual

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

            # Calculates pre-activation values
            pre_activation = model.pre_activation(samples)

            # Performs a forward pass over the samples
            samples, _ = model.hidden_sampling(samples)

            # Aggregates the residual learning
            samples = torch.mul(samples, self.zetta1) + torch.mul(self.calculate_residual(pre_activation), self.zetta2)

            # Normalizes the input for the next layer
            samples = torch.div(samples, torch.max(samples))

            # Checking whether GPU is being used
            if self.device == 'cuda':
                # If yes, get samples back to the CPU
                samples = samples.cpu()

            # Detaches the variable from the computing graph
            samples = samples.detach()

        return mse, pl

    def forward(self, x):
        """Re-writes the forward pass for classification purposes.

        Args:
            x (torch.Tensor): An input tensor for computing the forward pass.

        Returns:
            A tensor containing the DBN's outputs.
       
        """

        # For every possible model
        for model in self.models:
            # Calculates the pre-activations of current model
            pre_activation = model.pre_activation(x)

            # Performs a forward pass over the input
            x, _ = model.hidden_sampling(x)

            # Aggregates the residual learning
            x = torch.mul(x, self.zetta1) + torch.mul(self.calculate_residual(pre_activation), self.zetta2)

            # Normalizes the input for the next layer
            x = torch.div(x, torch.max(x))

        return x
