import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import learnergy.utils.exception as e
import learnergy.utils.logging as l
from learnergy.models.binary import RBM

logger = l.get_logger(__name__)


class EDropoutRBM(RBM):
    """An EDropoutRBM class provides the basic implementation for Bernoulli-Bernoulli Restricted Boltzmann Machines
    along with a Energy-based Dropout regularization.

    References:
        M. Roder, G. H. de Rosa, A. L. D. Rossi, J. P. Papa.
        Energy-based Dropout in Restricted Boltzmann Machines: Why Do Not Go Random. Publication pending (2020).

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

        logger.info('Overriding class: RBM -> EDropoutRBM.')

        # Override its parent class
        super(EDropoutRBM, self).__init__(n_visible, n_hidden, steps, learning_rate,
                                          momentum, decay, temperature, use_gpu)

        # Initializes the Energy-based Dropout mask
        self.M = torch.Tensor()

        logger.info('Class overrided.')

    @property
    def M(self):
        """torch.Tensor: Energy-based Dropout mask.

        """

        return self._M

    @M.setter
    def M(self, M):
        if not isinstance(M, torch.Tensor):
            raise e.TypeError('`M` should be a PyTorch tensor')

        self._M = M

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
            probs = torch.mul(torch.sigmoid(
                torch.div(activations, self.T)), self.M)

        # If scaling is false
        else:
            # Calculate probabilities as usual
            probs = torch.mul(torch.sigmoid(activations), self.M)

        # Sampling current states
        states = torch.bernoulli(probs)

        return probs, states

    def total_energy(self, h, v):
        """Calculates the total energy of the model.

        Args:
            h (torch.Tensor): Hidden sampling states.
            v (torch.Tensor): Visible sampling states.

        Returns:
            The total energy of the model.

        """

        # Calculates the energy of the hidden layer
        e_h = -torch.mv(h, self.b)

        # Calculates the energy of the visible layer
        e_v = -torch.mv(v, self.a)

        # Calculates the energy of the reconstruction
        e_rec = -torch.mean(torch.mm(v, torch.mm(self.W, h.t())), dim=1)

        # Calculates the total energy
        energy = torch.mean(e_h + e_v + e_rec)

        return energy

    def energy_dropout(self, e, p_prob, n_prob):
        """Performs the Energy-based Dropout over the model.

        Args:
            e (torch.Tensor): Model's total energy.
            p_prob (torch.Tensor): Positive phase hidden probabilities.
            n_prob (torch.Tensor): Negative phase hidden probabilities.

        """

        # Calculates the Importance Level
        I = torch.div(torch.div(n_prob, p_prob), torch.abs(e))

        # Normalizes the Importance Level
        I = torch.div(I, torch.max(I, 0)[0])

        # Samples a probability tensor
        p = torch.rand((I.size(0), I.size(1)), device=self.device)

        # Calculates the Energy-based Dropout mask
        self.M = (I < p).float()

    def fit(self, dataset, batch_size=128, epochs=10):
        """Fits a new RBM model.

        Args:
            dataset (torch.utils.data.Dataset): A Dataset object containing the training data.
            batch_size (int): Amount of samples per batch.
            epochs (int): Number of training epochs.

        Returns:
            MSE (mean squared error), log pseudo-likelihood and time from the training step.

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
                # Gathering the size of the batch
                batch_size = samples.size(0)

                # Returns the Energy-based Dropout mask to one
                self.M = torch.ones(
                    (batch_size, self.n_hidden), device=self.device)

                # Flattening the samples' batch
                samples = samples.reshape(len(samples), self.n_visible)

                # Checking whether GPU is avaliable and if it should be used
                if self.device == 'cuda':
                    # Applies the GPU usage to the data
                    samples = samples.cuda()

                # Performs the initial Gibbs sampling procedure (pre-dropout)
                pos_hidden_probs, pos_hidden_states, neg_hidden_probs, neg_hidden_states, visible_states = self.gibbs_sampling(
                    samples)

                # Calculating energy of positive phase sampling
                e = self.total_energy(pos_hidden_states, samples)

                # Calculating energy of negative phase sampling
                e1 = self.total_energy(neg_hidden_states, visible_states)

                # Performing the energy-based dropout
                self.energy_dropout(e1 - e, pos_hidden_probs, neg_hidden_probs)

                # Performs the post Gibbs sampling procedure (post-dropout)
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

                # Calculating current's batch MSE
                batch_mse = torch.div(
                    torch.sum(torch.pow(samples - visible_states, 2)), batch_size)

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
            # Returns the Energy-based Dropout mask to one
            self.M = torch.ones(
                (batch_size, self.n_hidden), device=self.device)

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
