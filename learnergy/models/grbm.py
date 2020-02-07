import torch

import learnergy.utils.logging as l
from learnergy.models.rbm import RBM

logger = l.get_logger(__name__)


class GRBM(RBM):
    """A Gaussian-Bernoulli RBM class provides the basic implementation for Restricted Boltzmann Machines.

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

        logger.info('Overriding class: RBM -> GRBM.')

        # Override its parent class
        super(GRBM, self).__init__(n_visible=n_visible, n_hidden=n_hidden, steps=steps,
                                         learning_rate=learning_rate, momentum=momentum,
                                         decay=decay, temperature=temperature)

        logger.info('Class overrided.')

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
            probs = activations / self.T

        # If scaling is false
        else:
            # Calculate probabilities as usual
            probs = activations
        
        return probs

    def energy(self, samples):
        """Calculates and frees the system's energy.

        TO-DO: REWRITE THE ENERGY FOR GAUSSIAN INPUT!

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
        
        TO-DO: NOT OPTIMAL FOR CONTINUOuS INPUT!

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
        """Fits a new GBRBM model. 

        TO-DO: INPUT MUST BE COLUMN-WISE NORMALIZED [0,1]!

        Args:
            batches (DataLoader): A DataLoader object containing the training batches.
            epochs (int): Number of training epochs.

        Returns:
            Error and log pseudo-likelihood from the training step.

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

                # Calculating visible probabilities
                visible_probs = self.visible_sampling(
                    pos_hidden_states)

                # Performing the Contrastive Divergence
                for _ in range(self.steps):
                    # Calculating negative phase hidden probabilities and states
                    neg_hidden_probs, neg_hidden_states = self.hidden_sampling(
                        visible_probs, scale=True)

                    # Calculating visible probabilities and states
                    visible_probs = self.visible_sampling(
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
                batch_error = torch.sum((samples - visible_probs) ** 2) / batch_size

                # Calculating the logarithm of current's batch pseudo-likelihood
                batch_pl = self.pseudo_likelihood(samples)

                # Summing up to epochs' error and pseudo-likelihood
                error += batch_error
                pl += batch_pl

            # Normalizing the error and pseudo-likelihood with the number of batches
            error /= len(batches)
            pl /= len(batches)

            # Dumps the desired variables to the model's history
            self.dump(error=error, pl=pl)

            logger.info(f'Error: {error} | log-PL: {pl}')

        return error, pl

    def reconstruct(self, batches):
        """Reconstruct batches of new samples.

        Args:
            batches (DataLoader): A DataLoader object containing batches to be reconstructed.

        Returns:
            Reconstruction error and visible probabilities, i.e., P(v|h).

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
            visible_probs = self.visible_sampling(
                pos_hidden_states)

            # Gathering the size of the batch
            batch_size = samples.size(0)

            # Calculating current's batch reconstruction error
            batch_error = torch.sum((samples - visible_probs) ** 2) / batch_size

            # Summing up to reconstruction's error
            error += batch_error

        # Normalizing the error with the number of batches
        error /= len(batches)

        logger.info(f'Error: {error}')

        return error, visible_probs
