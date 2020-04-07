import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.data import DataLoader

import learnergy.utils.constants as c
import learnergy.utils.exception as e
import learnergy.utils.logging as l
from learnergy.models.rbm import RBM

logger = l.get_logger(__name__)


class DRBM(RBM):
    """A DRBM class provides the basic implementation for Discriminative Bernoulli-Bernoulli Restricted Boltzmann Machines.

    References:
        

    """

    def __init__(self, n_visible=128, n_hidden=128, n_classes=1, steps=1,
                 learning_rate=0.1, momentum=0, decay=0, temperature=1, use_gpu=False):
        """Initialization method.

        Args:
            n_visible (int): Amount of visible units.
            n_hidden (int): Amount of hidden units.
            n_classes (int): Amount of classes.
            steps (int): Number of Gibbs' sampling steps.
            learning_rate (float): Learning rate.
            momentum (float): Momentum parameter.
            decay (float): Weight decay used for penalization.
            temperature (float): Temperature factor.
            use_gpu (boolean): Whether GPU should be used or not.

        """

        logger.info('Overriding class: RBM -> DRBM.')

        # Override its parent class
        super(DRBM, self).__init__(n_visible, n_hidden, steps, learning_rate,
                                   momentum, decay, temperature, use_gpu)
    
        # Number of classes
        self.n_classes = n_classes 

        # Class weights matrix
        self.U = nn.Parameter(torch.randn(n_classes, n_hidden) * 0.05)

        # Class bias
        self.c = nn.Parameter(torch.zeros(n_classes))

        # Creating the loss function for the DRBM
        self.loss = nn.CrossEntropyLoss()

        # Updating optimizer's parameters with `U`
        self.optimizer.add_param_group({'params': self.U})

        # Updating optimizer's parameters with `c`
        self.optimizer.add_param_group({'params': self.c})

        # Re-checks if current device is CUDA-based due to new parameter
        if self.device == 'cuda':
            # If yes, re-uses CUDA in the whole class
            self.cuda()

        logger.info('Class overrided.')

    @property
    def n_classes(self):
        """int: Number of classes.

        """

        return self._n_classes

    @n_classes.setter
    def n_classes(self, n_classes):
        if not isinstance(n_classes, int):
            raise e.TypeError('`n_classes` should be an integer')
        if n_classes <= 0:
            raise e.ValueError('`n_classes` should be > 0')

        self._n_classes = n_classes

    @property
    def U(self):
        """torch.nn.Parameter: Class weights' matrix.

        """

        return self._U

    @U.setter
    def U(self, U):
        if not isinstance(U, nn.Parameter):
            raise e.TypeError('`U` should be a PyTorch parameter')

        self._U = U

    @property
    def c(self):
        """torch.nn.Parameter: Class units bias.

        """

        return self._c

    @c.setter
    def c(self, c):
        if not isinstance(c, nn.Parameter):
            raise e.TypeError('`c` should be a PyTorch parameter')

        self._c = c

    @property
    def loss(self):
        """torch.nn.CrossEntropyLoss: Cross-Entropy loss function.

        """

        return self._loss

    @loss.setter
    def loss(self, loss):
        if not isinstance(loss, nn.CrossEntropyLoss):
            raise e.TypeError('`loss` should be a CrossEntropy')

        self._loss = loss

    def hidden_sampling(self, v, y, scale=False):
        """Performs the hidden layer sampling, i.e., P(h|y,v).

        Args:
            v (torch.Tensor): A tensor incoming from the visible layer.
            y (torch.Tensor): A tensor incoming from the class layer.
            scale (bool): A boolean to decide whether temperature should be used or not.

        Returns:
            The probabilities and states of the hidden layer sampling.

        """

        # Calculating neurons' activations
        activations = F.linear(v, self.W.t(), self.b) + torch.matmul(y, self.U)

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

    def class_sampling(self, h):
        """
        """

        #
        activations = F.linear(h, self.U, self.c)

        # Calculate probabilities as usual
        probs = torch.exp(activations)

        #
        probs = probs / torch.exp(F.linear(h, torch.sum(self.U), torch.sum(self.c)))

        # Sampling current states
        states = torch.nn.functional.one_hot(torch.argmax(probs, dim=1), n_classes=self.n_classes).float()

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

            #
            class_probs, class_states = self.class_sampling(neg_hidden_states)

            # Calculating hidden probabilities and states
            neg_hidden_probs, neg_hidden_states = self.hidden_sampling(
                visible_states, True)

        return pos_hidden_probs, pos_hidden_states, neg_hidden_probs, neg_hidden_states, visible_states

    def labels_sampling(self, samples):
        """Calculates labels probabilities by samplings, i.e., P(y|v).

        Args:
            samples (torch.Tensor): Samples to be labels-calculated

        Returns:
            Labels' probabilities based on input samples.

        """

        #
        y = torch.zeros(samples.size(0), self.n_classes)

        y_sum = torch.zeros(samples.size(0))

        # Calculate samples' activations
        activations = F.linear(samples, self.W.t(), self.b)

        # Creating a Softplus function for numerical stability
        s = nn.Softplus()

        for i in range(self.n_classes):
            y[:, i] = torch.exp(self.c[i]) + torch.sum(s(activations + self.U[i, :]), dim=1)

        y_sum = torch.exp(torch.sum(self.c)) + torch.sum(s(activations + torch.sum(self.U)), dim=1)

        for i in range(self.n_classes):
            y[:, i] /= y_sum

        preds = torch.argmax(y, 1)


        return y, preds

    def fit(self, dataset, batch_size=128, epochs=10):
        """Fits a new DRBM model.

        Args:
            dataset (torch.utils.data.Dataset): A Dataset object containing the training data.
            batch_size (int): Amount of samples per batch.
            epochs (int): Number of training epochs.

        Returns:
            Loss and accuracy from the training step.

        """

        # Transforming the dataset into training batches
        batches = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

        # For every epoch
        for e in range(epochs):
            logger.info(f'Epoch {e+1}/{epochs}')

            # Calculating the time of the epoch's starting
            start = time.time()

            # Resetting epoch's MSE and pseudo-likelihood to zero
            loss = 0
            acc = 0

            # For every batch
            for samples, labels in batches:
                # Flattening the samples' batch
                samples = samples.view(len(samples), self.n_visible)

                # Checking whether GPU is avaliable and if it should be used
                if self.device == 'cuda':
                    # Applies the GPU usage to the data
                    samples = samples.cuda()

                # Performs the Gibbs sampling procedure
                # _, _, _, _, visible_states = self.gibbs_sampling(samples)

                # Detaching the visible states from GPU for further computation
                # visible_states = visible_states.detach()

                # Calculates labels probabilities and predictions by sampling
                probs, preds = self.labels_sampling(samples)

                # Calculates the loss for further gradients' computation
                cost = self.loss(probs, labels)

                # Initializing the gradient
                self.optimizer.zero_grad()

                # Computing the gradients
                cost.backward()

                # Updating the parameters
                self.optimizer.step()

                # Gathering the size of the batch
                batch_size = samples.size(0)

                # Calculating current's batch accuracy
                batch_acc = torch.mean((torch.sum(preds == labels).float()) / batch_size)

                # Summing up to epochs' loss and accuracy
                loss += cost
                acc += batch_acc

            # Normalizing the loss and accuracy with the number of batches
            loss /= len(batches)
            acc /= len(batches)

            # Calculating the time of the epoch's ending
            end = time.time()

            # Dumps the desired variables to the model's history
            self.dump(loss=loss.item(), acc=acc.item(), time=end-start)

            logger.info(f'Loss: {loss} | Accuracy: {acc}')

        return loss, acc

    def predict(self, dataset):
        """Predicts batches of new samples.

        Args:
            dataset (torch.utils.data.Dataset): A Dataset object containing the testing data.

        Returns:
            Prediction probabilities and labels, i.e., P(y|v).

        """

        logger.info(f'Predicting new samples ...')

        # Resetting accuracy to zero
        acc = 0

        # Defining the batch size as the amount of samples in the dataset
        batch_size = len(dataset)

        # Transforming the dataset into training batches
        batches = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)

        # For every batch
        for samples, labels in batches:
            # Flattening the samples' batch
            samples = samples.view(len(samples), self.n_visible)

            # Checking whether GPU is avaliable and if it should be used
            if self.device == 'cuda':
                # Applies the GPU usage to the data
                samples = samples.cuda()

            # Calculating labels probabilities and predictions by sampling
            probs, preds = self.labels_sampling(samples)

            # Calculating current's batch accuracy
            batch_acc = torch.mean((torch.sum(preds == labels).float()) / batch_size)

            # Summing up the prediction accuracy
            acc += batch_acc

        # Normalizing the accuracy with the number of batches
        acc /= len(batches)

        logger.info(f'Accuracy: {acc}')

        return acc, probs, preds

