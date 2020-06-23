import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import learnergy.utils.exception as e
import learnergy.utils.logging as l
from learnergy.models.binary import RBM

logger = l.get_logger(__name__)


class DiscriminativeRBM(RBM):
    """A DiscriminativeRBM class provides the basic implementation for
    Discriminative Bernoulli-Bernoulli Restricted Boltzmann Machines.

    References:
        H. Larochelle and Y. Bengio. Classification using discriminative restricted Boltzmann machines.
        Proceedings of the 25th international conference on Machine learning (2008).

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

        logger.info('Overriding class: RBM -> DiscriminativeRBM.')

        # Override its parent class
        super(DiscriminativeRBM, self).__init__(n_visible, n_hidden, steps, learning_rate,
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

    def labels_sampling(self, samples):
        """Calculates labels probabilities by samplings, i.e., P(y|v).

        Args:
            samples (torch.Tensor): Samples to be labels-calculated.

        Returns:
            Labels' probabilities based on input samples.

        """

        # Creating an empty tensor for holding the probabilities per class
        probs = torch.zeros(samples.size(0), self.n_classes, device=self.device)

        # Creating an empty tensor for holding the probabilities considering all classes
        probs_sum = torch.zeros(samples.size(0), device=self.device)

        # Calculate samples' activations
        activations = F.linear(samples, self.W.t(), self.b)

        # Creating a Softplus function for numerical stability
        s = nn.Softplus()

        # Iterating through every possible class
        for i in range(self.n_classes):
            # Calculates the logit-probability for the particular class
            probs[:, i] = self.c[i] + torch.sum(s(activations + self.U[i, :]), dim=1)

        # Recovering the predictions based on the logit-probabilities
        preds = torch.argmax(probs.detach(), 1)

        return probs, preds

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
            for samples, labels in tqdm(batches):
                # Flattening the samples' batch
                samples = samples.reshape(len(samples), self.n_visible)

                # Checking whether GPU is avaliable and if it should be used
                if self.device == 'cuda':
                    # Applies the GPU usage to the data and labels
                    samples = samples.cuda()
                    labels = labels.cuda()

                # Calculates labels probabilities and predictions by sampling
                probs, _ = self.labels_sampling(samples)

                # Calculates the loss for further gradients' computation
                cost = self.loss(probs, labels)

                # Initializing the gradient
                self.optimizer.zero_grad()

                # Computing the gradients
                cost.backward()

                # Updating the parameters
                self.optimizer.step()

                # Calculating labels predictions by sampling
                _, preds = self.labels_sampling(samples)

                # Gathering the size of the batch
                batch_size = samples.size(0)

                # Calculating current's batch accuracy
                batch_acc = torch.mean((torch.sum(preds == labels).float()) / batch_size)

                # Summing up to epochs' loss and accuracy
                loss += cost.detach()
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
        for samples, labels in tqdm(batches):
            # Flattening the samples' batch
            samples = samples.reshape(len(samples), self.n_visible)

            # Checking whether GPU is avaliable and if it should be used
            if self.device == 'cuda':
                # Applies the GPU usage to the data and labels
                samples = samples.cuda()
                labels = labels.cuda()

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

class HybridDiscriminativeRBM(DiscriminativeRBM):
    """A HybridDiscriminativeRBM class provides the basic implementation for
    Hybrid Discriminative Bernoulli-Bernoulli Restricted Boltzmann Machines.

    References:
        H. Larochelle and Y. Bengio. Classification using discriminative restricted Boltzmann machines.
        Proceedings of the 25th international conference on Machine learning (2008).

    """

    def __init__(self, n_visible=128, n_hidden=128, n_classes=1, steps=1, learning_rate=0.1,
                 alpha=0.01, momentum=0, decay=0, temperature=1, use_gpu=False):
        """Initialization method.

        Args:
            n_visible (int): Amount of visible units.
            n_hidden (int): Amount of hidden units.
            n_classes (int): Amount of classes.
            steps (int): Number of Gibbs' sampling steps.
            learning_rate (float): Learning rate.
            alpha (float): Amount of penalization to the generative loss.
            momentum (float): Momentum parameter.
            decay (float): Weight decay used for penalization.
            temperature (float): Temperature factor.
            use_gpu (boolean): Whether GPU should be used or not.

        """

        logger.info('Overriding class: DiscriminativeRBM -> HybridDiscriminativeRBM.')

        # Override its parent class
        super(HybridDiscriminativeRBM, self).__init__(n_visible, n_hidden, n_classes, steps,
                                                      learning_rate, momentum, decay, temperature,
                                                      use_gpu)

        # Defining a property for the generative loss penalization
        self.alpha = alpha

    @property
    def alpha(self):
        """float: Generative loss penalization.

        """

        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        if not (isinstance(alpha, float) or isinstance(alpha, int)):
            raise e.TypeError('`alpha` should be a float or integer')
        if alpha < 0:
            raise e.ValueError('`alpha` should be >= 0')

        self._alpha = alpha

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

    def class_sampling(self, h):
        """Performs the class layer sampling, i.e., P(y|h).

        Args:
            h (torch.Tensor): A tensor incoming from the hidden layer.

        Returns:
            The probabilities and states of the class layer sampling.

        """

        # Calculating neurons' activations
        activations = torch.exp(F.linear(h, self.U, self.c))

        # Normalizing activations to calculate probabilities
        # probs = torch.nn.functional.normalize(activations, p=1, dim=1)
        probs = torch.div(activations, torch.sum(activations, dim=1).unsqueeze(1))

        # Sampling current states
        states = torch.nn.functional.one_hot(torch.argmax(probs, dim=1), num_classes=self.n_classes).float()

        return probs, states

    def gibbs_sampling(self, v, y):
        """Performs the whole Gibbs sampling procedure.

        Args:
            v (torch.Tensor): A tensor incoming from the visible layer.
            y (torch.Tensor): A tensor incoming from the class layer.

        Returns:
            The probabilities and states of the hidden layer sampling (positive),
            the probabilities and states of the hidden layer sampling (negative)
            and the states of the visible layer sampling (negative). 

        """

        # Transforming labels to one-hot encoding
        y = torch.nn.functional.one_hot(y, num_classes=self.n_classes).float()

        # Calculating positive phase hidden probabilities and states
        pos_hidden_probs, pos_hidden_states = self.hidden_sampling(v, y)

        # Initially defining the negative phase
        neg_hidden_states = pos_hidden_states

        # Performing the Contrastive Divergence
        for _ in range(self.steps):
            # Calculating visible probabilities and states
            visible_probs, visible_states = self.visible_sampling(
                neg_hidden_states, True)

            # Calculating class probabilities and states
            class_probs, class_states = self.class_sampling(neg_hidden_states)

            # Calculating hidden probabilities and states
            neg_hidden_probs, neg_hidden_states = self.hidden_sampling(
                visible_states, class_states, True)

        return pos_hidden_probs, pos_hidden_states, neg_hidden_probs, neg_hidden_states, visible_states

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

            # Resetting epoch's losses and accuracy to zero
            d_loss, g_loss, loss = 0, 0, 0
            acc = 0

            # For every batch
            for samples, labels in tqdm(batches):
                # Flattening the samples' batch
                samples = samples.reshape(len(samples), self.n_visible)

                # Checking whether GPU is avaliable and if it should be used
                if self.device == 'cuda':
                    # Applies the GPU usage to the data and labels
                    samples = samples.cuda()
                    labels = labels.cuda()

                # Performs the Gibbs sampling procedure
                _, _, _, _, visible_states = self.gibbs_sampling(samples, labels)

                # Detaching the visible states from GPU for further computation
                visible_states = visible_states.detach()

                # Calculates discriminator labels probabilities by sampling
                disc_probs, _ = self.labels_sampling(samples)

                # Calculates the discriminator loss for further gradients' computation
                d_cost = self.loss(disc_probs, labels)   

                # Calculates the generator loss for further gradients' computation
                g_cost = -self.pseudo_likelihood(samples)             

                # Calculates the total loss
                cost = d_cost + self.alpha * g_cost

                # Initializing the gradient
                self.optimizer.zero_grad()

                # Computing the gradients
                cost.backward()

                # Updating the parameters
                self.optimizer.step()

                # Calculating labels predictions by sampling
                _, preds = self.labels_sampling(samples)

                # Gathering the size of the batch
                batch_size = samples.size(0)

                # Calculating current's batch accuracy
                batch_acc = torch.mean((torch.sum(preds == labels).float()) / batch_size)

                # Summing up to epochs' genator, discriminator and total loss, and accuracy
                d_loss += d_cost
                g_loss += g_cost
                loss += cost.detach()
                acc += batch_acc

            # Normalizing the losses and accuracy with the number of batches
            d_loss /= len(batches)
            g_loss /= len(batches)
            loss /= len(batches)
            acc /= len(batches)

            # Calculating the time of the epoch's ending
            end = time.time()

            # Dumps the desired variables to the model's history
            self.dump(d_loss=d_loss.item(), g_loss=g_loss.item(), loss=loss.item(),
                      acc=acc.item(), time=end-start)

            logger.info(f'Loss(D): {d_loss} | Loss(G): {g_loss} | Loss: {loss} | Accuracy: {acc}')

        return loss, acc
