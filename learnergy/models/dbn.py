import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.data import DataLoader

import learnergy.utils.constants as c
import learnergy.utils.exception as e
import learnergy.utils.logging as l
from learnergy.core.model import Model
from learnergy.models.rbm import RBM

logger = l.get_logger(__name__)


class DBN(Model):
    """A DBN class provides the basic implementation for Deep Belief Networks.

    References:
        

    """

    def __init__(self, n_visible=128, n_hidden=[128], steps=1, learning_rate=0.1, momentum=0, decay=0, temperature=1, use_gpu=False):
        """Initialization method.

        Args:
            n_visible (int): Amount of visible units.
            n_hidden (list): Amount of hidden units per layer.
            steps (int): Number of Gibbs' sampling steps.
            learning_rate (float): Learning rate.
            momentum (float): Momentum parameter.
            decay (float): Weight decay used for penalization.
            temperature (float): Temperature factor.
            use_gpu (boolean): Whether GPU should be used or not.

        """

        logger.info('Overriding class: Model -> DBN.')

        # Override its parent class
        super(DBN, self).__init__(use_gpu=use_gpu)

        # Amount of visible units
        self.n_visible = n_visible

        # Amount of hidden units per layer
        self.n_hidden = n_hidden

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

        # Number of layers
        self.n_layers = len(n_hidden)

        #
        self.rbms = []

        for i in range(self.n_layers):
            if i == 0:
                rbm = RBM(self.n_visible, self.n_hidden[i], self.steps, self.lr, self.momentum, self.decay, self.T, use_gpu)
            else:
                rbm = RBM(self.n_hidden[i-1], self.n_hidden[i], self.steps, self.lr, self.momentum, self.decay, self.T, use_gpu)
            self.rbms.append(rbm)

        # Checks if current device is CUDA-based
        if self.device == 'cuda':
            # If yes, uses CUDA in the whole class
            self.cuda()


        logger.info('Class overrided.')
        logger.debug(
            f'Size: ({self.n_visible}, {self.n_hidden}) | Layers: {self.n_layers} | Learning: CD-{self.steps} | Hyperparameters: lr = {self.lr}, momentum = {self.momentum}, decay = {self.decay}, T = {self.T}.')


    def fit(self, train, epochs=10):
        """
        """

        data = train.transform(train.data)
        targets = train.targets
        dataset = torch.utils.data.TensorDataset(data, targets)

        print(dataset.tensors[0][0])

        for rbm in self.rbms:
            batches = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=1)

            # rbm.fit(batches, epochs)

            # data = dataset.tensors[0].view(len(train), self.n_visible).float()

            # targets = dataset.tensors[1]
            
            # data, _ = rbm.hidden_sampling(data)

            # data = data.detach()

            # dataset = torch.utils.data.TensorDataset(data, targets)





