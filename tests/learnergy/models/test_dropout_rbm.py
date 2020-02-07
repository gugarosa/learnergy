import pytest
import torch
from torch.utils.data import DataLoader

import torchvision
from learnergy.models import dropout_rbm


def test_rbm_properties():
    new_dropout_rbm = dropout_rbm.DropoutRBM()

    assert new_dropout_rbm.p == 0.5


def test_rbm_hidden_sampling():
    new_dropout_rbm = dropout_rbm.DropoutRBM()

    v = torch.ones(1, 128)

    probs, states = new_dropout_rbm.hidden_sampling(v)

    assert probs.size(1) == 128
    assert states.size(1) == 128


def test_rbm_reconstruct():
    test = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

    test_batches = DataLoader(test, batch_size=10000,
                              shuffle=True, num_workers=1)

    new_dropout_rbm = dropout_rbm.DropoutRBM(n_visible=784, n_hidden=128, steps=1,
                                             learning_rate=0.1, momentum=0, decay=0, temperature=1, dropout=0.5)

    e, v = new_dropout_rbm.reconstruct(test_batches)

    assert e >= 0
    assert v.size(1) == 784
