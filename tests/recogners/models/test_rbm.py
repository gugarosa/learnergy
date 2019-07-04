import pytest
import torch
import torchvision
from torch.utils.data import DataLoader

from recogners.models import rbm


def test_rbm_properties():
    new_rbm = rbm.RBM()

    assert new_rbm.n_visible == 128
    assert new_rbm.n_hidden == 128
    assert new_rbm.steps == 1
    assert new_rbm.lr == 0.1
    assert new_rbm.momentum == 0
    assert new_rbm.decay == 0
    assert new_rbm.T == 1

    assert new_rbm.W.size(0) == 128
    assert new_rbm.W.size(1) == 128
    assert new_rbm.a.size(1) == 128
    assert new_rbm.b.size(1) == 128


def test_rbm_hidden_sampling():
    new_rbm = rbm.RBM()

    v = torch.ones(1, 128)

    probs, states = new_rbm.hidden_sampling(v)

    assert probs.size(1) == 128
    assert states.size(1) == 128


def test_rbm_visible_sampling():
    new_rbm = rbm.RBM()

    h = torch.ones(1, 128)

    probs, states = new_rbm.visible_sampling(h)

    assert probs.size(1) == 128
    assert states.size(1) == 128


def test_rbm_energy():
    new_rbm = rbm.RBM()

    samples = torch.ones(1, 128)

    energy = new_rbm.energy(samples)

    assert energy.numpy() < 0


def test_rbm_pseudo_likelihood():
    new_rbm = rbm.RBM()

    samples = torch.ones(1, 128)

    pl = new_rbm.pseudo_likelihood(samples)

    assert pl.numpy() < 0


def test_rbm_fit():
    train = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())

    train_batches = DataLoader(
        train, batch_size=128, shuffle=True, num_workers=1)

    new_rbm = rbm.RBM(n_visible=784, n_hidden=128, steps=1,
                      learning_rate=0.1, momentum=0, decay=0, temperature=1)

    new_rbm.fit(train_batches, epochs=1)


def test_rbm_reconstruct():
    test = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

    test_batches = DataLoader(test, batch_size=10000,
                              shuffle=True, num_workers=1)

    new_rbm = rbm.RBM(n_visible=784, n_hidden=128, steps=1,
                      learning_rate=0.1, momentum=0, decay=0, temperature=1)

    v = new_rbm.reconstruct(test_batches)

    assert v.size(1) == 784
