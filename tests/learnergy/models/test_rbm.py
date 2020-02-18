import pytest
import torch
import torchvision

from learnergy.models import rbm


def test_rbm_properties():
    new_rbm = rbm.RBM()

    assert new_rbm.n_visible == 128
    assert new_rbm.n_hidden == 128
    assert new_rbm.steps == 1
    assert new_rbm.lr == 0.1
    assert new_rbm.momentum == 0
    assert new_rbm.decay == 0
    assert new_rbm.T == 1

    assert type(new_rbm.optimizer).__name__ == 'SGD'

    assert new_rbm.W.size(0) == 128
    assert new_rbm.W.size(1) == 128
    assert new_rbm.a.size(0) == 128
    assert new_rbm.b.size(0) == 128


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

    assert energy.detach().numpy() < 0


def test_rbm_pseudo_likelihood():
    new_rbm = rbm.RBM()

    samples = torch.ones(1, 128)

    pl = new_rbm.pseudo_likelihood(samples)

    assert pl.detach().numpy() < 0


def test_rbm_fit():
    train = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())

    new_rbm = rbm.RBM(n_visible=784, n_hidden=128, steps=1,
                      learning_rate=0.1, momentum=0, decay=0, temperature=1, use_gpu=False)

    e, pl = new_rbm.fit(train, batch_size=128, epochs=1)

    assert e >= 0
    assert pl <= 0


def test_rbm_reconstruct():
    test = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

    new_rbm = rbm.RBM(n_visible=784, n_hidden=128, steps=1,
                      learning_rate=0.1, momentum=0, decay=0, temperature=1)

    e, v = new_rbm.reconstruct(test)

    assert e >= 0
    assert v.size(1) == 784
