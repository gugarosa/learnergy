import pytest
import torch
import torchvision

from learnergy.models import dbn


def test_dbn_n_visible():
    new_dbn = dbn.DBN()

    assert new_dbn.n_visible == 128


def test_dbn_n_visible_setter():
    new_dbn = dbn.DBN()

    try:
        new_dbn.n_visible = 'a'
    except:
        new_dbn.n_visible = 1

    assert new_dbn.n_visible == 1

    try:
        new_dbn.n_visible = 0
    except:
        new_dbn.n_visible = 1

    assert new_dbn.n_visible == 1


def test_dbn_n_hidden():
    new_dbn = dbn.DBN()

    assert len(new_dbn.n_hidden) == 1


def test_dbn_n_hidden_setter():
    new_dbn = dbn.DBN()

    try:
        new_dbn.n_hidden = 'a'
    except:
        new_dbn.n_hidden = [128]

    assert len(new_dbn.n_hidden) == 1


def test_dbn_n_layers():
    new_dbn = dbn.DBN()

    assert new_dbn.n_layers == 1


def test_dbn_n_layers_setter():
    new_dbn = dbn.DBN()

    try:
        new_dbn.n_layers = 0
    except:
        new_dbn.n_layers = 1

    assert new_dbn.n_layers == 1

    try:
        new_dbn.n_layers = 'a'
    except:
        new_dbn.n_layers = 1

    assert new_dbn.n_layers == 1


def test_dbn_steps():
    new_dbn = dbn.DBN()

    assert len(new_dbn.steps) == 1


def test_dbn_steps_setter():
    new_dbn = dbn.DBN()

    try:
        new_dbn.steps = 'a'
    except:
        new_dbn.steps = [1]

    assert len(new_dbn.steps) == 1

    try:
        new_dbn.steps = [1, 1]
    except:
        new_dbn.steps = [1]

    assert len(new_dbn.steps) == 1


def test_dbn_lr():
    new_dbn = dbn.DBN()

    assert len(new_dbn.lr) == 1


def test_dbn_lr_setter():
    new_dbn = dbn.DBN()

    try:
        new_dbn.lr = 'a'
    except:
        new_dbn.lr = [0.1]

    assert len(new_dbn.lr) == 1

    try:
        new_dbn.lr = [0.1, 0.1]
    except:
        new_dbn.lr = [0.1]

    assert len(new_dbn.lr) == 1


def test_dbn_momentum():
    new_dbn = dbn.DBN()

    assert len(new_dbn.momentum) == 1


def test_dbn_momentum_setter():
    new_dbn = dbn.DBN()

    try:
        new_dbn.momentum = 'a'
    except:
        new_dbn.momentum = [0]

    assert len(new_dbn.momentum) == 1

    try:
        new_dbn.momentum = [0, 0]
    except:
        new_dbn.momentum = [0]

    assert len(new_dbn.momentum) == 1


def test_dbn_decay():
    new_dbn = dbn.DBN()

    assert len(new_dbn.decay) == 1


def test_dbn_decay_setter():
    new_dbn = dbn.DBN()

    try:
        new_dbn.decay = 'a'
    except:
        new_dbn.decay = [0]

    assert len(new_dbn.decay) == 1

    try:
        new_dbn.decay = [0, 0]
    except:
        new_dbn.decay = [0]

    assert len(new_dbn.decay) == 1


def test_dbn_T():
    new_dbn = dbn.DBN()

    assert len(new_dbn.T) == 1


def test_dbn_T_setter():
    new_dbn = dbn.DBN()

    try:
        new_dbn.T = 'a'
    except:
        new_dbn.T = [0]

    assert len(new_dbn.T) == 1

    try:
        new_dbn.T = [0, 0]
    except:
        new_dbn.T = [0]

    assert len(new_dbn.T) == 1


def test_dbn_models():
    new_dbn = dbn.DBN()

    assert len(new_dbn.models) == 1


def test_dbn_models_setter():
    new_dbn = dbn.DBN()

    try:
        new_dbn.models = 'a'
    except:
        new_dbn.models = []

    assert len(new_dbn.models) == 0


def test_dbn_fit():
    train = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())

    new_dbn = dbn.DBN(n_visible=784, n_hidden=[128, 128], steps=[1, 1],
                      learning_rate=[0.1, 0.1], momentum=[0, 0], decay=[0, 0], temperature=[1, 1], use_gpu=False)

    e, pl = new_dbn.fit(train, batch_size=128, epochs=[1, 1])

    assert len(e) == 2
    assert len(pl) == 2

def test_dbn_reconstruct():
    test = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

    new_dbn = dbn.DBN(n_visible=784, n_hidden=[128, 128], steps=[1, 1],
                      learning_rate=[0.1, 0.1], momentum=[0, 0], decay=[0, 0], temperature=[1, 1], use_gpu=False)

    e, v = new_dbn.reconstruct(test)

    assert e >= 0
    assert v.size(1) == 784
