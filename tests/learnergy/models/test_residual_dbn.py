import pytest
import torch
import torchvision

from learnergy.models import residual_dbn


def test_residual_dbn_zetta1():
    new_residual_dbn = residual_dbn.ResidualDBN()

    assert new_residual_dbn.zetta1 == 1


def test_residual_dbn_zetta1_setter():
    new_residual_dbn = residual_dbn.ResidualDBN()

    try:
        new_residual_dbn.zetta1 = 'a'
    except:
        new_residual_dbn.zetta1 = 0.1

    assert new_residual_dbn.zetta1 == 0.1

    try:
        new_residual_dbn.zetta1 = -1
    except:
        new_residual_dbn.zetta1 = 0.1

    assert new_residual_dbn.zetta1 == 0.1


def test_residual_dbn_zetta2():
    new_residual_dbn = residual_dbn.ResidualDBN()

    assert new_residual_dbn.zetta2 == 1


def test_residual_dbn_zetta2_setter():
    new_residual_dbn = residual_dbn.ResidualDBN()

    try:
        new_residual_dbn.zetta2 = 'a'
    except:
        new_residual_dbn.zetta2 = 0.1

    assert new_residual_dbn.zetta2 == 0.1

    try:
        new_residual_dbn.zetta2 = -1
    except:
        new_residual_dbn.zetta2 = 0.1

    assert new_residual_dbn.zetta2 == 0.1


def test_residual_dbn_calculate_residual():
    new_residual_dbn = residual_dbn.ResidualDBN(n_visible=784, n_hidden=[128, 128], steps=[1, 1],
                                                learning_rate=[0.1, 0.1], momentum=[0, 0], decay=[0, 0], temperature=[1, 1], use_gpu=False)

    v = torch.ones(1, 784)

    res = new_residual_dbn.calculate_residual(v)

    assert res.size(1) == 784


# def test_residual_dbn_fit():
#     train = torchvision.datasets.MNIST(
#         root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())

#     new_residual_dbn = residual_dbn.ResidualDBN(n_visible=784, n_hidden=[128, 128], steps=[1, 1],
#                                                 learning_rate=[0.1, 0.1], momentum=[0, 0], decay=[0, 0], temperature=[1, 1], use_gpu=False)

#     e, pl = new_residual_dbn.fit(train, batch_size=128, epochs=[1, 1])

#     assert len(e) == 2
#     assert len(pl) == 2


def test_residual_dbn_forward():
    new_residual_dbn = residual_dbn.ResidualDBN(n_visible=784, n_hidden=[128, 128], steps=[1, 1],
                                                learning_rate=[0.1, 0.1], momentum=[0, 0], decay=[0, 0], temperature=[1, 1], use_gpu=False)

    v = torch.ones(1, 784)

    probs = new_residual_dbn.forward(v)

    assert probs.size(1) == 128
