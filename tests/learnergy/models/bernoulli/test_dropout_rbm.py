import torch
import torchvision

from learnergy.models.bernoulli import dropout_rbm


def test_dropout_rbm_p():
    new_dropout_rbm = dropout_rbm.DropoutRBM()

    assert new_dropout_rbm.p == 0.5


def test_dropout_rbm_p_setter():
    new_dropout_rbm = dropout_rbm.DropoutRBM()

    try:
        new_dropout_rbm.p = -1
    except:
        new_dropout_rbm.p = 0

    assert new_dropout_rbm.p == 0

    try:
        new_dropout_rbm.p = "a"
    except:
        new_dropout_rbm.p = 0

    assert new_dropout_rbm.p == 0


def test_dropout_rbm_hidden_sampling():
    new_dropout_rbm = dropout_rbm.DropoutRBM()

    v = torch.ones(1, 128)

    probs, states = new_dropout_rbm.hidden_sampling(v, scale=True)

    assert probs.size(1) == 128
    assert states.size(1) == 128

    probs, states = new_dropout_rbm.hidden_sampling(v, scale=False)

    assert probs.size(1) == 128
    assert states.size(1) == 128


def test_dropout_rbm_reconstruct():
    test = torchvision.datasets.KMNIST(
        root="./data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    new_dropout_rbm = dropout_rbm.DropoutRBM(
        n_visible=784,
        n_hidden=128,
        steps=1,
        learning_rate=0.1,
        momentum=0,
        decay=0,
        temperature=1,
        dropout=0.5,
        use_gpu=False,
    )

    e, v = new_dropout_rbm.reconstruct(test)

    assert e >= 0
    assert v.size(1) == 784


def test_dropconnect_rbm_hidden_sampling():
    new_dropconnect_rbm = dropout_rbm.DropConnectRBM()

    v = torch.ones(1, 128)

    probs, states = new_dropconnect_rbm.hidden_sampling(v, scale=True)

    assert probs.size(1) == 128
    assert states.size(1) == 128

    probs, states = new_dropconnect_rbm.hidden_sampling(v, scale=False)

    assert probs.size(1) == 128
    assert states.size(1) == 128


def test_dropconnect_rbm_reconstruct():
    test = torchvision.datasets.KMNIST(
        root="./data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    new_dropconnect_rbm = dropout_rbm.DropConnectRBM(
        n_visible=784,
        n_hidden=128,
        steps=1,
        learning_rate=0.1,
        momentum=0,
        decay=0,
        temperature=1,
        dropout=0.5,
        use_gpu=False,
    )

    e, v = new_dropconnect_rbm.reconstruct(test)

    assert e >= 0
    assert v.size(1) == 784
