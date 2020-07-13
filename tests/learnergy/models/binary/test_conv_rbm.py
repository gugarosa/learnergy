import pytest
import torch
import torchvision

from learnergy.models.binary import conv_rbm


def test_conv_rbm_visible_shape():
    new_conv_rbm = conv_rbm.ConvRBM()

    assert new_conv_rbm.visible_shape == (28, 28)


def test_conv_rbm_visible_shape_setter():
    new_conv_rbm = conv_rbm.ConvRBM()

    try:
        new_conv_rbm.visible_shape = 'a'
    except:
        new_conv_rbm.visible_shape = (28, 28)

    assert new_conv_rbm.visible_shape == (28, 28)


def test_conv_rbm_filter_shape():
    new_conv_rbm = conv_rbm.ConvRBM()

    assert new_conv_rbm.filter_shape == (7, 7)


def test_conv_rbm_filter_shape_setter():
    new_conv_rbm = conv_rbm.ConvRBM()

    try:
        new_conv_rbm.filter_shape = 'a'
    except:
        new_conv_rbm.filter_shape = (7, 7)

    assert new_conv_rbm.filter_shape == (7, 7)


def test_conv_rbm_hidden_shape():
    new_conv_rbm = conv_rbm.ConvRBM()

    assert new_conv_rbm.hidden_shape == (22, 22)


def test_conv_rbm_hidden_shape_setter():
    new_conv_rbm = conv_rbm.ConvRBM()

    try:
        new_conv_rbm.hidden_shape = 'a'
    except:
        new_conv_rbm.hidden_shape = (22, 22)

    assert new_conv_rbm.hidden_shape == (22, 22)


def test_conv_rbm_n_filters():
    new_conv_rbm = conv_rbm.ConvRBM()

    assert new_conv_rbm.n_filters == 5


def test_conv_rbm_n_filters_setter():
    new_conv_rbm = conv_rbm.ConvRBM()

    try:
        new_conv_rbm.n_filters = 'a'
    except:
        new_conv_rbm.n_filters = 1

    assert new_conv_rbm.n_filters == 1

    try:
        new_conv_rbm.n_filters = 0
    except:
        new_conv_rbm.n_filters = 1

    assert new_conv_rbm.n_filters == 1


def test_conv_rbm_n_channels():
    new_conv_rbm = conv_rbm.ConvRBM()

    assert new_conv_rbm.n_channels == 1


def test_conv_rbm_n_channels_setter():
    new_conv_rbm = conv_rbm.ConvRBM()

    try:
        new_conv_rbm.n_channels = 'a'
    except:
        new_conv_rbm.n_channels = 1

    assert new_conv_rbm.n_channels == 1

    try:
        new_conv_rbm.n_channels = 0
    except:
        new_conv_rbm.n_channels = 1

    assert new_conv_rbm.n_channels == 1


def test_conv_rbm_steps():
    new_conv_rbm = conv_rbm.ConvRBM()

    assert new_conv_rbm.steps == 1


def test_conv_rbm_steps_setter():
    new_conv_rbm = conv_rbm.ConvRBM()

    try:
        new_conv_rbm.steps = 'a'
    except:
        new_conv_rbm.steps = 1

    assert new_conv_rbm.steps == 1

    try:
        new_conv_rbm.steps = 0
    except:
        new_conv_rbm.steps = 1

    assert new_conv_rbm.steps == 1


def test_conv_rbm_lr():
    new_conv_rbm = conv_rbm.ConvRBM()

    assert new_conv_rbm.lr == 0.1


def test_conv_rbm_lr_setter():
    new_conv_rbm = conv_rbm.ConvRBM()

    try:
        new_conv_rbm.lr = 'a'
    except:
        new_conv_rbm.lr = 0.1

    assert new_conv_rbm.lr == 0.1

    try:
        new_conv_rbm.lr = -1
    except:
        new_conv_rbm.lr = 0.1

    assert new_conv_rbm.lr == 0.1


def test_conv_rbm_momentum():
    new_conv_rbm = conv_rbm.ConvRBM()

    assert new_conv_rbm.momentum == 0


def test_conv_rbm_momentum_setter():
    new_conv_rbm = conv_rbm.ConvRBM()

    try:
        new_conv_rbm.momentum = 'a'
    except:
        new_conv_rbm.momentum = 0.1

    assert new_conv_rbm.momentum == 0.1

    try:
        new_conv_rbm.momentum = -1
    except:
        new_conv_rbm.momentum = 0.1

    assert new_conv_rbm.momentum == 0.1


def test_conv_rbm_decay():
    new_conv_rbm = conv_rbm.ConvRBM()

    assert new_conv_rbm.decay == 0


def test_conv_rbm_decay_setter():
    new_conv_rbm = conv_rbm.ConvRBM()

    try:
        new_conv_rbm.decay = 'a'
    except:
        new_conv_rbm.decay = 0.1

    assert new_conv_rbm.decay == 0.1

    try:
        new_conv_rbm.decay = -1
    except:
        new_conv_rbm.decay = 0.1

    assert new_conv_rbm.decay == 0.1


def test_conv_rbm_W():
    new_conv_rbm = conv_rbm.ConvRBM()

    assert new_conv_rbm.W.size(0) == 5
    assert new_conv_rbm.W.size(1) == 1
    assert new_conv_rbm.W.size(2) == 7
    assert new_conv_rbm.W.size(3) == 7


def test_conv_rbm_W_setter():
    new_conv_rbm = conv_rbm.ConvRBM()

    try:
        new_conv_rbm.W = 1
    except:
        new_conv_rbm.W = torch.nn.Parameter(torch.randn(5, 1, 7, 7) * 0.01)

    assert new_conv_rbm.W.size(0) == 5
    assert new_conv_rbm.W.size(1) == 1
    assert new_conv_rbm.W.size(2) == 7
    assert new_conv_rbm.W.size(3) == 7


def test_conv_rbm_a():
    new_conv_rbm = conv_rbm.ConvRBM()

    assert new_conv_rbm.a.size(0) == 1


def test_conv_rbm_a_setter():
    new_conv_rbm = conv_rbm.ConvRBM()

    try:
        new_conv_rbm.a = 1
    except:
        new_conv_rbm.a = torch.nn.Parameter(torch.zeros(1))

    assert new_conv_rbm.a.size(0) == 1


def test_conv_rbm_b():
    new_conv_rbm = conv_rbm.ConvRBM()

    assert new_conv_rbm.b.size(0) == 5


def test_conv_rbm_b_setter():
    new_conv_rbm = conv_rbm.ConvRBM()

    try:
        new_conv_rbm.b = 1
    except:
        new_conv_rbm.b = torch.nn.Parameter(torch.zeros(5))

    assert new_conv_rbm.b.size(0) == 5


def test_conv_rbm_optimizer():
    new_conv_rbm = conv_rbm.ConvRBM()

    assert type(new_conv_rbm.optimizer).__name__ == 'SGD'


def test_conv_rbm_optimizer_setter():
    new_conv_rbm = conv_rbm.ConvRBM()

    try:
        new_conv_rbm.optimizer = 'OPT'
    except:
        new_conv_rbm.optimizer = torch.optim.SGD(
            new_conv_rbm.parameters(), lr=0.1)

    assert type(new_conv_rbm.optimizer).__name__ == 'SGD'


def test_conv_rbm_hidden_sampling():
    new_conv_rbm = conv_rbm.ConvRBM()

    v = torch.ones(5, 1, 7, 7)

    probs, states = new_conv_rbm.hidden_sampling(v)

    assert probs.size(1) == 5
    assert states.size(1) == 5


def test_conv_rbm_visible_sampling():
    new_conv_rbm = conv_rbm.ConvRBM()

    h = torch.ones(1, 5, 7, 7)

    probs, states = new_conv_rbm.visible_sampling(h)

    assert probs.size(1) == 1
    assert states.size(1) == 1


def test_conv_rbm_energy():
    new_conv_rbm = conv_rbm.ConvRBM()

    samples = torch.ones(5, 1, 7, 7)

    energy = new_conv_rbm.energy(samples)

    assert torch.mean(energy).detach().numpy() < 0


def test_conv_rbm_fit():
    train = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())

    new_conv_rbm = conv_rbm.ConvRBM(visible_shape=(28, 28), filter_shape=(1, 1), n_filters=1, n_channels=1,
                                    steps=1, learning_rate=0.01, momentum=0, decay=0, use_gpu=True)

    e = new_conv_rbm.fit(train, batch_size=128, epochs=1)

    assert e >= 0


def test_conv_rbm_reconstruct():
    test = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

    new_conv_rbm = conv_rbm.ConvRBM(visible_shape=(28, 28), filter_shape=(1, 1), n_filters=1, n_channels=1,
                                    steps=1, learning_rate=0.01, momentum=0, decay=0, use_gpu=True)

    e, v = new_conv_rbm.reconstruct(test)

    assert e >= 0
    assert v.size(1) == 1
    assert v.size(2) == 28
    assert v.size(3) == 28


def test_conv_rbm_forward():
    new_conv_rbm = conv_rbm.ConvRBM()

    v = torch.ones(1, 1, 28, 28)

    probs = new_conv_rbm.forward(v)

    assert probs.size(1) == 5
    assert probs.size(2) == 22
    assert probs.size(3) == 22
