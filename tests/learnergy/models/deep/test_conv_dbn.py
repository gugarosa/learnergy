import torch
import torchvision

from learnergy.models.deep import conv_dbn


def test_dbn_visible_shape():
    new_dbn = conv_dbn.ConvDBN()

    assert len(new_dbn.visible_shape) == 2


def test_dbn_visible_shape_setter():
    new_dbn = conv_dbn.ConvDBN()

    try:
        new_dbn.visible_shape = 'a'
    except:
        new_dbn.visible_shape = (1, 1)

    assert len(new_dbn.visible_shape) == 2


def test_dbn_filter_shape():
    new_dbn = conv_dbn.ConvDBN()

    assert len(new_dbn.filter_shape) == 1


def test_dbn_filter_shape_setter():
    new_dbn = conv_dbn.ConvDBN()

    try:
        new_dbn.filter_shape = 'a'
    except:
        new_dbn.filter_shape = (1,)

    assert len(new_dbn.filter_shape) == 1


def test_dbn_n_filters():
    new_dbn = conv_dbn.ConvDBN()

    assert len(new_dbn.n_filters) == 1


def test_dbn_n_filters_setter():
    new_dbn = conv_dbn.ConvDBN()

    try:
        new_dbn.n_filters = 'a'
    except:
        new_dbn.n_filters = (1,)

    assert len(new_dbn.n_filters) == 1


def test_dbn_n_channels():
    new_dbn = conv_dbn.ConvDBN()

    assert new_dbn.n_channels == 1


def test_dbn_n_channels_setter():
    new_dbn = conv_dbn.ConvDBN()

    try:
        new_dbn.n_channels = 0
    except:
        new_dbn.n_channels = 1

    assert new_dbn.n_channels == 1

    try:
        new_dbn.n_channels = 'a'
    except:
        new_dbn.n_channels = 1

    assert new_dbn.n_channels == 1


def test_dbn_n_layers():
    new_dbn = conv_dbn.ConvDBN()

    assert new_dbn.n_layers == 1


def test_dbn_n_layers_setter():
    new_dbn = conv_dbn.ConvDBN()

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
    new_dbn = conv_dbn.ConvDBN()

    assert len(new_dbn.steps) == 1


def test_dbn_steps_setter():
    new_dbn = conv_dbn.ConvDBN()

    try:
        new_dbn.steps = 'a'
    except:
        new_dbn.steps = (1,)

    assert len(new_dbn.steps) == 1

    try:
        new_dbn.steps = (1, 1)
    except:
        new_dbn.steps = (1,)

    assert len(new_dbn.steps) == 1


def test_dbn_lr():
    new_dbn = conv_dbn.ConvDBN()

    assert len(new_dbn.lr) == 1


def test_dbn_lr_setter():
    new_dbn = conv_dbn.ConvDBN()

    try:
        new_dbn.lr = 'a'
    except:
        new_dbn.lr = (0.1,)

    assert len(new_dbn.lr) == 1

    try:
        new_dbn.lr = (0.1, 0.1)
    except:
        new_dbn.lr = (0.1,)

    assert len(new_dbn.lr) == 1


def test_dbn_momentum():
    new_dbn = conv_dbn.ConvDBN()

    assert len(new_dbn.momentum) == 1


def test_dbn_momentum_setter():
    new_dbn = conv_dbn.ConvDBN()

    try:
        new_dbn.momentum = 'a'
    except:
        new_dbn.momentum = (0,)

    assert len(new_dbn.momentum) == 1

    try:
        new_dbn.momentum = (0, 0)
    except:
        new_dbn.momentum = (0,)

    assert len(new_dbn.momentum) == 1


def test_dbn_decay():
    new_dbn = conv_dbn.ConvDBN()

    assert len(new_dbn.decay) == 1


def test_dbn_decay_setter():
    new_dbn = conv_dbn.ConvDBN()

    try:
        new_dbn.decay = 'a'
    except:
        new_dbn.decay = (0,)

    assert len(new_dbn.decay) == 1

    try:
        new_dbn.decay = (0, 0)
    except:
        new_dbn.decay = (0,)

    assert len(new_dbn.decay) == 1


def test_dbn_models():
    new_dbn = conv_dbn.ConvDBN()

    assert len(new_dbn.models) == 1


def test_dbn_models_setter():
    new_dbn = conv_dbn.ConvDBN()

    try:
        new_dbn.models = 'a'
    except:
        new_dbn.models = []

    assert len(new_dbn.models) == 0
