import torch

from learnergy.models.real import gaussian_conv_rbm


def test_gaussian_conv_rbm_hidden_sampling():
    new_gaussian_conv_rbm = gaussian_conv_rbm.GaussianConvRBM()

    v = torch.ones(5, 1, 7, 7)
    probs, states = new_gaussian_conv_rbm.hidden_sampling(v)

    assert probs.size(1) == 5
    assert states.size(1) == 5


def test_gaussian_conv_rbm_visible_sampling():
    new_gaussian_conv_rbm = gaussian_conv_rbm.GaussianConvRBM()

    h = torch.ones(1, 5, 7, 7)

    probs, states = new_gaussian_conv_rbm.visible_sampling(h)

    assert probs.size(1) == 1
    assert states.size(1) == 1
