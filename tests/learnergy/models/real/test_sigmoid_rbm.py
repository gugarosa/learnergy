import pytest
import torch

from learnergy.models.real import sigmoid_rbm


def test_sigmoid_rbm_hidden_sampling():
    new_sigmoid_rbm = sigmoid_rbm.SigmoidRBM()

    h = torch.ones(1, 128)

    probs, states = new_sigmoid_rbm.visible_sampling(h, scale=True)

    assert probs.size(1) == 128
    assert states.size(1) == 128

    probs, states = new_sigmoid_rbm.visible_sampling(h, scale=False)

    assert probs.size(1) == 128
    assert states.size(1) == 128
