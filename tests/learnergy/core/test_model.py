import pytest

from learnergy.core import model


def test_model():
    new_model = model.Model(use_gpu=False)

    assert new_model.device == 'cpu'

    try:
        new_model.device = 'gpu'
    except:
        new_model.device = 'cuda'

    assert new_model.device == 'cuda'

    try:
        new_model.history = []
    except:
        new_model.history = {}

    assert new_model.history == {}
