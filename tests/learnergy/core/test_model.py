import pytest

from learnergy.core import model


def test_model():
    new_model = model.Model(use_gpu=False)

    assert new_model.device == 'cpu'
