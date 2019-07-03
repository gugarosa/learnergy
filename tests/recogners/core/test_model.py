import os

import pytest

from recogners.core import model


def test_model_save():
    new_model = model.Model()

    new_model.save('model.pkl')

    assert os.path.isfile('./model.pkl')


def test_model_load():
    new_model = model.Model()

    new_model.load('model.pkl')
