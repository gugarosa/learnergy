# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import pytest

import learnergy
import learnergy.utils.exception as e
from learnergy.core import Model


def test_package_version():
    assert learnergy.__version__ == "2.0.2"


def test_model_public_attributes_have_descriptions():
    assert Model.device.__doc__
    assert Model.history.__doc__


def test_model_tracks_device_and_history():
    model = Model()

    assert model.device == "cpu"

    model.dump(mse=1.0)
    model.dump(mse=0.5, loss=2.0)

    assert model.history == {"mse": [1.0, 0.5], "loss": [2.0]}

    with pytest.raises(e.TypeError):
        model.device = "gpu"
