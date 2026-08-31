import logging as stdlib_logging

import numpy as np
import pytest
import torch

from learnergy.core import Dataset
from learnergy.math.metrics import calculate_ssim
from learnergy.math.scale import unitary_scale
from learnergy.models.extra import SigmoidRBM, SigmoidRBM4Deep
from learnergy.utils import exception, logging
from learnergy.visual import convergence, image, tensor


def test_dataset_applies_transform():
    data = np.array([[1, 2], [3, 4]])
    targets = np.array([0, 1])
    dataset = Dataset(data, targets, lambda sample: sample * 2, show_log=False)

    sample, target = dataset[0]
    assert np.array_equal(sample, np.array([2, 4]))
    assert target == 0

    with pytest.raises(TypeError):
        dataset.transform = 1


def test_math_helpers():
    scaled = unitary_scale(np.array([1, 2, 3]))
    assert np.allclose(scaled, np.array([0, 0.5, 1]))

    originals = torch.rand(2, 8, 8)
    reconstructed = originals.reshape(2, 64)
    assert calculate_ssim(reconstructed, originals) == pytest.approx(1.0)


def test_exception_types_remain_builtin_compatible():
    with pytest.raises(TypeError):
        raise exception.TypeError("invalid type")
    with pytest.raises(ValueError):
        raise exception.ValueError("invalid value")


def test_logger_does_not_duplicate_handlers():
    logger = logging.get_logger("learnergy.tests.utilities")
    count = len(logger.handlers)
    assert logging.get_logger("learnergy.tests.utilities") is logger
    assert len(logger.handlers) == count == 2

    for handler in logger.handlers:
        if isinstance(handler, stdlib_logging.FileHandler):
            handler.close()


def test_visual_helpers(tmp_path, monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    convergence.plot([1, 2], labels=["metric"])

    with pytest.raises(exception.SizeError):
        convergence.plot([1, 2], labels=["one", "two"])

    output = tmp_path / "tensor.png"
    tensor.save_tensor(torch.rand(1, 4, 4), str(output))
    assert output.exists()

    raster = image._rasterize(
        np.arange(16).reshape(4, 4),
        img_shape=(2, 2),
        tile_shape=(2, 2),
    )
    assert raster.shape == (4, 4)


def test_deep_sigmoid_name_remains_available():
    assert issubclass(SigmoidRBM4Deep, SigmoidRBM)
