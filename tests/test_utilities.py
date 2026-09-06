# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import logging as stdlib_logging

import matplotlib.pyplot as plt
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
        raise exception.TypeError("`value` has an invalid type.")

    with pytest.raises(ValueError):
        raise exception.ValueError("`value` is invalid.")


@pytest.mark.parametrize(
    "message",
    ["`value` is invalid", "`value` is invalid."],
)
def test_value_error_preserves_message_and_formats_diagnostic(message, monkeypatch):
    diagnostics = []
    monkeypatch.setattr(exception.logger, "error", diagnostics.append)

    error = exception.ValueError(message)

    assert error.args == (message,)
    assert str(error) == message
    assert diagnostics == ["`exception=ValueError` was raised: `value` is invalid."]


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


@pytest.mark.parametrize("n_originals, n_reconstructed", [(1, 2), (2, 1)])
def test_ssim_rejects_unequal_batch_lengths(n_originals, n_reconstructed):
    sample = torch.arange(64, dtype=torch.float32).reshape(1, 8, 8)
    originals = sample.repeat(n_originals, 1, 1)
    reconstructed = sample.reshape(1, 64).repeat(n_reconstructed, 1)

    with pytest.raises(ValueError):
        calculate_ssim(reconstructed, originals)


@pytest.fixture
def existing_figures():
    figures = set(plt.get_fignums())
    yield figures
    for number in set(plt.get_fignums()) - figures:
        plt.close(number)


@pytest.mark.parametrize("shape", [(3, 5), (1, 5, 7), (3, 5, 7)])
def test_tensor_render_preserves_image_layout(shape, monkeypatch, existing_figures):
    samples = torch.rand(shape)
    rendered = []

    def capture_image():
        rendered.append(np.asarray(plt.gca().images[0].get_array()))

    monkeypatch.setattr(plt, "show", capture_image)
    tensor.show_tensor(samples)

    expected = samples.numpy()
    if len(shape) == 3:
        expected = np.moveaxis(expected, 0, -1) if shape[0] == 3 else expected[0]
    np.testing.assert_array_equal(rendered[0], expected)
    assert set(plt.get_fignums()) == existing_figures


@pytest.mark.parametrize("operation", ["save", "show"])
def test_tensor_render_closes_figure_after_failure(operation, tmp_path, existing_figures):
    samples = torch.zeros(2, 5, 7)

    with pytest.raises(TypeError):
        if operation == "save":
            tensor.save_tensor(samples, str(tmp_path / "invalid.png"))
        else:
            tensor.show_tensor(samples)

    assert set(plt.get_fignums()) == existing_figures
