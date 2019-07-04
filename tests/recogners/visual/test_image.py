import numpy as np
import pytest
import torch

from recogners.visual import image


def test_rasterize():
    array = np.zeros((64, 64))

    d = int(np.sqrt(array.shape[0]))
    s = int(np.sqrt(array.shape[1]))

    rasterized_array = image.rasterize(array.T, img_shape=(
        d, d), tile_shape=(s, s), tile_spacing=(1, 1))

    assert rasterized_array.shape == (71, 71)


def test_create_mosaic():
    t = torch.zeros(64, 64)

    image.create_mosaic(t)


def test_show():
    t = torch.zeros(28, 28)

    image.show(t)
