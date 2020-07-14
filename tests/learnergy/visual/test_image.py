import numpy as np
import torch

from learnergy.visual import image


def test_rasterize():
    array = np.zeros((64, 64))

    d = int(np.sqrt(array.shape[0]))
    s = int(np.sqrt(array.shape[1]))

    rasterized_array = image._rasterize(array.T, img_shape=(
        d, d), tile_shape=(s, s), tile_spacing=(1, 1))

    assert rasterized_array.shape == (71, 71)

    rasterized_array = image._rasterize(array.T, img_shape=(
        d, d), tile_shape=(s, s), tile_spacing=(1, 1), scale=False)

    assert rasterized_array.shape == (71, 71)

    tuple = (np.zeros((64, 64)), np.zeros((64, 64)), np.zeros((64, 64)), None)

    rasterized_tuple = image._rasterize(tuple, img_shape=(
        d, d), tile_shape=(s, s), tile_spacing=(1, 1))

    assert rasterized_tuple.shape == (71, 71, 4)

    rasterized_tuple = image._rasterize(tuple, img_shape=(
        d, d), tile_shape=(s, s), tile_spacing=(1, 1), output=False)

    assert rasterized_tuple.shape == (71, 71, 4)


def test_create_mosaic():
    t = torch.zeros(64, 64)

    image.create_mosaic(t)
