# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Display weight and sample mosaics.

The internal rasterizer accepts flattened images or four RGBA channel arrays.
It can scale image values and return either integer pixels or arrays with the input dtype.

"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from learnergy.math.scale import unitary_scale


def _rasterize(
    x: np.ndarray | tuple[np.ndarray | None, ...],
    img_shape: tuple[int, int],
    tile_shape: tuple[int, int],
    tile_spacing: tuple[int, int] = (0, 0),
    scale: bool = True,
    output: bool = True,
) -> np.ndarray:
    if len(img_shape) != 2 or len(tile_shape) != 2 or len(tile_spacing) != 2:
        raise ValueError("`img_shape`, `tile_shape`, and `tile_spacing` should have two values.")

    out_shape = [
        (image + spacing) * tiles - spacing for image, tiles, spacing in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(x, tuple):
        if len(x) != 4:
            raise ValueError("`x` should contain four RGBA channels.")

        dtype = "uint8" if output else x[0].dtype
        out = np.zeros((*out_shape, 4), dtype=dtype)
        defaults = [0, 0, 0, 255] if output else [0.0, 0.0, 0.0, 1.0]
        for channel, values in enumerate(x):
            out[:, :, channel] = (
                defaults[channel]
                if values is None
                else _rasterize(
                    values,
                    img_shape,
                    tile_shape,
                    tile_spacing,
                    scale,
                    output,
                )
            )

        return out

    height, width = img_shape
    height_spacing, width_spacing = tile_spacing
    out = np.zeros(out_shape, dtype="uint8" if output else x.dtype)

    for row in range(tile_shape[0]):
        for column in range(tile_shape[1]):
            index = row * tile_shape[1] + column
            if index >= x.shape[0]:
                continue
            image = x[index].reshape(img_shape)
            if scale:
                image = unitary_scale(image)
            out[
                row * (height + height_spacing) : row * (height + height_spacing) + height,
                column * (width + width_spacing) : column * (width + width_spacing) + width,
            ] = image * (255 if output else 1)

    return out


def create_mosaic(tensor: torch.Tensor) -> None:
    """Display a square grid of flattened image filters.

    The largest complete square of filters is displayed through Pillow's external image viewer.
    Tensor data is detached and moved to CPU without modifying the input.

    Args:
        tensor: Filter matrix shaped (pixels_per_filter, filters), with square spatial filters.

    Raises:
        ValueError: A filter cannot be reshaped to the inferred square image.

    """

    array = tensor.detach().cpu().numpy()
    image_size = int(np.sqrt(array.shape[0]))
    tile_size = int(np.sqrt(array.shape[1]))

    image = Image.fromarray(
        _rasterize(
            array.T,
            img_shape=(image_size, image_size),
            tile_shape=(tile_size, tile_size),
            tile_spacing=(1, 1),
        )
    )

    image.show()


def create_rgb_mosaic(tensor: torch.Tensor, n_samples: int = 1) -> None:
    """Display a square grid of channel-first RGB samples.

    Samples are detached and moved to CPU before Matplotlib displays the grid.
    This function does not explicitly close the figure.

    Args:
        tensor: RGB images shaped (batch, 3, height, width).
        n_samples: Number of rows and columns in the displayed grid.

    """

    array = tensor.detach().cpu().permute(0, 2, 3, 1).numpy()

    for i in range(n_samples * n_samples):
        plt.subplot(n_samples, n_samples, i + 1)
        plt.axis("off")
        plt.imshow(array[i])

    plt.show()
