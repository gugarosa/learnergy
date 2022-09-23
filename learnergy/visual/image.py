"""Image-related visualization.
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import learnergy.math.scale as scl


def _rasterize(
    x: np.array,
    img_shape: Tuple[int, int],
    tile_shape: Tuple[int, int],
    tile_spacing: Optional[Tuple[int, int]] = (0, 0),
    scale: Optional[bool] = True,
    output: Optional[bool] = True,
) -> np.array:
    """Rasterizes and prepares an image to be outputted as a mosaic.

    Args:
        x: An input array to be rasterized.
        img_shape: A tuple for the image shape.
        tile_shape: A tuple holding the shape of each tile.
        tile_spacing: A tuple containing the spacing between tiles.
        scale: If output array should be scaled between 0 and 1.
        output: If output values should be returned as pixels or not.

    Returns:
        (np.array): Rasterized version of input array.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(x, tuple):
        assert len(x) == 4

        if output:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype="uint8")
            channel_defaults = [0, 0, 0, 255]
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=x[0].dtype)
            channel_defaults = [0.0, 0.0, 0.0, 1.0]

        for i in range(4):
            if x[i] is None:
                # If there is no channel, fill it with zeros of the correct dtype
                out_array[:, :, i] = (
                    np.zeros(out_shape, dtype=out_array.dtype) + channel_defaults[i]
                )
            else:
                # Use a recurrent call to compute the channel and store it
                out_array[:, :, i] = _rasterize(
                    x[i], img_shape, tile_shape, tile_spacing, scale, output
                )

        return out_array

    H, W = img_shape
    Hs, Ws = tile_spacing

    dt = x.dtype
    if output:
        dt = "uint8"

    out_array = np.zeros(out_shape, dtype=dt)

    # For every row of tiles
    for tile_row in range(tile_shape[0]):
        # For every column of tiles
        for tile_col in range(tile_shape[1]):
            # Checks if belongs to an specific row
            if tile_row * tile_shape[1] + tile_col < x.shape[0]:
                x1 = x[tile_row * tile_shape[1] + tile_col]

                if scale:
                    img = scl.unitary_scale(x1.reshape(img_shape))
                else:
                    img = x1.reshape(img_shape)

                c = 1
                if output:
                    c = 255

                out_array[
                    tile_row * (H + Hs) : tile_row * (H + Hs) + H,
                    tile_col * (W + Ws) : tile_col * (W + Ws) + W,
                ] = (
                    img * c
                )

    return out_array


def create_mosaic(tensor: torch.Tensor) -> None:
    """Creates a mosaic from a tensor using Pillow.

    Args:
        tensor: An input tensor to have its mosaic created.

    """

    array = tensor.detach().numpy()

    d = int(np.sqrt(array.shape[0]))
    s = int(np.sqrt(array.shape[1]))

    img = Image.fromarray(
        _rasterize(array.T, img_shape=(d, d), tile_shape=(s, s), tile_spacing=(1, 1))
    )
    img.show()


def create_rgb_mosaic(tensor: torch.Tensor, n_samples: Optional[int] = 1) -> None:
    """Creates a squared mosaic for RGB images.

    Args:
        tensor: An input tensor to have its mosaic created.
        n_samples: The amount of samples to be plotted (width or height).

    """

    array = tensor.detach().permute(0, 2, 3, 1).numpy()

    for i in range(n_samples * n_samples):
        plt.subplot(n_samples, n_samples, 1 + i)
        plt.axis("off")
        plt.imshow(array[i])

    plt.show()
