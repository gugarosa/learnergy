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

    # Asserts if tuple lengths are equal to 2
    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # Creates an output shape
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    # Asserts if input array is a tuple
    if isinstance(x, tuple):
        # Checks if its length is equal to 4
        assert len(x) == 4

        # If output boolean is true
        if output:
            # Output values as pixels
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype="uint8")

            # Apply the default channels
            channel_defaults = [0, 0, 0, 255]

        # If output boolean is false
        else:
            # Output values as its input type
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=x[0].dtype)

            # Apply the default channels
            channel_defaults = [0.0, 0.0, 0.0, 1.0]

        # For every possible item in tuple
        for i in range(4):
            # If there is no channel
            if x[i] is None:
                # Fill it with zeros of the correct dtype
                out_array[:, :, i] = (
                    np.zeros(out_shape, dtype=out_array.dtype) + channel_defaults[i]
                )

            # If there is a channel
            else:
                # Use a recurrent call to compute the channel and store it
                out_array[:, :, i] = _rasterize(
                    x[i], img_shape, tile_shape, tile_spacing, scale, output
                )

        return out_array

    # Gathers the image shape and its tile spacing
    H, W = img_shape
    Hs, Ws = tile_spacing

    # Checks the current input dtype
    dt = x.dtype

    # If output boolean is true
    if output:
        # Output type should be an unsigned integer
        dt = "uint8"

    # Creates a zeros array based on output shape
    out_array = np.zeros(out_shape, dtype=dt)

    # For every row of tiles
    for tile_row in range(tile_shape[0]):
        # For every column of tiles
        for tile_col in range(tile_shape[1]):
            # Checks if belongs to an specific row
            if tile_row * tile_shape[1] + tile_col < x.shape[0]:
                # Replace its value
                x1 = x[tile_row * tile_shape[1] + tile_col]

                # If scale boolean is true
                if scale:
                    # We should scale values to be between 0 and 1
                    img = scl.unitary_scale(x1.reshape(img_shape))

                # If not
                else:
                    # We just reshape the image to the input shape
                    img = x1.reshape(img_shape)

                # Add the slice to the corresponding position in the output array
                c = 1

                # If output boolean is true
                if output:
                    # The slice should be a maximum pixel value
                    c = 255

                # Creates the output array
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

    # Gets the numpy array from the tensor
    array = tensor.detach().numpy()

    # Calculate their maximum possible squared dimension
    d = int(np.sqrt(array.shape[0]))
    s = int(np.sqrt(array.shape[1]))

    # Creates a Pillow image from the array's rasterized version
    img = Image.fromarray(
        _rasterize(array.T, img_shape=(d, d), tile_shape=(s, s), tile_spacing=(1, 1))
    )

    # Shows the image
    img.show()


def create_rgb_mosaic(tensor: torch.Tensor, n_samples: Optional[int] = 1) -> None:
    """Creates a squared mosaic for RGB images.

    Args:
        tensor: An input tensor to have its mosaic created.
        n_samples: The amount of samples to be plotted (width or height).

    """

    # Permutes the tensor and transforms into numpy-based array
    array = tensor.detach().permute(0, 2, 3, 1).numpy()

    # plot images from the dataset
    for i in range(n_samples * n_samples):
        # Creates the subplots
        plt.subplot(n_samples, n_samples, 1 + i)

        # Removes the axis
        plt.axis("off")

        # Plots the raw data
        plt.imshow(array[i])

    # Shows the plot
    plt.show()
