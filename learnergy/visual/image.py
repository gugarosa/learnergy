import numpy as np
from PIL import Image

import learnergy.math.scale as s
import learnergy.utils.logging as l

logger = l.get_logger(__name__)


def _rasterize(x, img_shape, tile_shape, tile_spacing=(0, 0), scale=True, output=True):
    """Rasterizes and prepares an image to be outputted as a mosaic.

    Args:
        x (array): An input array to be rasterized.
        img_shape (tuple): A tuple for the image shape.
        tile_shape (tuple): A tuple holding the shape of each tile.
        tile_spacing (tuple): A tuple containing the spacing between tiles.
        scale (bool): If output array should be scaled between 0 and 1.
        output (bool): If output values should be returned as pixels or not.

    Returns:
        An output array containing the rasterized version of input array.

    """

    # Asserts if tuple lengths are equal to 2
    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # Creates an output shape
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp,
                 tsp in zip(img_shape, tile_shape, tile_spacing)]

    # Asserts if input array is a tuple
    if isinstance(x, tuple):
        # Checks if its length is equal to 4
        assert len(x) == 4

        # If output boolean is true
        if output:
            # Output values as pixels
            out_array = np.zeros(
                (out_shape[0], out_shape[1], 4), dtype='uint8')

            # Apply the default channels
            channel_defaults = [0, 0, 0, 255]

        # If output boolean is false
        else:
            # Output values as its input type
            out_array = np.zeros(
                (out_shape[0], out_shape[1], 4), dtype=x[0].dtype)

            # Apply the default channels
            channel_defaults = [0., 0., 0., 1.]

        # For every possible item in tuple
        for i in range(4):
            # If there is no channel
            if x[i] is None:
                # Fill it with zeros of the correct dtype
                out_array[:, :, i] = np.zeros(
                    out_shape, dtype=out_array.dtype) + channel_defaults[i]

            # If there is a channel
            else:
                # Use a recurrent call to compute the channel and store it
                out_array[:, :, i] = _rasterize(
                    x[i], img_shape, tile_shape, tile_spacing, scale, output)

        return out_array

    # Asserts if input array is not a tuple
    else:
        # Gathers the image shape
        H, W = img_shape

        # Also gathers the tile spacing
        Hs, Ws = tile_spacing

        # Checks the current input dtype
        dt = x.dtype

        # If output boolean is true
        if output:
            # Output type should be an unsigned integer
            dt = 'uint8'

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
                        img = s.unitary_scale(x1.reshape(img_shape))

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
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = img * c

        return out_array


def create_mosaic(tensor):
    """Creates a mosaic from a tensor using Pillow.

    Args:
        tensor (Tensor): An input tensor to have its mosaic created.

    """

    logger.debug(f'Creating mosaic ...')

    # Gets the numpy array from the tensor
    array = tensor.detach().numpy()

    # Calculate their maximum possible squared dimension
    d = int(np.sqrt(array.shape[0]))
    s = int(np.sqrt(array.shape[1]))

    # Creates a Pillow image from the array's rasterized version
    img = Image.fromarray(_rasterize(array.T, img_shape=(
        d, d), tile_shape=(s, s), tile_spacing=(1, 1)))

    # Shows the image
    img.show()

    logger.debug('Mosaic created.')
