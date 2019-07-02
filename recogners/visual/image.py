import numpy as np

import recogners.math.scale as scale


def rasterize(x, img_shape, tile_shape, tile_spacing=(0, 0), scale=True, output=True):
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
                       
    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(x, tuple):
        assert len(x) == 4
        # Create an output numpy ndarray to store the image
        if output:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=x.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if x[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = rasterize(
                    x[i], img_shape, tile_shape, tile_spacing,
                    scale, output)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = x.dtype
        if output:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < x.shape[0]:
                    this_x = x[tile_row * tile_shape[1] + tile_col]
                    if scale:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale.unitary(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array
