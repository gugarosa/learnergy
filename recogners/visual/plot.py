import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import recogners.visual.image as im


def show(tensor):
    """Plots a tensor in grayscale mode using Matplotlib.

    Args:
        tensor (tensor): An input tensor to be plotted.

    """

    # Creates a matplotlib figure
    plt.figure()

    # Plots the numpy version of the tensor (grayscale)
    plt.imshow(tensor.numpy(), cmap=plt.cm.gray)

    # Disable all axis' ticks
    plt.xticks([])
    plt.yticks([])

    # Shows the plot
    plt.show()


def create_mosaic(tensor):
    """Creates a mosaic from a tensor using Pillow.

    Args:
        tensor (tensor): An input tensor to have its mosaic created.

    """

    # Gets the numpy array from the tensor
    array = tensor.numpy()

    # Calculate their maximum possible squared dimension
    d = int(np.sqrt(array.shape[0]))
    s = int(np.sqrt(array.shape[1]))

    # Creates a Pillow image from the array's rasterized version
    img = Image.fromarray(im.rasterize(array.T, img_shape=(
        d, d), tile_shape=(s, s), tile_spacing=(1, 1)))

    # Shows the image
    img.show()
