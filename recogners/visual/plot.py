import matplotlib.pyplot as plt
import numpy as np
import recogners.utils.wv as wv
import recogners.math.scale as scale
from PIL import Image

def show(tensor):
    """Plots a tensor in grayscale mode using Matplotlib.

    Args:
        tensor (tensor): An input tensor to be plotted.
    """

    #
    plt.figure()

    #
    plt.imshow(tensor.numpy(), cmap=plt.cm.gray)

    #
    plt.xticks([])
    plt.yticks([])

    #
    plt.show()

def create_mosaic(tensor):
    """
    """

    #
    t = tensor.numpy()

    #
    d = int(np.sqrt(t.shape[0]))
    s = int(np.sqrt(t.shape[1]))

    #
    img = Image.fromarray(wv.tile_raster_images(X = w8.T, img_shape=(d, d), tile_shape=(s,s), tile_spacing=(1,1)))

    #
    img.show()