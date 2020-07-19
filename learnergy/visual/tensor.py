"""Tensor-related visualization.
"""

import matplotlib.pyplot as plt

import learnergy.utils.logging as l

logger = l.get_logger(__name__)


def show_tensor(tensor):
    """Plots a tensor in grayscale mode using Matplotlib.

    Args:
        tensor (Tensor): An input tensor to be plotted.

    """

    logger.debug('Showing tensor ...')

    # Creates a matplotlib figure
    plt.figure()

    # Checks if tensor has 3 channels
    if tensor.size(0) == 3:
        # If yes, permutes the tensor
        tensor = tensor.permute(1, 2, 0)

        # Plots without a color map
        plt.imshow(tensor.cpu().detach().numpy())

    # If the tensor is grayscale
    else:
        # Plots the numpy version of the tensor (grayscale)
        plt.imshow(tensor.cpu().detach().numpy(), cmap=plt.cm.get_cmap('gray'))

    # Disables all axis' ticks
    plt.xticks([])
    plt.yticks([])

    # Shows the plot
    plt.show()

    logger.debug('Tensor showed.')
