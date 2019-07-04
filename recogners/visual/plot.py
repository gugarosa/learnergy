import matplotlib.pyplot as plt

import recogners.utils.logging as l

logger = l.get_logger(__name__)


def show_tensor(tensor):
    """Plots a tensor in grayscale mode using Matplotlib.

    Args:
        tensor (tensor): An input tensor to be plotted.

    """

    logger.debug(f'Showing tensor ...')

    # Creates a matplotlib figure
    plt.figure()

    # Plots the numpy version of the tensor (grayscale)
    plt.imshow(tensor.numpy(), cmap=plt.cm.gray)

    # Disables all axis' ticks
    plt.xticks([])
    plt.yticks([])

    # Shows the plot
    plt.show()

    logger.debug('Tensor showed.')
