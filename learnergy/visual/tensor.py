import matplotlib.pyplot as plt

import learnergy.utils.logging as l

logger = l.get_logger(__name__)


def show_tensor(tensor):
    """Plots a tensor in grayscale mode using Matplotlib.

    Args:
        tensor (Tensor): An input tensor to be plotted.

    """

    logger.debug(f'Showing tensor ...')

    # Creates a matplotlib figure
    plt.figure()
    if tensor.size(0)==3:
        tensor = tensor.permute(1, 2, 0)
        plt.imshow(tensor.cpu().detach().numpy())
    else:
        # Plots the numpy version of the tensor (grayscale)
        plt.imshow(tensor.cpu().detach().numpy(), cmap=plt.cm.gray)

    # Disables all axis' ticks
    plt.xticks([])
    plt.yticks([])

    # Shows the plot
    plt.show()

    logger.debug('Tensor showed.')
