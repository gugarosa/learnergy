"""Tensor-related visualization.
"""

import matplotlib.pyplot as plt


def save_tensor(tensor, output_path):
    """Saves a tensor in grayscale mode using Matplotlib.

    Args:
        tensor (Tensor): An input tensor to be saved.
        output_path (str): An outputh path to save the tensor.

    """

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
    plt.savefig(output_path)


def show_tensor(tensor):
    """Plots a tensor in grayscale mode using Matplotlib.

    Args:
        tensor (Tensor): An input tensor to be plotted.

    """

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
