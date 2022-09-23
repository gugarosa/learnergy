"""Tensor-related visualization.
"""

import matplotlib.pyplot as plt
import torch


def save_tensor(tensor: torch.Tensor, output_path: str) -> None:
    """Saves a tensor in grayscale mode using Matplotlib.

    Args:
        tensor: An input tensor to be saved.
        output_path: An outputh path to save the tensor.

    """

    plt.figure()

    if tensor.size(0) == 3:
        tensor = tensor.permute(1, 2, 0)
        plt.imshow(tensor.cpu().detach().numpy())
    else:
        plt.imshow(tensor.cpu().detach().numpy(), cmap=plt.cm.get_cmap("gray"))

    plt.xticks([])
    plt.yticks([])

    plt.savefig(output_path)


def show_tensor(tensor: torch.Tensor) -> None:
    """Plots a tensor in grayscale mode using Matplotlib.

    Args:
        tensor: An input tensor to be plotted.

    """

    plt.figure()

    if tensor.size(0) == 3:
        tensor = tensor.permute(1, 2, 0)
        plt.imshow(tensor.cpu().detach().numpy())
    else:
        plt.imshow(tensor.cpu().detach().numpy(), cmap=plt.cm.get_cmap("gray"))

    plt.xticks([])
    plt.yticks([])

    plt.show()
