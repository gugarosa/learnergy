"""Tensor visualization."""

import matplotlib.pyplot as plt
import torch


def _show(tensor: torch.Tensor) -> None:
    image = tensor.permute(1, 2, 0) if tensor.size(0) == 3 else tensor
    plt.imshow(
        image.detach().cpu().numpy(),
        cmap=None if tensor.size(0) == 3 else "gray",
    )
    plt.xticks([])
    plt.yticks([])


def save_tensor(tensor: torch.Tensor, output_path: str) -> None:
    """Save a tensor as an image."""

    plt.figure()
    _show(tensor)
    plt.savefig(output_path)
    plt.close()


def show_tensor(tensor: torch.Tensor) -> None:
    """Display a tensor as an image."""

    plt.figure()
    _show(tensor)
    plt.show()
    plt.close()
