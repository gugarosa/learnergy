"""Tensor visualization."""

import matplotlib.pyplot as plt
import torch


def _show(tensor: torch.Tensor) -> None:
    image = tensor
    if tensor.ndim == 3:
        image = tensor.permute(1, 2, 0) if tensor.size(0) == 3 else tensor.squeeze(0)
    plt.imshow(
        image.detach().cpu().numpy(),
        cmap=None if image.ndim == 3 else "gray",
    )
    plt.xticks([])
    plt.yticks([])


def save_tensor(tensor: torch.Tensor, output_path: str) -> None:
    """Save an (H, W), (1, H, W), or (3, H, W) image tensor."""

    figure = plt.figure()
    try:
        _show(tensor)
        figure.savefig(output_path)
    finally:
        plt.close(figure)


def show_tensor(tensor: torch.Tensor) -> None:
    """Display an (H, W), (1, H, W), or (3, H, W) image tensor."""

    figure = plt.figure()
    try:
        _show(tensor)
        plt.show()
    finally:
        plt.close(figure)
