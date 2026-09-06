# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Render image tensors with Matplotlib."""

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
    """Save a grayscale or channel-first RGB tensor as an image.

    The new Matplotlib figure is closed even if rendering or writing fails.
    Tensor data is detached and moved to CPU without modifying the input.

    Args:
        tensor: Image shaped (height, width), (1, height, width), or (3, height, width).
        output_path: Destination image path interpreted by Matplotlib.

    Raises:
        TypeError: The tensor layout cannot be rendered as an image.
        OSError: The destination image cannot be written.

    """

    figure = plt.figure()
    try:
        _show(tensor)
        figure.savefig(output_path)
    finally:
        plt.close(figure)


def show_tensor(tensor: torch.Tensor) -> None:
    """Display a grayscale or channel-first RGB tensor.

    The new Matplotlib figure is closed when display returns or fails.
    Tensor data is detached and moved to CPU without modifying the input.

    Args:
        tensor: Image shaped (height, width), (1, height, width), or (3, height, width).

    Raises:
        TypeError: The tensor layout cannot be rendered as an image.

    """

    figure = plt.figure()
    try:
        _show(tensor)
        plt.show()
    finally:
        plt.close(figure)
