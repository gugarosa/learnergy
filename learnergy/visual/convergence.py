# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Plot model history curves."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

import learnergy.utils.exception as e


def plot(
    *args: Sequence[float] | np.ndarray,
    labels: list[str] | None = None,
    title: str = "",
    subtitle: str = "",
    xlabel: str = "epoch",
    ylabel: str = "value",
    grid: bool = True,
    legend: bool = True,
) -> None:
    """Plot one or more epoch-wise metric sequences.

    This creates and displays a Matplotlib figure without explicitly closing it.

    Args:
        *args: Nonempty metric sequences with a common epoch count.
        labels: Optional legend labels aligned with the metric sequences.
        title: Title displayed on the left side of the plot.
        subtitle: Subtitle displayed on the right side of the plot.
        xlabel: Label for the epoch axis.
        ylabel: Label for the metric axis.
        grid: Whether to display the plot grid.
        legend: Whether to display the metric legend.

    Raises:
        learnergy.utils.exception.SizeError: Supplied labels do not match the number of metric sequences.

    """

    ticks = np.arange(1, len(args[0]) + 1)
    _, ax = plt.subplots(figsize=(7, 5))
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_xticks(ticks)
    ax.set_xlim(xmin=1, xmax=ticks[-1])
    ax.set_title(title, loc="left", fontsize=14)
    ax.set_title(subtitle, loc="right", fontsize=8, color="grey")

    if grid:
        ax.grid()

    if labels and len(labels) != len(args):
        raise e.SizeError("`labels` should have the same size as `args`.")

    labels = labels or [f"variable_{i}" for i in range(len(args))]

    for arg, label in zip(args, labels):
        ax.plot(ticks, arg, label=label)

    if legend:
        ax.legend()

    plt.show()
