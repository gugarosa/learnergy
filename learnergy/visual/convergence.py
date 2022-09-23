"""Convergence-related visualization.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

import learnergy.utils.exception as e


def plot(
    *args,
    labels: Optional[List[str]] = None,
    title: Optional[str] = "",
    subtitle: Optional[str] = "",
    xlabel: Optional[str] = "epoch",
    ylabel: Optional[str] = "value",
    grid: Optional[bool] = True,
    legend: Optional[bool] = True,
) -> None:
    """Plots the convergence graph of desired variables.

    Essentially, each variable is a list or numpy array
    with size equals to (epochs x 1).

    Args:
        labels: Labels to be applied for each plot in legend.
        title: The title of the plot.
        subtitle: The subtitle of the plot.
        xlabel: The `x` axis label.
        ylabel: The `y` axis label.
        grid: If grid should be used or not.
        legend: If legend should be displayed or not.

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

    if labels:
        if len(labels) != len(args):
            raise e.SizeError("`args` and `labels` should have the same size")
    else:
        labels = [f"variable_{i}" for i in range(len(args))]

    for (arg, label) in zip(args, labels):
        ax.plot(ticks, arg, label=label)

    if legend:
        ax.legend()

    plt.show()
