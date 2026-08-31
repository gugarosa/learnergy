"""Convergence plotting."""

import matplotlib.pyplot as plt
import numpy as np

import learnergy.utils.exception as e


def plot(
    *args,
    labels: list[str] | None = None,
    title: str = "",
    subtitle: str = "",
    xlabel: str = "epoch",
    ylabel: str = "value",
    grid: bool = True,
    legend: bool = True,
) -> None:
    """Plot one or more metric sequences."""

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
        raise e.SizeError("`args` and `labels` should have the same size")
    labels = labels or [f"variable_{i}" for i in range(len(args))]

    for arg, label in zip(args, labels):
        ax.plot(ticks, arg, label=label)

    if legend:
        ax.legend()
    plt.show()
