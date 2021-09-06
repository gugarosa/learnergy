"""Convergence-related visualization.
"""

import matplotlib.pyplot as plt
import numpy as np

import learnergy.utils.exception as e


def plot(*args, labels=None, title='', subtitle='', xlabel='epoch', ylabel='value', grid=True, legend=True):
    """Plots the convergence graph of desired variables.

    Essentially, each variable is a list or numpy array
    with size equals to (epochs x 1).

    Args:
        labels (list): Labels to be applied for each plot in legend.
        title (str): The title of the plot.
        subtitle (str): The subtitle of the plot.
        xlabel (str): The `x` axis label.
        ylabel (str): The `y` axis label.
        grid (bool): If grid should be used or not.
        legend (bool): If legend should be displayed or not.

    """

    # Gathering the amount of possible ticks
    ticks = np.arange(1, len(args[0]) + 1)

    # Creating figure and axis subplots
    _, ax = plt.subplots(figsize=(7, 5))

    # Defining some properties, such as axis labels, ticks and limits
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_xticks(ticks)
    ax.set_xlim(xmin=1, xmax=ticks[-1])

    # Setting both title and subtitles
    ax.set_title(title, loc='left', fontsize=14)
    ax.set_title(subtitle, loc='right', fontsize=8, color='grey')

    if grid:
        ax.grid()

    if labels:
        if not isinstance(labels, list):
            raise e.TypeError('`labels` should be a list')

        if len(labels) != len(args):
            raise e.SizeError('`args` and `labels` should have the same size')

    else:
        labels = [f'variable_{i}' for i in range(len(args))]

    # Plotting the axis
    for (arg, label) in zip(args, labels):
        ax.plot(ticks, arg, label=label)

    if legend:
        ax.legend()

    # Displaying the plot
    plt.show()
