import pytest
import torch

from recogners.visual import plot


def test_show_tensor():
    t = torch.zeros(28, 28)

    plot.show_tensor(t)
