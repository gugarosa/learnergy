import torch

from learnergy.math import metrics


def test_calculate_ssim():
    v = torch.normal(0, 1, size=(10, 784))
    x = torch.normal(0, 1, size=(10, 28, 28))

    ssim = metrics.calculate_ssim(v, x)

    assert ssim != 0
