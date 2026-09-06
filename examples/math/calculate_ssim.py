# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import torch

import learnergy.math.metrics as m

v = torch.normal(0, 1, size=(10, 784))
x = torch.normal(0, 1, size=(10, 28, 28))

ssim = m.calculate_ssim(v, x)

print(ssim)
