import torch

import learnergy.math.metrics as m

# Defines two random tensors
v = torch.normal(0, 1, size=(10, 784))
x = torch.normal(0, 1, size=(10, 28, 28))

# Calculates the SSIM between them
ssim = m.calculate_ssim(v, x)

# Prints the result
print(ssim)
