# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import numpy as np

import learnergy.math.scale as s

a = np.array([1, 2, 3, 4, 5])

u = s.unitary_scale(a)

print(u)
