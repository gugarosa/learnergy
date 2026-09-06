# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import numpy as np
from torch.utils.data import DataLoader

from learnergy.core import Dataset

data = np.asarray([[1, 2], [2, 4]])
targets = np.asarray([1, 2])

dataset = Dataset(data, targets)

batches = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

for samples, labels in batches:
    print(samples, labels)
