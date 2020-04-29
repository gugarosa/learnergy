import numpy as np
from torch.utils.data import DataLoader

from learnergy.core import Dataset

# Declaring samples
data = np.asarray([[1, 2], [2, 4]])

# Declaring labels
targets = np.asarray([1, 2])

# Creating the dataset object
dataset = Dataset(data, targets)

# Creating PyTorch's batches
batches = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

# For every batch in the generator
for samples, labels in batches:
    # Check if its correct
    print(samples, labels)
