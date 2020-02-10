import numpy as np

from learnergy.core.dataset import Dataset
from torch.utils.data import DataLoader

# Declaring samples
X = np.asarray([[1, 2], [2, 4]])

# Declaring labels
Y = np.asarray([1, 2])

# Creating the dataset object
dataset = Dataset(X, Y)

# Creating PyTorch's batches
batches = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

# For every batch in the generator
for samples, labels in batches:
    # Check if its correct
    print(samples, labels)
