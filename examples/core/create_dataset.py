import numpy as np
from torch.utils.data import DataLoader

from recogners.core.dataset import Dataset

# Declaring samples
X = np.asarray([[1, 2], [2, 4]])

# Declaring labels
Y = np.asarray([1, 2])

# Creating the dataset object
d = Dataset(X, Y)

# Creating a PyTorch's generator
g = DataLoader(d, batch_size=1, shuffle=True, num_workers=1)

# For every batch in the generator
for samples, labels in g:
    # Check if its correct
    print(samples, labels)
