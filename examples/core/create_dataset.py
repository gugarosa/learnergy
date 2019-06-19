import numpy as np
from torch.utils import data

from recogners.core.dataset import Dataset

# Declaring samples
X = np.asarray([[1, 2], [2, 4]])

# Declaring labels
Y = np.asarray([1, 2])

# Creating dataset
d = Dataset(X, Y)

#
g = data.DataLoader(d, batch_size=2, shuffle=True, num_workers=1)


for input_batch, label_batch in g:
    print(input_batch, label_batch)
