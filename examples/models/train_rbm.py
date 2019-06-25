from torch.utils.data import DataLoader

from recogners.datasets.opf import OPFDataset
from recogners.models.rbm import RBM

# Creating training dataset
train = OPFDataset(path='data/boat.txt')

# Creating training batches
train_batch = DataLoader(train, batch_size=16, shuffle=True, num_workers=1)

# Creating an RBM
r = RBM(n_visible=2, n_hidden=128, learning_rate=0.1, steps=1, temperature=1)

# Training an RBM
r.fit(train_batch, epochs=100)
