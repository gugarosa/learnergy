import torchvision
from torch.utils.data import DataLoader

import recogners.visual.image as im
from recogners.models.rbm import RBM

# Creating training and testing dataset
train = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
test = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

# Creating training and testing batches
train_batches = DataLoader(train, batch_size=128, shuffle=True, num_workers=1)
test_batches = DataLoader(test, batch_size=10000, shuffle=True, num_workers=1)

# Creating an RBM
model = RBM(n_visible=784, n_hidden=128, steps=1,
            learning_rate=0.1, momentum=0, decay=0, temperature=1)

# Training an RBM
model.fit(train_batches, epochs=1)

# Reconstructing test set
v = model.reconstruct(test_batches)

# Showing a reconstructed sample
im.show(v[0].view(28, 28))
