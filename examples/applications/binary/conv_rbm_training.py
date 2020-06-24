import torch
import torchvision

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from learnergy.models.binary import ConvRBM
import learnergy.visual.tensor as t

# Creating training and testing dataset
train = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
test = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

# Creating a ConvRBM
model = ConvRBM(visible_shape=(28, 28), filter_shape=(7, 7), n_filters=5, learning_rate=0.1)

# Training a ConvRBM
mse, pl = model.fit(train, batch_size=128, epochs=3)

# Reconstructing test set
_, v = model.reconstruct(test)

# Showing a reconstructed sample
t.show_tensor(v[0].reshape(28, 28))