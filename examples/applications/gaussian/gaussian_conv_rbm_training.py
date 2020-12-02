import torch
import torchvision

from learnergy.models.gaussian import GaussianConvRBM

# Creating training and testing dataset
train = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
test = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

# Creating a GaussianConvRBM
model = GaussianConvRBM(visible_shape=(32, 32), filter_shape=(9, 9), n_filters=16, n_channels=3,
                		steps=1, learning_rate=0.00001, momentum=0.5, decay=0, use_gpu=True)

# Training a GaussianConvRBM
mse = model.fit(train, batch_size=100, epochs=5)

# Reconstructing test set
_, v = model.reconstruct(test)
