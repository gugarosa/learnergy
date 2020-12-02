import torch
import torchvision

from learnergy.models.bernoulli import ConvRBM

# Creating training and testing dataset
train = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
test = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

# Creating a ConvRBM
model = ConvRBM(visible_shape=(28, 28), filter_shape=(7, 7), n_filters=10, n_channels=1,
                steps=1, learning_rate=0.01, momentum=0, decay=0, use_gpu=True)

# Training a ConvRBM
mse = model.fit(train, batch_size=128, epochs=5)

# Reconstructing test set
_, v = model.reconstruct(test)
