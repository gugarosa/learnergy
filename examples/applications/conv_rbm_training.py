import torch
import torchvision

from learnergy.models import ConvRBM

# Creating training and testing dataset
train = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
test = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

# Creating a ConvRBM
model = ConvRBM(visible_shape=(28, 28), filter_shape=(7, 7), n_filters=5, learning_rate=0.1)

# Training a ConvRBM
mse, pl = model.fit(train, batch_size=128, epochs=5)