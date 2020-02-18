import torch
import torchvision

# Creating testing dataset
test = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

# Loading pre-trained model
model = torch.load('model.pth')

# Reconstructing test set
rec_mse, v = model.reconstruct(test)
