import torch
import torchvision
from torch.utils.data import DataLoader

from learnergy.models.gaussian_rbm import GaussianRBM

# Creating training and testing dataset
train = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ]))
test = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ]))

# Creating training and testing batches
train_batches = DataLoader(train, batch_size=128, shuffle=True, num_workers=1)
test_batches = DataLoader(test, batch_size=10000, shuffle=True, num_workers=1)

# Creating a GaussianRBM
model = GaussianRBM(n_visible=784, n_hidden=128, steps=1, learning_rate=0.0005,
                    momentum=0, decay=0, temperature=1, use_gpu=True)

# Training a GaussianRBM
mse, pl = model.fit(train_batches, epochs=50)

# Reconstructing test set
rec_mse, v = model.reconstruct(test_batches)

# Saving model
torch.save(model, 'model.pth')

# Checking the model's history
print(model.history)
