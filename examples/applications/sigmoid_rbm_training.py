import torch
import torchvision
from torch.utils.data import DataLoader

from learnergy.models.sigmoid_rbm import SigmoidRBM

# Creating training and testing dataset
train = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
test = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

# Creating training and testing batches
train_batches = DataLoader(train, batch_size=128, shuffle=True, num_workers=1)
test_batches = DataLoader(test, batch_size=10000, shuffle=True, num_workers=1)

# Creating a SigmoidRBM
model = SigmoidRBM(n_visible=784, n_hidden=128, steps=1, learning_rate=0.1,
                   momentum=0, decay=0, temperature=1, use_gpu=True)

# Training a SigmoidRBM
mse, pl = model.fit(train_batches, epochs=5)

# Reconstructing test set
rec_mse, v = model.reconstruct(test_batches)

# Saving model
torch.save(model, 'model.pth')

# Checking the model's history
print(model.history)
