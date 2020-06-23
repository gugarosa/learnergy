import torch
import torchvision

from learnergy.models.stack import DBN

# Creating training and testing dataset
train = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
test = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

# Creating a DBN
model = DBN(model='bernoulli', n_visible=784, n_hidden=[128, 256, 128], steps=[1, 1, 1],
            learning_rate=[0.1, 0.1, 0.1], momentum=[0, 0, 0], decay=[0, 0, 0], temperature=[1, 1, 1],
            use_gpu=True)

# Training a DBN
model.fit(train, batch_size=128, epochs=[3, 3, 3])

# Reconstructing test set
rec_mse, v = model.reconstruct(test)

# Saving model
torch.save(model, 'model.pth')

# Checking the model's history
for m in model.models:
    print(m.history)
