import torch
import torchvision

from learnergy.models.e_dropout_rbm import EDropoutRBM

# Creating training and testing dataset
train = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
test = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

# Creating an EDropoutRBM
model = EDropoutRBM(n_visible=784, n_hidden=128, steps=1, learning_rate=0.1,
                    momentum=0, decay=0, temperature=1, use_gpu=True)

# Training an EDropoutRBM
mse, pl = model.fit(train, batch_size=128, epochs=5)

# Reconstructing test set
rec_mse, v = model.reconstruct(test, batch_size=10000)

# Saving model
torch.save(model, 'model.pth')

# Checking the model's history
print(model.history)
