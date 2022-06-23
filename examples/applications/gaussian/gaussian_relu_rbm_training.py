import torch
import torchvision

from learnergy.models.gaussian import GaussianReluRBM

# Creating training and testing dataset
train = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
)
test = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
)

# Creating a GaussianReluRBM
model = GaussianReluRBM(
    n_visible=784,
    n_hidden=256,
    steps=1,
    learning_rate=0.001,
    momentum=0.9,
    decay=0,
    temperature=1,
    use_gpu=False,
)

# Training a GaussianReluRBM
mse, pl = model.fit(train, batch_size=128, epochs=5)

# Reconstructing test set
rec_mse, v = model.reconstruct(test)

# Saving model
torch.save(model, "model.pth")

# Checking the model's history
print(model.history)
