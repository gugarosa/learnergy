import torchvision

import learnergy.visual.tensor as t
from learnergy.models.rbm import RBM

# Creating training and testing dataset
train = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
test = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

# Creating an RBM
model = RBM(n_visible=784, n_hidden=128, steps=1, learning_rate=0.1,
            momentum=0, decay=0, temperature=1, use_gpu=True)

# Training an RBM
model.fit(train, epochs=1)

# Reconstructing test set
_, v = model.reconstruct(test)

# Showing a reconstructed sample
t.show_tensor(v[0].view(28, 28))
