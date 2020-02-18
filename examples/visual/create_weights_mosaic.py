import torchvision

import learnergy.visual.image as im
from learnergy.models.rbm import RBM

# Creating training dataset
train = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())

# Creating an RBM
model = RBM(n_visible=784, n_hidden=128, steps=1, learning_rate=0.1,
            momentum=0, decay=0, temperature=1, use_gpu=True)

# Training an RBM
model.fit(train, epochs=5)

# Creating weights' mosaic
im.create_mosaic(model.W)
