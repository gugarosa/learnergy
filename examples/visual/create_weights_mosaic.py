import torchvision
from torch.utils.data import DataLoader

import learnergy.visual.image as im
from learnergy.models.rbm import RBM

# Creating training dataset
train = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())

# Creating training batches
train_batches = DataLoader(train, batch_size=128, shuffle=True, num_workers=1)

# Creating an RBM
model = RBM(n_visible=784, n_hidden=128, steps=1,
            learning_rate=0.1, momentum=0, decay=0, temperature=1)

# Training an RBM
model.fit(train_batches, epochs=1)

# Creating weights' mosaic
im.create_mosaic(model.W)
