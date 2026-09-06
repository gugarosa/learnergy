# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import torchvision

from learnergy.models.bernoulli import ConvRBM

train = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
test = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)

model = ConvRBM(
    visible_shape=(28, 28),
    filter_shape=(7, 7),
    n_filters=10,
    n_channels=1,
    steps=1,
    learning_rate=0.01,
    momentum=0,
    decay=0,
    use_gpu=True,
)

mse = model.fit(train, batch_size=128, epochs=5)

_, v = model.reconstruct(test)
