# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import torchvision

import learnergy.visual.tensor as t
from learnergy.models.bernoulli import RBM

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

model = RBM(
    n_visible=784,
    n_hidden=128,
    steps=1,
    learning_rate=0.1,
    momentum=0,
    decay=0,
    temperature=1,
    use_gpu=True,
)

model.fit(train, epochs=1)

_, v = model.reconstruct(test)

t.show_tensor(v[0].reshape(28, 28))
