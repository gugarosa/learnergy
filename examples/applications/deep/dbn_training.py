# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import torch
import torchvision

from learnergy.models.deep import DBN

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

model = DBN(
    model="bernoulli",
    n_visible=784,
    n_hidden=(128, 256, 128),
    steps=(1, 1, 1),
    learning_rate=(0.1, 0.1, 0.1),
    momentum=(0, 0, 0),
    decay=(0, 0, 0),
    temperature=(1, 1, 1),
    use_gpu=True,
)

model.fit(train, batch_size=128, epochs=(3, 3, 3))

rec_mse, v = model.reconstruct(test)

torch.save(model, "model.pth")

for m in model.models:
    print(m.history)
