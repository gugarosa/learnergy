# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import torch
import torchvision

from learnergy.models.deep import ConvDBN

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

model = ConvDBN(
    model="bernoulli",
    visible_shape=(28, 28),
    filter_shape=((2, 2), (2, 2)),
    n_filters=(4, 4),
    steps=(1, 1),
    n_channels=1,
    learning_rate=(0.1, 0.1),
    momentum=(0, 0),
    decay=(0, 0),
    maxpooling=True,
    use_gpu=True,
)

model.fit(train, batch_size=256, epochs=(1, 1))

rec_mse, v = model.reconstruct(test)

torch.save(model, "model.pth")

for m in model.models:
    print(m.history)
