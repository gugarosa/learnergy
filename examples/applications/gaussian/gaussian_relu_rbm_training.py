# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import torch
import torchvision

from learnergy.models.gaussian import GaussianReluRBM

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

mse, pl = model.fit(train, batch_size=128, epochs=5)

rec_mse, v = model.reconstruct(test)

torch.save(model, "model.pth")

print(model.history)
