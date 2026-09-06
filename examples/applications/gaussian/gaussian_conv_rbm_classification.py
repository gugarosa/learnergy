# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

from learnergy.models.gaussian import GaussianConvRBM

v_shape = 32
n_filters = 16
f_shape = 5
n_channels = 3
batch_size = 100
n_classes = 10
fine_tune_epochs = 10

train = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
test = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)

model = GaussianConvRBM(
    visible_shape=(v_shape, v_shape),
    filter_shape=(f_shape, f_shape),
    n_filters=n_filters,
    n_channels=n_channels,
    learning_rate=0.00001,
    momentum=0.5,
    decay=0,
    maxpooling=True,
    use_gpu=True,
)

model.fit(train, batch_size=batch_size, epochs=5)

h1 = model.hidden_shape[0]
h2 = model.hidden_shape[1]
nf = model.n_filters

if model.maxpooling:
    input_fc = nf * (h1 // 2 + 1) * (h2 // 2 + 1)
else:
    input_fc = nf * h1 * h2
fc = nn.Linear(input_fc, n_classes).to(model.device)

criterion = nn.CrossEntropyLoss()

optimizer = [
    optim.Adam(model.parameters(), lr=0.0001),
    optim.Adam(fc.parameters(), lr=0.001),
]

train_batch = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=0)
val_batch = DataLoader(test, batch_size=10000, shuffle=False, num_workers=0)

for e in range(fine_tune_epochs):
    print(f"Epoch {e+1}/{fine_tune_epochs}")

    train_loss, val_acc = 0, 0

    for x_batch, y_batch in train_batch:
        for opt in optimizer:
            opt.zero_grad()

        x_batch = x_batch.to(model.device)
        y_batch = y_batch.to(model.device)

        y = model(x_batch)
        y = y.reshape(x_batch.size(0), input_fc)
        y = fc(y)

        loss = criterion(y, y_batch)

        loss.backward()

        for opt in optimizer:
            opt.step()

        train_loss += loss.item()

    for x_batch, y_batch in val_batch:
        x_batch = x_batch.to(model.device)
        y_batch = y_batch.to(model.device)

        y = model(x_batch)
        y = y.reshape(x_batch.size(0), input_fc)
        y = fc(y)

        _, preds = torch.max(y, 1)

        val_acc += torch.mean((torch.sum(preds == y_batch).float()) / x_batch.size(0))

    print(f"Loss: {train_loss / len(train_batch)} | Val Accuracy: {val_acc / len(val_batch)}")

torch.save(model, "tuned_model.pth")

print(model.history)
