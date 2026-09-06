# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Fine-tune DBN representations with a supervised linear classifier.

The residual_dbn_training.py example demonstrates the corresponding residual-stack configuration.

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

from learnergy.models.deep import DBN

batch_size = 128
n_classes = 10
fine_tune_epochs = 10

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
    model=("gaussian", "sigmoid"),
    n_visible=784,
    n_hidden=(256, 256),
    steps=(1, 1),
    learning_rate=(0.0001, 0.001),
    momentum=(0, 0),
    decay=(0, 0),
    temperature=(1, 1),
    use_gpu=True,
)

model.fit(train, batch_size=batch_size, epochs=(5, 5))

fc = torch.nn.Linear(model.n_hidden[model.n_layers - 1], n_classes).to(model.device)

criterion = nn.CrossEntropyLoss()

optimizer = [optim.Adam(m.parameters(), lr=0.0001) for m in model.models]
optimizer.append(optim.Adam(fc.parameters(), lr=0.001))

train_batch = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
val_batch = DataLoader(test, batch_size=10000, shuffle=False, num_workers=0)

for e in range(fine_tune_epochs):
    print(f"Epoch {e+1}/{fine_tune_epochs}")

    train_loss, val_acc = 0, 0

    for x_batch, y_batch in train_batch:
        for opt in optimizer:
            opt.zero_grad()

        x_batch = x_batch.reshape(x_batch.size(0), model.n_visible)

        x_batch = x_batch.to(model.device)
        y_batch = y_batch.to(model.device)

        y = model(x_batch)
        y = fc(y)

        loss = criterion(y, y_batch)

        loss.backward()

        for opt in optimizer:
            opt.step()

        train_loss += loss.item()

    for x_batch, y_batch in val_batch:
        x_batch = x_batch.reshape(x_batch.size(0), model.n_visible)

        x_batch = x_batch.to(model.device)
        y_batch = y_batch.to(model.device)

        y = model(x_batch)
        y = fc(y)

        _, preds = torch.max(y, 1)

        val_acc += torch.mean((torch.sum(preds == y_batch).float()) / x_batch.size(0))

    print(f"Loss: {train_loss / len(train_batch)} | Val Accuracy: {val_acc / len(val_batch)}")

torch.save(model, "tuned_model.pth")

for m in model.models:
    print(m.history)
