# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

from learnergy.models.bernoulli import RBM

batch_size = 128
n_classes = 10
fine_tune_epochs = 20

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

model.fit(train, batch_size=batch_size, epochs=1)

fc = nn.Linear(model.n_hidden, n_classes).to(model.device)

criterion = nn.CrossEntropyLoss()

optimizer = [
    optim.Adam(model.parameters(), lr=0.001),
    optim.Adam(fc.parameters(), lr=0.001),
]

train_batch = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=1)
val_batch = DataLoader(test, batch_size=10000, shuffle=False, num_workers=1)

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

print(model.history)
