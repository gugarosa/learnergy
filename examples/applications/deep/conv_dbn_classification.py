# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Fine-tune convolutional DBN representations with a supervised linear classifier.

Each Gaussian layer's normalize attribute controls batch standardization during fitting.

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

from learnergy.models.deep import ConvDBN

train = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
test = torchvision.datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)

vshape = 28
channels = 1

model = ConvDBN(
    model="gaussian",
    visible_shape=(vshape, vshape),
    filter_shape=((3, 3), (5, 5)),
    n_filters=(32, 32),
    steps=(1, 1),
    n_channels=channels,
    learning_rate=(0.00001, 0.000001),
    momentum=(0.9, 0.9),
    decay=(0, 0),
    maxpooling=(False, True),
    use_gpu=True,
)

batch_size = 128
n_classes = 10
fine_tune_epochs = 20
epochs = (20, 20)

model.fit(train, batch_size=batch_size, epochs=epochs)

torch.save(model, "model.pth")

h1 = model.models[len(model.models) - 1].hidden_shape[0]
h2 = model.models[len(model.models) - 1].hidden_shape[1]
nf = model.models[len(model.models) - 1].n_filters

if model.models[len(model.models) - 1].maxpooling:
    input_fc = nf * (h1 // 2 + 1) * (h2 // 2 + 1)
    print("Pooling:", input_fc)
else:
    input_fc = nf * h1 * h2
fc = nn.Linear(input_fc, n_classes).to(model.device)

criterion = nn.CrossEntropyLoss()

optimizer = [optim.Adam(m.parameters(), lr=0.00001) for m in model.models]
optimizer.append(optim.Adam(fc.parameters(), lr=0.001))

train_batch = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
val_batch = DataLoader(test, batch_size=256, shuffle=False, num_workers=0)

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

    print(f"Loss: {train_loss / len(train_batch)} | Val Accuracy: {val_acc/len(val_batch)}")

torch.save(model, "tuned_model.pth")
