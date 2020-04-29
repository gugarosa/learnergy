import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

from learnergy.models import RBM

# Defining some input variables
batch_size = 128
n_classes = 10
fine_tune_epochs = 20

# Creating training and validation/testing dataset
train = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
test = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

# Creating an RBM
model = RBM(n_visible=784, n_hidden=128, steps=1, learning_rate=0.1,
            momentum=0, decay=0, temperature=1, use_gpu=True)

# Training an RBM
model.fit(train, batch_size=batch_size, epochs=5)

# Creating the Fully Connected layer to append on top of DBNs
fc = torch.nn.Linear(model.n_hidden, n_classes)

# Check if model uses GPU
if model.device == 'cuda':
    # If yes, put fully-connected on GPU
    fc = fc.cuda()

# Cross-Entropy loss is used for the discriminative fine-tuning
criterion = nn.CrossEntropyLoss()

# Creating the optimzers
optimizer = [optim.Adam(model.parameters(), lr=0.001),
             optim.Adam(fc.parameters(), lr=0.001)]

# Creating training and validation batches
train_batch = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=1)
val_batch = DataLoader(test, batch_size=10000, shuffle=False, num_workers=1)

# For amount of fine-tuning epochs
for e in range(fine_tune_epochs):
    print(f'Epoch {e+1}/{fine_tune_epochs}')

    # Resetting metrics
    train_loss, val_acc = 0, 0
    
    # For every possible batch
    for x_batch, y_batch in train_batch:
        # For every possible optimizer
        for opt in optimizer:
            # Resets the optimizer
            opt.zero_grad()
        
        # Flatenning the samples batch
        x_batch = x_batch.view(x_batch.size(0), model.n_visible)

        # Checking whether GPU is avaliable and if it should be used
        if model.device == 'cuda':
            # Applies the GPU usage to the data and labels
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

        # Passing the batch down the model
        y = model(x_batch)

        # Calculating the fully-connected outputs
        y = fc(y)
        
        # Calculating loss
        loss = criterion(y, y_batch)
        
        # Propagating the loss to calculate the gradients
        loss.backward()
        
        # For every possible optimizer
        for opt in optimizer:
            # Performs the gradient update
            opt.step()

        # Adding current batch loss
        train_loss += loss.item()
        
    # Calculate the test accuracy for the model:
    for x_batch, y_batch in val_batch:
        # Flatenning the testing samples batch
        x_batch = x_batch.view(x_batch.size(0), model.n_visible)

        # Checking whether GPU is avaliable and if it should be used
        if model.device == 'cuda':
            # Applies the GPU usage to the data and labels
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

        # Passing the batch down the model
        y = model(x_batch)

        # Calculating the fully-connected outputs
        y = fc(y)

        # Calculating predictions
        _, preds = torch.max(y, 1)

        # Calculating validation set accuracy
        val_acc = torch.mean((torch.sum(preds == y_batch).float()) / x_batch.size(0))

    print(f'Loss: {train_loss / len(train_batch)} | Val Accuracy: {val_acc}')

# Saving the fine-tuned model
torch.save(model, 'tuned_model.pth')

# Checking the model's history
print(model.history)
