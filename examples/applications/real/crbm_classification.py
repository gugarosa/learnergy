import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import learnergy.visual.tensor as t
#from learnergy.models.binary import ConvRBM
from learnergy.models.real import ConvRBM

# Creating training and testing dataset
train = torchvision.datasets.CIFAR10(
	root='./data', train=True, download=True, transform=torchvision.transforms.Compose(
	[#torchvision.transforms.Grayscale(num_output_channels=1), 
	torchvision.transforms.ToTensor()]))
test = torchvision.datasets.CIFAR10(
	root='./data', train=False, download=True, transform=torchvision.transforms.Compose(
	[#torchvision.transforms.Grayscale(num_output_channels=1), 
	torchvision.transforms.ToTensor()]))

n_classes = 10
dim = 32
if dim==32:
    n_channels=3
else:
    n_channels=1

j=0
np.random.seed(j)
torch.manual_seed(j)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

n_filters=16
f_shape = 9
batch_size=100
# Creating a ConvRBM
model = ConvRBM(visible_shape=(dim, dim), filter_shape=(f_shape, f_shape), n_filters=n_filters, n_channels=n_channels, learning_rate=0.00001, momentum=0.5, use_gpu=True)

# Training a ConvRBM
mse = model.fit(train, batch_size=batch_size, epochs=1)

torch.save(model, 'model.pth')

# Creating the Fully Connected layer to append on top of RBM
fc = torch.nn.Linear(model.hidden_shape[0]*model.hidden_shape[1]*n_filters, n_classes)

# Check if model uses GPU
if model.device == 'cuda':
    # If yes, put fully-connected on GPU
    fc = fc.cuda()

# Cross-Entropy loss is used for the discriminative fine-tuning
criterion = nn.CrossEntropyLoss()

# Creating the optimzers
optimizer = [optim.Adam(model.parameters(), lr=0.0001),
             optim.Adam(fc.parameters(), lr=0.001)]

# Creating training and validation batches
train_batch = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=1)
val_batch = DataLoader(test, batch_size=10000, shuffle=True, num_workers=1)

fine_tune_epochs = 5
# For amount of fine-tuning epochs	
for e in range(fine_tune_epochs):
    print(f'Epoch {e+1}/{fine_tune_epochs}')

    # Resetting metrics
    train_loss, val_acc = 0, 0
    
    # For every possible batch
    for x_batch, y_batch in tqdm(train_batch):
        # For every possible optimizer
        for opt in optimizer:
            # Resets the optimizer
            opt.zero_grad()        

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
    for x_batch, y_batch in tqdm(val_batch):

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
