import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

import learnergy.visual.image as im
import learnergy.visual.tensor as t
from learnergy.models.dbn import DBN
from learnergy.models.residual_dbn import ResidualDBN
from learnergy.models.rbm import RBM
from learnergy.models.sigmoid_rbm import SigmoidRBM as SRBM

if __name__ == '__main__':
    # Defining some input variables
    epochs = 10

    # Creating training and testing dataset
    train = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    test = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

    # Creating a DBN
    model = DBN(model='bernoulli', n_visible=784, n_hidden=[256, 256], steps=[1, 1],
                learning_rate=[0.1, 0.1], momentum=[0, 0], decay=[0, 0], temperature=[1, 1],
                use_gpu=True)

    # Or you may create a ResidualDBN
    # model = ResidualDBN(model='bernoulli', n_visible=784, n_hidden=[256, 256], steps=[1, 1],
    #                     learning_rate=[0.1, 0.1], momentum=[0, 0], decay=[0, 0], temperature=[1, 1],
    #                     alpha=1, beta=1, use_gpu=True)

    # Training a DBN
    model.fit(train, batch_size=128, epochs=[5, 5])

    # Creating the Fully Connected layer to append on top of DBNs
    fc = torch.nn.Linear(model.n_hidden[model.n_layers - 1], 10)

    # Check if model uses GPU
    if model.device == 'cuda':
        # If yes, put fully-connected on GPU
        fc = fc.cuda()

    # Cross-Entropy loss is used for the discriminative fine-tuning
    criterion = nn.CrossEntropyLoss()

    # Creating the optimzers
    optimizer = [optim.Adam(model.models[i].parameters(), lr=0.001) for i in range(model.n_layers)]

    # Creating training and testing batches
    train_batch = DataLoader(train, batch_size=128, shuffle=False, num_workers=1)
    test_batch = DataLoader(test, batch_size=10000, shuffle=False, num_workers=1)

    # For amount of fine-tuning epochs
    for e in range(epochs):
        print(f'Epoch {e+1}/{epochs}')

        # Resetting metrics
        train_loss, acc = 0, 0
        
        # For every possible batch
        for x_batch, y_batch in train_batch:
            # For every possible layer
            for i in range(model.n_layers):
                # Resets the optimizer
                optimizer[i].zero_grad()
            
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
            
            # For every possible layer
            for i in range(model.n_layers):
                # Performs the gradient update
                optimizer[i].step()

            # Adding current batch loss
            train_loss += loss.item()
            
        # Calculate the test accuracy for the model:
        for x_batch, y_batch in test_batch:
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

        _, predicted = torch.max(y, 1)
        pred = predicted.cpu().numpy()
        acc = 0
        for z in range(y_batch.size(0)):
            if (y_batch[z] == pred[z]):
                acc += 1
        acc = np.round(acc / y_batch.shape[0], 4)
        print('[%d] Loss: %.4f | Acc: %.4f' % (e + 1, train_loss / len(train_batch), acc))

    # Visualizing some reconstructions after discriminative fine-tuning:
    rec_mse, v = model.reconstruct(test)
    t.show_tensor(v[0].view(28, 28)), t.show_tensor(test.data[0,:].view(28, 28))

    # Saving model
    torch.save(model, 'model.pth')

    # Checking the model's history
    for m in model.models:
        print(m.history)
