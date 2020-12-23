import torch

import learnergy.visual.convergence as c

# Loading pre-trained model
model = torch.load('model.pth')

# Plotting convergence per layer from pre-trained model
c.plot(model.history['mse'], model.history['pl'], model.history['time'], labels=['MSE', 'log-PL', 'time (s)'],
       title='convergence over MNIST dataset', subtitle='Model: Restricted Boltzmann Machine')
