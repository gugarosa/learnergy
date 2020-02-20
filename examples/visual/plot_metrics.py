import torch

import learnergy.visual.metrics as m

# Loading pre-trained model
model = torch.load('model.pth')

# Plotting metrics per layer from pre-trained model
m.plot(model.history['mse'], model.history['pl'], model.history['time'], labels=['MSE', 'log-PL', 'time (s)'],
       title='Metrics over MNIST dataset', subtitle='Model: Restricted Boltzmann Machine')
