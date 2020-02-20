import pytest
import torch
from learnergy.visual import metrics


def test_metrics_plot():
    new_model = torch.load('model.pth')

    try:
        metrics.plot(
            new_model.history['mse'], new_model.history['pl'], new_model.history['time'], labels=1)
    except:
        metrics.plot(new_model.history['mse'], new_model.history['pl'],
                     new_model.history['time'], labels=['MSE', 'log-PL', 'time (s)'])

    try:
        metrics.plot(new_model.history['mse'], new_model.history['pl'],
                     new_model.history['time'], labels=['MSE'])
    except:
        metrics.plot(
            new_model.history['mse'], new_model.history['pl'], new_model.history['time'])
