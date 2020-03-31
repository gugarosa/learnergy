import pytest

from learnergy.visual import metrics


def test_metrics_plot():
    new_model = {
        'mse': [1, 2, 3],
        'pl': [1.5, 2, 2.5],
        'time': [0.1, 0.2, 0.3]
    }

    try:
        metrics.plot(
            new_model['mse'], new_model['pl'], new_model['time'], labels=1)
    except:
        metrics.plot(new_model['mse'], new_model['pl'],
                     new_model['time'], labels=['MSE', 'log-PL', 'time (s)'])

    try:
        metrics.plot(new_model['mse'], new_model['pl'],
                     new_model['time'], labels=['MSE'])
    except:
        metrics.plot(
            new_model['mse'], new_model['pl'], new_model['time'])
