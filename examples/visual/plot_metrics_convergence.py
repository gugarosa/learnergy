# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import torch

import learnergy.visual.convergence as c

model = torch.load("model.pth")

c.plot(
    model.history["mse"],
    model.history["pl"],
    model.history["time"],
    labels=["MSE", "log-PL", "time (s)"],
    title="convergence over MNIST dataset",
    subtitle="Model: Restricted Boltzmann Machine",
)
