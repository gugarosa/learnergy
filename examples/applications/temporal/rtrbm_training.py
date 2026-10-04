# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import torch
from torch.utils.data import TensorDataset

from learnergy.models.temporal import RTRBM

torch.manual_seed(3)
sequences = torch.zeros(128, 8, 6)
sequences[:, 0::2, :3] = 1
sequences[:, 1::2, 3:] = 1
sequences = torch.where(torch.rand_like(sequences) < 0.03, 1 - sequences, sequences)
dataset = TensorDataset(sequences, torch.zeros(len(sequences)))

model = RTRBM(n_visible=6, n_hidden=12, learning_rate=0.05)

mse = model.fit(dataset, batch_size=32, epochs=20)
reconstruction_mse, reconstructed = model.reconstruct(dataset)

with torch.no_grad():
    hidden_sequences = model(sequences)

generated = model.sample(n_samples=8, n_steps=8, gibbs_steps=50)

print(f"Training MSE: {mse.item():.4f}")
print(f"Reconstruction MSE: {reconstruction_mse.item():.4f}")
print(f"Hidden sequences: {tuple(hidden_sequences.shape)}")
print(f"Reconstructed sequences: {tuple(reconstructed.shape)}")
print(f"Generated sequences: {tuple(generated.shape)}")
