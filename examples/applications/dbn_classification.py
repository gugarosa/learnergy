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

# Creating training and testing dataset
train = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
test = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

# Creating a DBN
model = DBN(model='bernoulli', n_visible=784, n_hidden=[256, 256], steps=[1, 1],
            learning_rate=[0.1, 0.1], momentum=[0, 0], decay=[0, 0], temperature=[1, 1],
            use_gpu=True)

# model = ResidualDBN(model='bernoulli', n_visible=784, n_hidden=[256, 256], steps=[1, 1],
#             learning_rate=[0.1, 0.1], momentum=[0, 0], decay=[0, 0], temperature=[1, 1],
#             alpha=1, beta=1, use_gpu=True)

# Training a DBN
model.fit(train, batch_size=128, epochs=[5, 5])


fc = torch.nn.Linear(model.n_hidden[model.n_layers-1], 10)

# Cross-Entropy Loss for the discriminative fine-tuning
criterion = nn.CrossEntropyLoss()

# Creating the optimzers
optimizer = []
for k in range(model.n_layers):
    optimizer.append(optim.Adam(model.models[k].parameters(), lr=10 ** (-3)))

bs = 2**8 # Size of the batch for fine-tuning
n = 5     # Epochs for fine-tuning
ac = np.zeros(n)

batches = DataLoader(train, batch_size=bs, shuffle=False, num_workers=1)
test_ = DataLoader(test, batch_size=test.data.size(0), shuffle=False, num_workers=1)

for epoch in range(n):
    running_loss = 0.0
    val_loss = 0.0
    
    for samples, y in batches:
        for k in range(model.n_layers):
            optimizer[k].zero_grad()
            
        samples = samples.view(len(samples), model.n_visible)
        outputs = model(samples)
        outputs = fc(outputs)
        # print(outputs)
        loss = criterion(outputs, y)
        #for x_test, y_test in test_:
        #    out = model(x_test)
        #    val_loss += criterion(out, y_test).item()
        loss.backward()
        
        for k in range(model.n_layers):
            optimizer[k].step()

        running_loss += loss.item()
        
    # Calculate the test accuracy for the model:
    for x_test, y_test in test_:
        x = x_test.view(len(x_test), model.n_visible)
        out = model(x)
        out = fc(out)

    _, predicted = torch.max(out, 1)
    pred = predicted.cpu().numpy()
    acc = 0
    for z in range(y_test.size(0)):
        if (y_test[z] == pred[z]):
            acc += 1
    ac[epoch] = np.round(acc / y_test.shape[0], 4)
    print('[%d] Loss: %.4f | Acc: %.4f' % (epoch + 1, running_loss / (train.data.size(0) - bs), ac[epoch]))

# Visualizing some reconstructions after discriminative fine-tuning:
rec_mse, v = model.reconstruct(test)
t.show_tensor(v[0].view(28, 28)), t.show_tensor(test.data[0,:].view(28, 28))

# Saving model
torch.save(model, 'fine_model.pth')

# Checking the model's history
for m in model.models:
    print(m.history)
