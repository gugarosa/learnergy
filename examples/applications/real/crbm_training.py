import torch
import torchvision
import numpy as np
import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

#from learnergy.models.binary import ConvRBM
from learnergy.models.real import ConvRBM
import learnergy.visual.tensor as t

# Creating training and testing dataset
train = torchvision.datasets.CIFAR10(
	root='./data', train=True, download=True, transform=torchvision.transforms.Compose(
	[#torchvision.transforms.Grayscale(num_output_channels=1), 
	torchvision.transforms.ToTensor()]))
test = torchvision.datasets.CIFAR10(
	root='./data', train=False, download=True, transform=torchvision.transforms.Compose(
	[#torchvision.transforms.Grayscale(num_output_channels=1), 
	torchvision.transforms.ToTensor()]))

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

# Creating a ConvRBM
model = ConvRBM(visible_shape=(dim, dim), filter_shape=(9, 9), n_filters=32, n_channels=n_channels, learning_rate=0.00001, momentum=0.5, use_gpu=True)

# Training a ConvRBM
mse = model.fit(train, batch_size=100, epochs=5)

# Reconstructing test set
_, v = model.reconstruct(test)

# Showing a reconstructed sample
t.show_tensor(v[0].squeeze())
t.show_tensor(test.__getitem__(0)[0].squeeze())

