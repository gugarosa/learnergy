import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

#from learnergy.models.real import GaussianConvRBM
from learnergy.models.stack import CDBN
import learnergy.visual.image as im
#import learnergy.visual.tensor as t

# Defining some input variables
v_shape = 32
n_filters = [16*2, 16*2*2]
f_shape = [(9, 9), (7, 7)] # 9 for cifar dataset
n_channels = 3
batch_size = 100
n_classes = 10
fine_tune_epochs = 100//2

# Creating training and validation/testing dataset
banco = 'cifar'
train = torchvision.datasets.CIFAR10(
#train = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
test = torchvision.datasets.CIFAR10(
#test = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

# Creating a GaussianConvRBM
model = CDBN(visible_shape=(v_shape, v_shape), filter_shape=f_shape,
             n_filters=n_filters, n_channels=n_channels, n_layers=2,
             learning_rate=(0.00001, 0.00001), momentum=(0.9, 0.9), decay=(0., 0.), use_gpu=True)

torch.manual_seed(0)
try:
    model = torch.load(banco+'_cdbn_model.pth')
    model.eval()
    model.cuda()
    print("Model loaded")
except:
    # Training a ConvRBM
    model.fit(train, batch_size=batch_size, epochs=(50, 50))

    torch.save(model, banco+'_cdbn_model.pth')

rec_mse, v = model.reconstruct(test)

img = im.vis_square(torch.from_numpy(test.data[:1000]), 10)
#img = im.vis_square(test.data[:1000], 10)
img.savefig(banco+"_orig.jpg", dpi=300, quality=95, bbox_inches='tight')
img = im.vis_square(v[:1000].detach().cpu(), 10)
img.savefig(banco+"_rec.jpg", dpi=300, quality=95, bbox_inches='tight')


# Creating the Fully Connected layer to append on top of RBM
in_shape = model.hidden_shape[0] * model.hidden_shape[1] * n_filters[len(n_filters)-1]
fc = nn.Linear(in_shape, n_classes)

# Check if model uses GPU
if model.device == 'cuda':
    # If yes, put fully-connected on GPU
    fc = fc.cuda()

# Cross-Entropy loss is used for the discriminative fine-tuning
criterion = nn.CrossEntropyLoss()

# Creating the optimzers
optimizer = []
for i in range(model.n_layers):
    optimizer.append(optim.Adam(model.models[i].parameters(), lr=0.001))
optimizer.append(optim.Adam(fc.parameters(), lr=0.001))

# Creating training and validation batches
batch_size = 2**7
train_batch = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=0)
val_batch = DataLoader(test, batch_size=10000, shuffle=False, num_workers=0)

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
        y = model(x_batch).detach()

        # Reshaping the outputs
        y = y.reshape(x_batch.size(0), in_shape)

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

        # Reshaping the outputs
        y = y.reshape(x_batch.size(0), in_shape)

        # Calculating the fully-connected outputs
        y = fc(y).detach()

        # Calculating predictions
        _, preds = torch.max(y, 1)

        # Calculating validation set accuracy
        val_acc = torch.mean((torch.sum(preds == y_batch).float()) / x_batch.size(0))

    print(f'Loss: {train_loss / len(train_batch)} | Val Accuracy: {val_acc}')

# Saving the fine-tuned model
torch.save(model, banco+'_tuned_model.pth')

# Checking the model's history
#print(model.history)
