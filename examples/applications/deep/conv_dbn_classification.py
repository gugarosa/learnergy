import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm
from learnergy.models.deep import ConvDBN

# Creating training and testing dataset
train = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
test = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)

# Creating a ConvDBN
model = ConvDBN(
    model="gaussian",
    visible_shape=(28, 28),
    filter_shape=((5, 5), (5, 5)),
    n_filters=(32, 32),
    steps=(1, 1),
    n_channels=1,
    learning_rate=(0.0001, 0.00001),
    momentum=(0, 0),
    decay=(0, 0),
    maxpooling=(False, True),
    use_gpu=True,
)

batch_size = 128
n_classes = 10
fine_tune_epochs = 10
epochs = (1, 1)

# Training a ConvDBN
model.fit(train, batch_size=batch_size, epochs=epochs)

# Reconstructing test set
#rec_mse, v = model.reconstruct(test)

# Saving model
torch.save(model, "model.pth")

# Creating the Fully Connected layer to append on top of DBN
h1 = model.models[len(model.models)-1].hidden_shape[0]
h2 = model.models[len(model.models)-1].hidden_shape[1]
nf = model.models[len(model.models)-1].n_filters

if model.models[len(model.models)-1].maxpooling:
    input_fc = nf * (h1//2 + 1) * (h2//2 + 1)
else:
    input_fc = nf * h1 * h2
fc = nn.Linear(input_fc , n_classes)

# Check if model uses GPU
if model.device == "cuda":
    # If yes, put fully-connected on GPU
    fc = fc.cuda()

# Cross-Entropy loss is used for the discriminative fine-tuning
criterion = nn.CrossEntropyLoss()

# Creating the optimzers
optimizer = [
    optim.Adam(model.models[0].parameters(), lr=0.0001),
    optim.Adam(model.models[1].parameters(), lr=0.0001),
    optim.Adam(fc.parameters(), lr=0.001),
]

# Creating training and validation batches
train_batch = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
val_batch = DataLoader(test, batch_size=10000, shuffle=False, num_workers=0)

# For amount of fine-tuning epochs
for e in range(fine_tune_epochs):
    print(f"Epoch {e+1}/{fine_tune_epochs}")

    # Resetting metrics
    train_loss, val_acc = 0, 0

    # For every possible batch
    for x_batch, y_batch in tqdm(train_batch):
        # For every possible optimizer
        for opt in optimizer:
            # Resets the optimizer
            opt.zero_grad()

        # Checking whether GPU is avaliable and if it should be used
        if model.device == "cuda":
            # Applies the GPU usage to the data and labels
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

        # Passing the batch down the model
        y = model(x_batch)
        
        # Reshaping the outputs
        y = y.reshape(
            x_batch.size(0), input_fc)

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
        if model.device == "cuda":
            # Applies the GPU usage to the data and labels
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

        # Passing the batch down the model
        y = model(x_batch)
        
        # Reshaping the outputs
        y = y.reshape(
            x_batch.size(0), input_fc)

        # Calculating the fully-connected outputs
        y = fc(y)

        # Calculating predictions
        _, preds = torch.max(y, 1)

        # Calculating validation set accuracy
        val_acc = torch.mean((torch.sum(preds == y_batch).float()) / x_batch.size(0))

    print(f"Loss: {train_loss / len(train_batch)} | Val Accuracy: {val_acc}")

# Saving the fine-tuned model
torch.save(model, "tuned_model.pth")

# Checking the model's history
for m in model.models:
    print(m.history)
