import torch
import torchvision

from learnergy.models.discriminative_rbm import DiscriminativeRBM

# Creating training and testing dataset
train = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
test = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

# Creating a DiscriminativeRBM
model = DiscriminativeRBM(n_visible=784, n_hidden=128, n_classes=10, learning_rate=0.1,
                          momentum=0, decay=0, use_gpu=True)

# Training a DiscriminativeRBM
loss, acc = model.fit(train, batch_size=128, epochs=5)

# Predicting test set
pred_acc, pred_probs, pred_labels = model.predict(test)

# Saving model
torch.save(model, 'model.pth')

# Checking the model's history
print(model.history)
