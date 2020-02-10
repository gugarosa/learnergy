from torch.utils.data import DataLoader

import torchvision
from learnergy.models.rbm3 import RBM3

if __name__ == '__main__':
    # Creating training and testing dataset
    train = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    test = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

    # Creating training and testing batches
    train_batches = DataLoader(train, batch_size=128, shuffle=True, num_workers=1)
    test_batches = DataLoader(test, batch_size=10000, shuffle=True, num_workers=1)

    # Creating an RBM
    model = RBM3(n_visible=784, n_hidden=4096, steps=1,
                learning_rate=0.1, momentum=0, decay=0, temperature=1, use_gpu=True)

    error, pl = model.fit(train_batches, epochs=3)

    print(model.history)