import matplotlib.pyplot as plt

def show(tensor):
    plt.figure()
    plt.imshow(tensor.numpy(), cmap=plt.cm.gray)
    plt.show()