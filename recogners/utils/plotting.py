import matplotlib.pyplot as plt

def showTensor(aTensor):
    plt.figure()
    plt.imshow(aTensor.numpy(), cmap=plt.cm.gray)
    plt.show()