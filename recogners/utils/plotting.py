import matplotlib.pyplot as plt
import numpy as np
import recogners.utils.wv as wv
from PIL import Image

def show(tensor):
    plt.figure()
    plt.imshow(tensor.numpy(), cmap=plt.cm.gray)
    plt.show()
    
def weights_visualize(tensor):
    w8 = tensor.numpy()
    d = int(np.sqrt(w8.shape[0]))
    s = int(np.sqrt(w8.shape[1]))
    img = Image.fromarray(wv.tile_raster_images(X = w8.T, img_shape=(d, d), tile_shape=(s,s), tile_spacing=(1,1)))
    img.show()
