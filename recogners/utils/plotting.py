import matplotlib.pyplot as plt
from PIL import Image
import recogners.utils.wv as wv

def show(tensor):
    plt.figure()
    plt.imshow(tensor.numpy(), cmap=plt.cm.gray)
    plt.show()
    
def weights_visualize(tensor):
    w8 = tensor.numpy()
    d1 = w8.shape[0]
    d2 = w8.shape[1]
    s = int(np.sqrt(w8.shape[1]))
    img = Image.fromarray(wv.tile_raster_images(X = w8.T, img_shape=(d1, d2), tile_shape=(s,s), tile_spacing=(1,1)))
    img.show()
