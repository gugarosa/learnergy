import numpy as np

import learnergy.utils.logging as l

from skimage.metrics import structural_similarity as ssim

logger = l.get_logger(__name__)

def make_ssim(v, test):
    """A function to make the SSIM among images.
    
    Input:
          v (torch.tensor): Tensor containing reconstructed images (batch, h, w).
          test (torch.tensor): Tensor containing test images.
    """
    
    ssim_rec = 0
    v = v.cpu().detach().numpy()
    test = test.cpu().numpy()
    dx = test.shape[1]
    dy = test.shape[2]
    
    for z in range(v.shape[0]):
        img = (test[z, :, :]/255.0).round()
        new = v[z, :].reshape((dx, dy))
        ssim_rec += ssim(img, new, data_range=img.max() - img.min())
        
    mean = ssim_rec/v.shape[0]
    
    logger.info('Mean SSIM: %f', np.round(mean, 4))
    
    return np.round(mean, 4)
