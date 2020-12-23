from skimage.metrics import structural_similarity as ssim

import learnergy.utils.logging as l

logger = l.get_logger(__name__)

def calculate_ssim(v, x):
    """Calculates the structural similarity of images.

    Args:
        v (torch.Tensor): Reconstructed images.
        x (torch.Tensor): Original images.

    Returns:
        The structural similarity between input images.
        
    """
    
    ssim_rec = 0
    v = v.cpu().detach().numpy()
    x = x.cpu().numpy()
    dx = x.shape[1]
    dy = x.shape[2]
    
    for z in range(v.shape[0]):
        img = (x[z, :, :]/255.0).round()
        new = v[z, :].reshape((dx, dy))
        ssim_rec += ssim(img, new, data_range=img.max() - img.min())
        
    mean = ssim_rec/v.shape[0]
    
    logger.info('Mean SSIM: %f', mean)
    
    return mean
