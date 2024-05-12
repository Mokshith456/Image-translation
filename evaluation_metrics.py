import numpy as np
from skimage.metrics import structural_similarity as ssim_skimage
import torch

def ssim(img1, img2):
    # Convert tensors to numpy arrays if not already
    img1 = img1.cpu().detach().numpy() if isinstance(img1, torch.Tensor) else img1
    img2 = img2.cpu().detach().numpy() if isinstance(img2, torch.Tensor) else img2

    # Ensure images are in range [0, 1]
    img1 = img1 / 255.0 if img1.max() > 1 else img1
    img2 = img2 / 255.0 if img2.max() > 1 else img2

    # Calculate SSIM for each image in the batch
    ssim_values = []
    for i in range(img1.shape[0]):  # Assuming batch dimension is first
        current_img1 = img1[i]
        current_img2 = img2[i]

        # Adjust window size to fit the image size
        min_dim = min(current_img1.shape[:2])  # Get the smaller dimension of the image
        win_size = min(7, min_dim - (min_dim % 2 - 1))  # Ensure it's odd and less than or equal to 7

        # Calculate SSIM, ensuring the window size is smaller than the smallest image dimension
        ssim_value = ssim_skimage(current_img1, current_img2, win_size=win_size, multichannel=True, data_range=img1.max() - img1.min())
        ssim_values.append(ssim_value)

    return np.mean(ssim_values)
