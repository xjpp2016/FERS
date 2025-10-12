import numpy as np

import torch
import torch.linalg as linalg

def RX_Torch(hsi_batch):
    """
    Mahalanobis Distance anomaly detector (Batch PyTorch Version)
    Uses global image mean and covariance as background estimates

    Inputs:
        hsi_batch - Tensor of shape (batch_size, n_band, n_row, n_col)

    Outputs:
        dist_batch - Tensor of shape (batch_size, 1, n_row, n_col)

    8/7/2012 - Taylor C. Glenn
    5/5/2018 - Edited by Alina Zare
    11/2018 - Python Implementation by Yutai Zhou
    2023 - PyTorch Batch Implementation
    """
    batch_size, n_band, n_row, n_col = hsi_batch.shape
    n_pixels = n_row * n_col
    
    # Normalize each image in the batch
    hsi_normalized = (hsi_batch - hsi_batch.view(batch_size, -1).min(dim=1, keepdim=True)[0].unsqueeze(-1)) / \
                     (hsi_batch.view(batch_size, -1).max(dim=1, keepdim=True)[0].unsqueeze(-1) - 
                      hsi_batch.view(batch_size, -1).min(dim=1, keepdim=True)[0].unsqueeze(-1) + 1e-10)
    
    # Reshape to (batch_size, n_band, n_pixels)
    hsi_data = hsi_normalized.reshape(batch_size, n_band, n_pixels)
    
    # Calculate mean and covariance for each image in the batch
    mu = torch.mean(hsi_data, dim=2, keepdim=True)  # (batch_size, n_band, 1)
    
    # Calculate covariance matrices
    z = hsi_data - mu  # (batch_size, n_band, n_pixels)
    sigma = torch.matmul(z, z.transpose(1, 2)) / (n_pixels - 1)  # (batch_size, n_band, n_band)
    
    # Calculate pseudo-inverse for each covariance matrix
    sig_inv = linalg.pinv(sigma)  # (batch_size, n_band, n_band)
    
    # Calculate Mahalanobis distance for each pixel
    # Using einsum for batch matrix multiplication: bij,bjk,bkl->bil
    dist_data = torch.einsum('bni,bij,bjn->bn', 
                            z.transpose(1, 2), 
                            sig_inv, 
                            z)  # (batch_size, n_pixels)
    
    # Reshape to output format
    dist_batch = dist_data.reshape(batch_size, 1, n_row, n_col)
    
    return dist_batch


def RX(hsi_img):
    """

    Mahalanobis Distance anomaly detector
    uses global image mean and covariance as background estimates

    Inputs:
     hsi_image - n_row x n_col x n_band hyperspectral image
     mask - binary image limiting detector operation to pixels where mask is true
            if not present or empty, no mask restrictions are used

    Outputs:
      dist_img - detector output image

    8/7/2012 - Taylor C. Glenn
    5/5/2018 - Edited by Alina Zare
    11/2018 - Python Implementation by Yutai Zhou
    """
    hsi_img = hsi_img.detach().squeeze().cpu().numpy().transpose(1, 2, 0)
    hsi_img = (hsi_img-np.min(hsi_img))/(np.max(hsi_img)-np.min(hsi_img))
    n_row, n_col, n_band = hsi_img.shape
    n_pixels = n_row * n_col
    hsi_data = np.reshape(hsi_img, (n_pixels, n_band), order='F').T

    mu = np.mean(hsi_data, 1)
    sigma = np.cov(hsi_data.T, rowvar=False)

    z = hsi_data - mu[:, np.newaxis]
    sig_inv = np.linalg.pinv(sigma)

    dist_data = np.zeros(n_pixels)
    for i in range(n_pixels):
        dist_data[i] = z[:, i].T @ sig_inv @ z[:, i]

    return dist_data.reshape([n_col, n_row]).T