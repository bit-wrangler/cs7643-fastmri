"""
SSIM loss implementation based on FastMRI's implementation.

This implementation is adapted from:
https://github.com/facebookresearch/fastMRI/blob/91f2df4711adbb6d643df1810f234e4abcf5881b/banding_removal/fastmri/ssim_loss_mixin.py
"""

import torch
import torch.nn.functional as F
from torch import nn


class SSIM(nn.Module):
    def __init__(self, win_size=7, k1=0.01, k2=0.03):
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer('w', torch.ones(1, 1, win_size, win_size) / win_size**2)
        NP = win_size**2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X, Y, data_range):
        """
        Calculate SSIM between X and Y.
        
        Args:
            X: First image
            Y: Second image
            data_range: Data range of the images, usually the maximum value
                       Should be a tensor of shape [batch_size]
        
        Returns:
            Mean SSIM value
        """
        data_range = data_range[:, None, None, None]

        C1 = (self.k1 * data_range)**2
        C2 = (self.k2 * data_range)**2

        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)

        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        
        A1, A2, B1, B2 = (2 * ux * uy + C1, 2 * vxy + C2, ux ** 2 + uy ** 2 + C1, vx + vy + C2)
        D = B1 * B2
        S = (A1 * A2) / D
        
        return S.mean()


def ssim_loss(X, Y, data_range, win_size=7, k1=0.01, k2=0.03):
    """
    Functional interface for SSIM loss.
    
    Args:
        X: First image
        Y: Second image
        data_range: Data range of the images, usually the maximum value
        win_size: Window size for SSIM calculation
        k1, k2: Constants for SSIM calculation
        
    Returns:
        1 - SSIM (to use as a loss where lower is better)
    """
    ssim_module = SSIM(win_size=win_size, k1=k1, k2=k2).to(X.device)
    return 1.0 - ssim_module(X, Y, data_range)
