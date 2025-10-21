import torch
import torch.nn as nn
import torch.nn.functional as F
    

class CrossCovarianceLoss(nn.Module):
    def __init__(self, bands=200):
        super().__init__()
        self.bands = bands

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        A: (b, f, h, w), B: (b, f1, h, w)
        """
        b, f, h, w = A.shape
        _, f1, _, _ = B.shape
        L = h * w

        # Flatten and zero-mean
        A_flat = A.reshape(b, f, h*w)
        B_flat = B.reshape(b, f1, h*w)

        A_flat = (A_flat - A_flat.mean(dim=(0, 2), keepdim=True)) / (A_flat.std(dim=(0, 2), keepdim=True) + 1e-6)
        B_flat = (B_flat - B_flat.mean(dim=(0, 2), keepdim=True)) / (B_flat.std(dim=(0, 2), keepdim=True) + 1e-6)

        # Compute cross-covariance
        cov = torch.einsum('bik,bjk->ij', A_flat, B_flat) / (b * L)  # einsum for accelerated matrix computation

        # Loss is the sum of squares of all elements
        loss = cov.pow(2).sum()
        return loss

def variance_preserve_loss(B, eps=1e-6):
    N, C, H, W = B.shape
    x = B.view(N, C, -1)
    var = x.var(dim=2).mean(dim=0)
    # We want variance >= target, penalize only if below target
    target = 1.0  
    penalty = F.relu(target - var).mean()
    penalty = penalty + torch.abs(B).mean()
    return penalty


def cos_sim_loss(spe_f, spa_f):
    cos_sim_loss = 1 + (F.cosine_similarity(spe_f, spa_f, dim=1)).mean()
    return cos_sim_loss