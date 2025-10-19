import torch 
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import copy

def refine_anomaly_map(H, gt, max_iter=50, lr=1e-3, stop_loss=1e-7):
    """
    Hyperspectral anomaly detect based on autoencoder

    Args:
        H: [1, bands, H, W] hyperspectral tensor (torch, usually cuda)
        gt: numpy or torch.Tensor, ground truth labels [H, W], values in {0,1}
        max_iter: maximum iterations
        lr: learning rate
        stop_loss: early stopping threshold

    Returns:
        best_net: network with highest AUC during iterations
        best_Sp: corresponding reconstruction error map at highest AUC (numpy)
    """
    device = H.device
    H = H.to(torch.float32)
    H = (H - H.min()) / (H.max() - H.min() + 1e-8)  
    gt = torch.as_tensor(gt, device=device, dtype=torch.float32)
    gt_flat = gt.flatten().detach().cpu().numpy()

    _, bands, Hh, Ww = H.shape

    # --- Define lightweight autoencoder ---
    class LightAutoEncoder(nn.Module):
        def __init__(self, in_ch):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Sigmoid(),
                nn.Conv2d(in_ch, 128, kernel_size=1, bias=True),
                nn.Conv2d(128, 64, kernel_size=1, bias=True),
                nn.Conv2d(64, 32, kernel_size=1, bias=True),
                nn.Tanh(),
            )
            self.decoder = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=1, bias=True),
                nn.Conv2d(64, 128, kernel_size=1, bias=True),
                nn.Conv2d(128, in_ch, kernel_size=1, bias=True),
                nn.Sigmoid()  
            )

        def forward(self, x):
            z = self.encoder(x)
            x_rec = self.decoder(z)
            return x_rec

    net = LightAutoEncoder(bands).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    mse_loss = nn.MSELoss(reduction='mean')

    # --- Initialization ---
    best_auc = 0.0
    best_Sp = None
    best_net = copy.deepcopy(net).cpu()

    # --- Iterative training ---
    for it in range(max_iter):
        optimizer.zero_grad(set_to_none=True)

        # Forward pass + reconstruction
        H_rec = net(H)
        loss = mse_loss(H_rec, H)

        # Backward pass
        loss.backward()
        optimizer.step()

        # --- Compute current reconstruction error map ---
        with torch.no_grad():
            S_p = ((H_rec - H) ** 2).mean(dim=1).squeeze(0)  # [H,W]
            Sp_np = S_p.detach().cpu().numpy().flatten()
            auc = roc_auc_score(gt_flat, Sp_np)

        # print(f"[Iter {it+1:02d}] loss={loss.item():.6f}, AUC={auc:.4f}")

        # --- Record best model ---
        if auc > best_auc:
            best_auc = auc
            best_Sp = S_p.detach().cpu().numpy()
            best_net = copy.deepcopy(net).cpu()

        if loss.item() < stop_loss:
            print(f"Early stop at iter {it+1}, loss={loss.item():.6f}, AUC={auc:.4f}")
            break

    # print(f" Best AUC = {best_auc:.4f}")
    return best_net, best_Sp  #, best_auc