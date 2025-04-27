import torch
import torch.nn as nn
import torch.nn.functional as F

class KspacePositionEmbedding(nn.Module):
    def __init__(self, H: int, W: int):
        super().__init__()

        ky, kx = torch.meshgrid(
            torch.linspace(-H/2,  H/2-1, H),
            torch.linspace(-W/2,  W/2-1, W),
            indexing="ij"
        )
        kx, ky = kx.float(), ky.float()

        rho       = torch.sqrt(kx**2 + ky**2)
        rho_max   = rho.max()
        theta     = torch.atan2(ky, kx)
        n_rho     = rho / rho_max

        base = torch.stack(
            [
                n_rho,
                torch.log1p(rho),
                torch.sin(theta),
                torch.cos(theta),
                n_rho * torch.sin(theta),
                n_rho * torch.cos(theta),
            ],
            dim=0,          # (6, H, W)
        )
        self.register_buffer("base_emb", base)   # stored once, no grad

    def forward(self, x):
        n = x.size(0)
        return self.base_emb.expand(n, -1, -1, -1)   # (N, 6, H, W) view