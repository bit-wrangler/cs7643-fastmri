import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedSpectralAutoencoder(nn.Module):
    def __init__(
            self,
            in_c: int = 2,
            mask_pos_c: int = 4,
            out_c: int = 2,
            H: int = 320,
            W: int = 320,
            latent_dim: int = 1024,
            ):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.H = H
        self.W = W

        flat_dim = 128 * (H // 8) * (W // 8)

        self.pos_embedding = nn.Parameter(torch.randn(mask_pos_c, H, W))
        self.mask_embedding = nn.Embedding(2, mask_pos_c)

        d1 = 32
        d2 = 64
        d3 = 128
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3

        self.encoder = nn.Sequential(
            nn.Conv2d(in_c+mask_pos_c, d1,  kernel_size=3, padding=1),
            nn.PReLU(d1),

            nn.Conv2d(d1, d2, kernel_size=3, padding=1),
            nn.PReLU(d2),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(d2, d3, kernel_size=3, padding=1),
            nn.PReLU(d3),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(d3, d3, kernel_size=3, padding=1),
            nn.PReLU(d3),
            nn.MaxPool2d(2, 2)
        )

        self.enc_fc = nn.Linear(flat_dim, latent_dim)

        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim, flat_dim),
            nn.PReLU(flat_dim)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(d3, d3, kernel_size=2, stride=2),  # 1/4 → 1/2
            nn.PReLU(d3),
            nn.ConvTranspose2d(d3, d2, kernel_size=2, stride=2),  # 1/2 → 1
            nn.PReLU(d2),
            nn.ConvTranspose2d(d2, d1,  kernel_size=2, stride=2),  # 1   → 2
            nn.PReLU(d1),
            nn.ConvTranspose2d(d1, out_c,    kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, col_mask):
        # x is of shape (n_slices, 2, H, W)
        # mask is of shape (W)
        n_slices = x.shape[0]
        H = x.shape[2]
        W = x.shape[3]
        M = col_mask.sum()

        # create 4d mask tensor of shape (n_slices, 1, H, W)
        mask4d = col_mask.view(1, 1, 1, W).repeat(n_slices, 1, H, 1)
        mask_embedding = self.mask_embedding(mask4d.squeeze().long()).permute(0, 3, 1, 2) # (n_slices, mask_pos_c, H, W)
        embeddings = self.pos_embedding.unsqueeze(0).repeat(n_slices, 1, 1, 1) + mask_embedding
        x = torch.cat([x, embeddings], dim=1) # (n_slices, in_c+mask_pos_c, H, W)

        x = self.encoder(x) # (n_slices, 256, H/8, W/8)
        x = x.reshape(n_slices, -1) # (n_slices, 256 * H/8 * W/8)
        x = self.enc_fc(x) # (n_slices, latent_dim)

        x = self.dec_fc(x) # (n_slices, 256 * H/8 * W/8)
        x = x.reshape(n_slices, self.d3, H//8, W//8) # (n_slices, 256, H/8, W/8)
        x = self.decoder(x) # (n_slices, out_c, H, W)

        return x
