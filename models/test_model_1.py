import torch
import torch.nn as nn
import torch.nn.functional as F
from models.kspace_postion_embedding import KspacePositionEmbedding

class PointwiseMixer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, depth: int = 2):
        super().__init__()
        def layer(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, bias=False),
                nn.BatchNorm2d(c_out),
                # nn.GroupNorm(1, c_out),
                nn.GELU(),
            )
        blocks = [layer(in_ch, out_ch)]
        for _ in range(depth - 1):
            blocks.append(layer(out_ch, out_ch))
        self.net = nn.Sequential(*blocks)
        self.residual = (in_ch == out_ch)

    def forward(self, x):
        y = self.net(x)
        return x + y if self.residual else y
    
class Compressor(nn.Module):
    def __init__(self, in_ch, H,W, patch_size: int = 16):
        super().__init__()
        self.patches = nn.Sequential(
            nn.Conv2d(in_ch, in_ch*2, patch_size, stride=patch_size, bias=False),
            nn.BatchNorm2d(in_ch*2),
            nn.GELU(),
        )
        self.net = nn.Sequential(
            nn.Conv2d(in_ch*2, in_ch*2, (H//patch_size,W//patch_size), groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch*2),
            nn.GELU(),
        )

    def forward(self, x):
        # x is (N, C, H, W)
        x = self.patches(x) # N, C, H/patch, W/patch
        y1 = self.net(x) # N, C, 1, 1
        y = y1
        return y
    
class Expander(nn.Module):
    def __init__(self, in_ch, H,W, patch_size: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch*2, in_ch*2, (H//patch_size,W//patch_size), groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch*2),
            nn.GELU(),
        )
        self.unpatches = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch, patch_size, stride=patch_size, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.GELU(),
        )

    def forward(self, x):
        # x is (N, C, H, W)
        y1 = self.net(x) # N, C, H, W
        y1 = self.unpatches(y1) # N, C, H, W
        y = y1 
        return y

class PreNet(nn.Module):
    def __init__(
            self,
            H: int = 320,
            W: int = 320,
            embedding_dim: int = 32,
            embedding_block_depths: tuple[int] = (2, 2),
            ):
        super().__init__()

        self.pos_embedding = KspacePositionEmbedding(H, W)
        self.pos_proj = nn.Conv2d(6, embedding_dim, 1) 

        in_ch = 2 + embedding_dim           # 34 if embedding_dim = 32
        embedding = [PointwiseMixer(in_ch, embedding_dim, embedding_block_depths[0])]
        for i in range(1, len(embedding_block_depths)):
            embedding.append(PointwiseMixer(embedding_dim, embedding_dim, embedding_block_depths[i]))
        self.embedding = nn.Sequential(*embedding)  
        self.mask_embedding = nn.Embedding(2, embedding_dim)
        

    def forward(self, masked_kspace, col_mask, filter_masked: bool = True):
        # kspace is of shape (n_slices, 2, H, W)
        # mask is of shape (W)
        n_slices = masked_kspace.shape[0]
        H = masked_kspace.shape[2]
        W = masked_kspace.shape[3]
        M = col_mask.sum()
        M_d = M
        if not filter_masked:
            M = W

        if filter_masked:
            cols = col_mask.squeeze()
        else:
            cols = torch.ones(W, dtype=torch.bool, device=masked_kspace.device)

        mask_ids = col_mask.squeeze().long() # (W,)
        mask_embedding = self.mask_embedding(mask_ids) # (W, embed_dim)
        mask_embedding = mask_embedding.permute(1, 0) # (embed_dim, W)
        # (embed_dim, W) -> (n_slices, embed_dim, H, W)
        mask_embedding = mask_embedding[None, :, None, :]
        mask_embedding = mask_embedding.expand(n_slices, -1, H, -1)

        pos_embedding = self.pos_embedding(masked_kspace)
        pos_embedding = self.pos_proj(pos_embedding)
        pos_embedding = pos_embedding + mask_embedding

        x = torch.cat([masked_kspace, pos_embedding], dim=1) # (N, 8, H, W)
        x = x[:, :, :, cols] # (N, 8, H, M)
        x = self.embedding(x) # (N, embed_dim, H, M)

        return x

class ColumnEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 4,
        nhead: int = 8,
        ff_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True, 
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        N, C, H, M = x.shape
        x = F.adaptive_avg_pool2d(x, (1, M)).squeeze(2) # (N, C, M)
        x = x.permute(0, 2, 1) # (N, M, C)
        x = self.enc(x) # (N, M, C)
        return x.permute(0, 2, 1) # (N, C, M)
    
class ColumnDecoder(nn.Module):
    def __init__(self, embed_dim: int, H: int, W: int, depth: int = 2):
        super().__init__()
        self.H, self.W = H, W
        self.missing_tok = nn.Parameter(torch.zeros(embed_dim))
        self.mix = PointwiseMixer(embed_dim, embed_dim, depth)

    def forward(self, tokens, col_mask):
        N, C, M   = tokens.shape
        device    = tokens.device

        full = self.missing_tok.view(1, C, 1).expand(N, -1, self.W).clone()

        full = full.to(device) 
        full[:, :, col_mask] = tokens

        full = full.unsqueeze(2).expand(N, C, self.H, self.W)
        return self.mix(full)
    
class PostNet(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, 2, kernel_size=1)

    def forward(self, x):
        return self.proj(x)
    

# based on https://github.com/locuslab/convmixer/blob/main/convmixer.py
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixerLayer(dim, kernel_size=9):
    return nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim))

def UnConvMixerLayer(dim, kernel_size=9):
    return nn.Sequential(
                Residual(nn.Sequential(
                    nn.ConvTranspose2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.ConvTranspose2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim))

def ConvMixerStack(dim, depth, kernel_size=9):
    layers = [ConvMixerLayer(dim, kernel_size) for _ in range(depth)]
    return nn.Sequential(*layers)

def UnConvMixerLayer(dim, kernel_size=9):
    return ConvMixerLayer(dim, kernel_size)

def UnConvMixerStack(dim, depth, kernel_size=9):
    return ConvMixerStack(dim, depth, kernel_size)

def PatchEmbedding(in_ch, dim, patch_size=20):
    return nn.Sequential(
        nn.Conv2d(in_ch, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
    )

def UnPatchEmbedding(in_ch, out_ch, patch_size=20):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_ch, out_ch,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            output_padding=0
        ),
        nn.GELU(),
        nn.BatchNorm2d(out_ch),
    )

"""
Bottleneck down to a single pixel with 1600 channels
1024 1x1
512 2x2 <- (1 adapter + 4 deep convmixer)
256 8x8 <- (1 adapter + 4 deep convmixer)
128 16x16 <- (1 adapter + 4 deep convmixer)
64 16x16 <- after patching
embed(32) 320x320 <- after mixing
2+embed 320x320 <- after concatenating position and mask embedding
2 320x320
"""

class KspaceAutoencoder(nn.Module):
    def __init__(
        self,
        H: int = 320,
        W: int = 320,
        embed_dim: int = 32,
        embedding_block_depths: tuple[int] = (2, 2),
        enc_layers: int = 4,
        enc_heads: int = 8,
    ):
        super().__init__()
        self.prenet   = PreNet(H, W, embed_dim, embedding_block_depths)
        self.net = nn.Sequential(
            PatchEmbedding(embed_dim, embed_dim*2, patch_size=20), # 32,320,320 -> 64,16,16
            ConvMixerStack(embed_dim*2, 4, kernel_size=9), # 64,16,16 -> 64,16,16
            PatchEmbedding(embed_dim*2, embed_dim*4, patch_size=2), # 64,16,16 -> 128,8,8
            ConvMixerStack(embed_dim*4, 4, kernel_size=5), # 128,8,8 -> 128,8,8
            PatchEmbedding(embed_dim*4, embed_dim*8, patch_size=2), # 128,8,8 -> 256,4,4
            ConvMixerStack(embed_dim*8, 4, kernel_size=3), # 256,4,4 -> 256,4,4
            PatchEmbedding(embed_dim*8, embed_dim*16, patch_size=2), # 256,4,4 -> 512,2,2
            # ConvMixerStack(embed_dim*16, 4, kernel_size=2), # 512,2,2 -> 512,2,2
            PatchEmbedding(embed_dim*16, embed_dim*32, patch_size=2), # 512,2,2 -> 1024,1,1
            UnPatchEmbedding(embed_dim*32, embed_dim*16, patch_size=2), # 1024,1,1 -> 512,2,2
            # UnConvMixerStack(embed_dim*16, 4, kernel_size=2), # 512,2,2 -> 512,2,2
            UnPatchEmbedding(embed_dim*16, embed_dim*8, patch_size=2), # 512,2,2 -> 256,4,4
            UnConvMixerStack(embed_dim*8, 4, kernel_size=3), # 256,4,4 -> 256,4,4
            UnPatchEmbedding(embed_dim*8, embed_dim*4, patch_size=2), # 256,4,4 -> 128,8,8
            UnConvMixerStack(embed_dim*4, 4, kernel_size=5), # 128,8,8 -> 128,8,8
            UnPatchEmbedding(embed_dim*4, embed_dim*2, patch_size=2), # 128,8,8 -> 64,16,16
            UnConvMixerStack(embed_dim*2, 4, kernel_size=9), # 64,16,16 -> 64,16,16
            UnPatchEmbedding(embed_dim*2, embed_dim, patch_size=20), # 64,16,16 -> 32,320,320
            PostNet(embed_dim), # 32,320,320 -> 2,320,320
        )

    def forward(self, masked_kspace, col_mask, filter_masked: bool = False):
        feats   = self.prenet(masked_kspace, col_mask, filter_masked) # (N, C, H, M)
        output  = self.net(feats) # (N, 2, H, W)
        return output