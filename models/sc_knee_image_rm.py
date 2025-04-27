import torch
import torch.nn as nn
import torch.nn.functional as F
import fastmri
from models.test_model_1 import ConvMixerStack, UnConvMixerStack, PatchEmbedding, UnPatchEmbedding

class IFFT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        with torch.amp.autocast(x.device.type, enabled=False):
            x = x.float()
            img = fastmri.ifft2c(x.permute(0, 2, 3, 1))
        return img.permute(0, 3, 1, 2)
    
class FFT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        with torch.amp.autocast(x.device.type, enabled=False):
            x = x.float()
            k = fastmri.fft2c(x.permute(0, 2, 3, 1))
        return k.permute(0, 3, 1, 2)
    
class PatchEmbeddingOverlap(nn.Module):
    def __init__(self, in_ch, dim, patch, stride=None):
        super().__init__()
        s = stride or patch                     # s = 4 in your call
        p = (patch - s) // 2                   # for patch=8, s=4  ->  p = 2
        self.prefilter = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch, bias=False)
        with torch.no_grad():
            self.prefilter.weight.fill_(1 / 9)
        self.embed = nn.Conv2d(in_ch, dim, patch, s, padding=p)

    def forward(self, x):
        return self.embed(self.prefilter(x))

class UnPatchEmbeddingPixelShuffle(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, patch_size: int, stride: int | None = None):
        super().__init__()
        s = stride or patch_size               # s = 4 here
        assert (patch_size % s) == 0
        scale = s                              # ← upsample by the *same* factor
        self.proj    = nn.Conv2d(in_ch, in_ch * scale * scale, 1)
        self.shuffle = nn.PixelShuffle(scale)
        self.out     = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        return self.out(self.shuffle(self.proj(x)))

class RM1DCStack(nn.Module):
    def __init__(
            self,
            dim:int,
            depth:int,
            k:int,
            patch_size:int,
            ):
        super().__init__()
        self.rm = nn.Sequential(
            PatchEmbeddingOverlap(2, dim, patch_size, stride=patch_size//2), # (N, 2, H, W) -> (N, dim, H/ps*2, W/ps*2)
            ConvMixerStack(dim, depth, kernel_size=k), # (N, dim, H/ps*2, W/ps*2) -> (N, dim, H/ps*2, W/ps*2)
            UnConvMixerStack(dim, depth, kernel_size=k),
            # UnPatchEmbedding(dim, dim//2, patch_size=patch_size),
            # nn.Conv2d(dim//2, 2, kernel_size=patch_size+1, padding='same'),
            UnPatchEmbeddingPixelShuffle(dim, 2, patch_size=patch_size, stride=patch_size//2),
            FFT(),
        )
        self.ifft = IFFT()

    def forward(self, kspace_masked, mask4d):
        img = self.ifft(kspace_masked) # (N, 2, H, W)
        kspace = self.rm(img) # (N, 2, H, W)
        return mask4d * kspace_masked + (~mask4d) * kspace

class RM1Multi(nn.Module):
    def __init__(
            self,
            dim:int = 64,
            depth:int = 1,
            k:int = 9,
            patch_size:int = 8,
            n_layers:int = 4,
            ):
        super().__init__()
        self.rms = nn.ModuleList([RM1DCStack(dim, depth, k, patch_size) for _ in range(n_layers)])

        self.ifft = IFFT()

    def forward(self, kspace_masked, mask4d):
        kspace = kspace_masked
        for rm in self.rms:
            kspace = rm(kspace, mask4d)
        img = self.ifft(kspace)
        return fastmri.complex_abs(img.permute(0, 2, 3, 1))



class RM1(nn.Module):
    def __init__(self):
        super().__init__()
        dim = 128
        depth = 6
        k = 9
        patch_size = 8

        self.rm = nn.Sequential(
            IFFT(),
            PatchEmbeddingOverlap(2, dim, patch_size, stride=patch_size//2), # (N, 2, H, W) -> (N, dim, H/ps*2, W/ps*2)
            ConvMixerStack(dim, depth, kernel_size=k), # (N, dim, H/ps*2, W/ps*2) -> (N, dim, H/ps*2, W/ps*2)
            UnConvMixerStack(dim, depth, kernel_size=k),
            # UnPatchEmbedding(dim, dim//2, patch_size=patch_size),
            # nn.Conv2d(dim//2, 2, kernel_size=patch_size+1, padding='same'),
            UnPatchEmbeddingPixelShuffle(dim, 2, patch_size=patch_size, stride=patch_size//2),
            FFT(),
        )
        dim2 = 64
        depth2 = 2
        k2 = 9
        patch_size2 = 8
        # self.rm2 = nn.Sequential(
        #     IFFT(),
        #     PatchEmbeddingOverlap(2, dim2, patch_size2, stride=patch_size2//2), # (N, 2, H, W) -> (N, dim2, H/ps*2, W/ps*2)
        #     ConvMixerStack(dim2, depth2, kernel_size=k2), # (N, dim2, H/ps*2, W/ps*2) -> (N, dim2, H/ps*2, W/ps*2)
        #     UnConvMixerStack(dim2, depth2, kernel_size=k2),
        #     # UnPatchEmbedding(dim2, dim2//2, patch_size2=patch_size2),
        #     # nn.Conv2d(dim2//2, 2, kernel_size=patch_size2+1, padding='same'),
        #     UnPatchEmbeddingPixelShuffle(dim2, 2, patch_size2, stride=patch_size2//2),
        #     FFT(),
        # )
        self.ifft = IFFT()

    def forward(self, kspace_masked, col_mask):
        n_iters = 4
        mask4d  = col_mask.view(1, 1, 1, col_mask.shape[-1])
        x = self.rm(kspace_masked)
        # for _ in range(n_iters-1):
        #     x = mask4d * kspace_masked + (~mask4d) * x
        #     x = self.rm2(x)
        x = mask4d * kspace_masked + (~mask4d) * x
        x = self.ifft(x)
        x = fastmri.complex_abs(x.permute(0, 2, 3, 1))
        return x