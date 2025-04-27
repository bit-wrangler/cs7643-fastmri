import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as SN

def depthwise_pointwise_block(in_c, out_c, k=3, s=1):
    return nn.Sequential(
        SN(nn.Conv2d(in_c, in_c, k, s, k//2, groups=in_c, bias=False)),
        nn.LeakyReLU(0.2, inplace=True),
        SN(nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)),
        nn.LeakyReLU(0.2, inplace=True)
    )

class Residual(nn.Module):
    def __init__(
            self,
            fn,
    ):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixerStack(
            dim: int,
            depth: int,
            kernel_size: int = 9,
            activation: str = 'relu',
    ):
        return nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            ) for _ in range(depth)]
        )

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=1, channel_mult=32):
        super().__init__()
        cm = channel_mult
        self.net = nn.Sequential(
            SN(nn.Conv2d(in_ch, cm, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            depthwise_pointwise_block(cm,   cm*2, 3, 2),
            depthwise_pointwise_block(cm*2, cm*4, 3, 2),
            depthwise_pointwise_block(cm*4, cm*8, 3, 2),
            depthwise_pointwise_block(cm*8, cm*8, 3, 2),
            depthwise_pointwise_block(cm*8, cm*8, 3, 2),

            SN(nn.Conv2d(cm*8, 1, 3, 1, 1))
        )

    def forward(self, x):
        return self.net(x)
    
class PairDiscriminator(nn.Module):
    def __init__(self, in_ch=1, ch=64):
        super().__init__()
        self.shared = nn.Sequential(
            SN(nn.Conv2d(in_ch, ch, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            SN(nn.Conv2d(ch,  ch*2, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            SN(nn.Conv2d(ch*2,ch*4,4, 2, 1)), nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.head = SN(nn.Linear(ch*4, 1))

    def forward(self, img_a, img_b):
        fa = self.shared(img_a).view(img_a.size(0), -1)
        fb = self.shared(img_b).view(img_b.size(0), -1)
        logit = self.head(fa - fb)
        return logit
    
class PairConvMixerDiscriminator(nn.Module):
    """
    Discriminator that scores an (img_real, img_fake) pair.
    img_a, img_b: (B, C, H, W).  Scores are returned as (B, 1, H', W').
    """
    def __init__(
        self,
        in_ch: int = 1,
        dim: int = 64,
        depth: int = 8,
        patch: int = 16,       # stride / kernel for the patch embed
        ksize: int = 9
    ):
        super().__init__()

        # Patch embedding (concat pair → 2*in_ch)
        self.embed = SN(nn.Conv2d(in_ch * 2, dim, kernel_size=patch,
                                  stride=patch))

        # ConvMixer backbone
        self.backbone = ConvMixerStack(dim, depth, kernel_size=ksize)

        # 1×1 head → patch map (or scalar if you .mean())
        self.head = SN(nn.Conv2d(dim, 1, kernel_size=1))

    def forward(self, img_a, img_b):
        x = torch.cat([img_a, img_b], dim=1)   # (B, 2C, H, W)
        x = self.embed(x)                      # (B, dim, H/patch, W/patch)
        x = self.backbone(x)                   # ConvMixer
        return self.head(x)                    # (B, 1, H', W')