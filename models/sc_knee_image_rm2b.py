import torch
import torch.nn as nn
import torch.nn.functional as F
import fastmri
from models.test_model_1 import ConvMixerStack, UnConvMixerStack, PatchEmbedding, UnPatchEmbedding
from models.masked_kspace_embedding import MaskedKspaceColumnEmbedding

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

class RM2b(nn.Module):
    def __init__(
            self,
            H: int = 320,
            W: int = 320,
            autoencoder_dim: int = 32,
            autoencoder_depth: int = 4,
            autoencoder_kernel_size: int = 9,
            autoencoder_patch_size: int = 8,
            encoder_dim: int = 64,
            encoder_depth: int = 4,
            encoder_kernel_size: int = 9,
            encoder_patch_size: int = 8,
            kspace_embedding_dim: int = 512,
            transformer_hidden_size: int = 256,
            transformer_num_heads: int = 16,
            transformer_num_layers: int = 1,
            apply_final_dc: bool = False,
            ):
        super().__init__()
        self.H = H
        self.W = W
        self.autoencoder_dim = autoencoder_dim
        self.autoencoder_depth = autoencoder_depth
        self.autoencoder_kernel_size = autoencoder_kernel_size
        self.autoencoder_patch_size = autoencoder_patch_size
        self.encoder_dim = encoder_dim
        self.encoder_depth = encoder_depth
        self.encoder_kernel_size = encoder_kernel_size
        self.encoder_patch_size = encoder_patch_size
        self.kspace_embedding_dim = kspace_embedding_dim
        self.transformer_hidden_size = transformer_hidden_size
        self.transformer_num_heads = transformer_num_heads
        self.transformer_num_layers = transformer_num_layers
        self.apply_final_dc = apply_final_dc


        # step 1 - use a conv mixer autoencoder to denoise the image domain
        # step 2 - FFT to kspace domain and apply data consistency
        # step 3 - in parallel:
        #   - IFFT to image domain and use a conv mixer to encode the image, then add position embedding and flatten into 
        #   - embed the kspace into column vectors
        # step 4 - fan in image domain encoding (query) and kspace domain encoding (key) into a transformer
        # step 5 - decode the transformer output into an image

        self.autoencoder = nn.Sequential(
            PatchEmbeddingOverlap(2, autoencoder_dim, autoencoder_patch_size, stride=autoencoder_patch_size//2),
            ConvMixerStack(autoencoder_dim, autoencoder_depth, kernel_size=autoencoder_kernel_size),
            UnConvMixerStack(autoencoder_dim, autoencoder_depth, kernel_size=autoencoder_kernel_size),
            UnPatchEmbeddingPixelShuffle(autoencoder_dim, 2, autoencoder_patch_size, stride=autoencoder_patch_size//2),
        )

        self.kspace_embedding = MaskedKspaceColumnEmbedding(dim1=kspace_embedding_dim, H=H, W=W)

        self.encoder = nn.Sequential(
            PatchEmbeddingOverlap(2, encoder_dim, encoder_patch_size, stride=encoder_patch_size//2),
            ConvMixerStack(encoder_dim, encoder_depth, kernel_size=encoder_kernel_size),
        )

        self.decoder = nn.Sequential(
            UnConvMixerStack(encoder_dim, encoder_depth, kernel_size=encoder_kernel_size),
            UnPatchEmbeddingPixelShuffle(encoder_dim, 2, encoder_patch_size, stride=encoder_patch_size//2),
        )
        self.ifft = IFFT()
        self.fft = FFT()

    def forward(self, kspace_masked, col_mask):
        n_iters = 1
        mask4d  = col_mask.view(1, 1, 1, col_mask.shape[-1])
        image1_complex = self.ifft(kspace_masked) # (N, 2, H, W)
        image1_denoised = self.autoencoder(image1_complex) # (N, 2, H, W)
        image = image1_denoised
        for _ in range(n_iters):
            kspace1 = self.fft(image) # (N, 2, H, W)
            kspace1_dc = mask4d * kspace_masked + (~mask4d) * kspace1 # (N, 2, H, W)
            image2_complex = self.ifft(kspace1_dc) # (N, 2, H, W)
            image2_encoded = self.encoder(image2_complex) # (N, C_enc, H/ps,W/ps)

            image3_denoised = self.decoder(image2_encoded) # (N, 2, H, W)
            image = image3_denoised
        if self.apply_final_dc:
            kspace2 = self.fft(image) # (N, 2, H, W)
            kspace2_dc = mask4d * kspace_masked + (~mask4d) * kspace2 # (N, 2, H, W)
            image = self.ifft(kspace2_dc) # (N, 2, H, W)
        return fastmri.complex_abs(image.permute(0, 2, 3, 1))
