import torch
import torch.nn as nn
import torch.nn.functional as F

"""
This model takes in a masked kspace tensor and a mask tensor and returns a denoised kspace tensor.
The forward function first generates a position tensor of shape (n_slices, 2, H, W), where the first channel is the row index and the second channel is the column index.
The position tensor is then embedded to a vector of size pos_embed_dim. (n_slices, 2, H, W) -> (n_slices, pos_embed_dim, H, W)
The masked kspace tensor is then pointwise convoluted to project the number of channels to pre_dims channels. (n_slices, 2, H, W) -> (n_slices, pre_dims, H, W)
The position embedding is then concatenated to the pre_conv output. (n_slices, pre_dims, H, W) -> (n_slices, pre_dims + pos_embed_dim, H, W)
The mask is repeated to match the shape of the pre_conv output. (1, 1, W, 1) -> (n_slices, 1, H, W)
The full mask is then concatenated to the pre_conv output. (n_slices, pre_dims + pos_embed_dim, H, W) -> (n_slices, pre_dims + pos_embed_dim + 1, H, W)
The pre_conv is then passed through block(s) of convmixer layers. (n_slices, pre_dims + pos_embed_dim + 1, H, W) -> (n_slices, conv_dims, H, W)
The tensor is then pointwise convoluted to project the number of channels to output channels. (n_slices, conv_dims, H, W) -> (n_slices, 2, H, W)
"""
def create_activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f'Unknown activation: {activation}')

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
                    create_activation(activation),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                create_activation(activation),
                nn.BatchNorm2d(dim)
            ) for _ in range(depth)]
        )


class SingleCoilKspaceConvmixerDenoiser(nn.Module):
    def __init__(
            self,
            patch_size: int = 8,
            conv_dims: int = 16,
            pos_embed_dim: int = 4,
            conv_blocks: int = 1,
            conv_kernel_size: int = 9,
            H: int = 320,
            W: int = 320,
            activation: str = 'relu',
    ):
        super().__init__()
        
        self.pos_embed_dim = pos_embed_dim
        self.conv_blocks = conv_blocks
        self.conv_kernel_size = conv_kernel_size
        
        # self.pre_conv = nn.Sequential(
        #     nn.Conv2d(2, pre_dims, kernel_size=1),
        #     # create_activation(activation),
        #     # nn.BatchNorm2d(pre_dims),
        # )
        self.pos_embed_H = nn.Embedding(H, pos_embed_dim)
        self.pos_embed_W = nn.Embedding(W, pos_embed_dim)

        self.patch = nn.Sequential(
            nn.Conv2d(2+pos_embed_dim+1, conv_dims, kernel_size=patch_size, stride=patch_size),
            create_activation(activation),
            nn.BatchNorm2d(conv_dims),
        )

        self.conv_mixer_stack = ConvMixerStack(
            dim=conv_dims,
            depth=conv_blocks,
            kernel_size=conv_kernel_size,
            activation=activation,
        )

        self.unpatch = nn.Sequential(
            nn.ConvTranspose2d(conv_dims, 2, kernel_size=patch_size, stride=patch_size),
            create_activation(activation),
            nn.BatchNorm2d(2),
        )

    def forward(self, masked_kspace, mask):
        # masked_kspace is of shape (n_slices, 2, H, W)
        # mask is of shape (1, 1, W, 1)
        n_slices = masked_kspace.shape[0]
        H = masked_kspace.shape[2]
        W = masked_kspace.shape[3]
        position = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij'), dim=0).to(masked_kspace.device) # (2, H, W)
        position_H = self.pos_embed_H(position[0, :, :]) # (H, W, pos_embed_dim)
        position_W = self.pos_embed_W(position[1, :, :]) # (H, W, pos_embed_dim)
        position_embedding = position_H + position_W # (H, W, pos_embed_dim)
        position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(n_slices, 1, 1, 1) # (n_slices, pos_embed_dim, H, W)

        mask = mask.permute(0, 3, 1, 2).repeat(n_slices, 1, H, 1) # (n_slices, 1, H, W)

        # pre_conv = self.pre_conv(masked_kspace) # (n_slices, pre_dims, H, W)
        pre_conv = torch.cat([masked_kspace, position_embedding, mask], dim=1) # (n_slices, 2 + pos_embed_dim + 1, H, W)

        patches = self.patch(pre_conv) # (n_slices, 2 + pos_embed_dim + 1, H, W) -> (n_slices, conv_dims, H / patch_size, W / patch_size)

        conv_mixer_output = self.conv_mixer_stack(patches) # (n_slices, conv_dims, H / patch_size, W / patch_size) 

        output = self.unpatch(conv_mixer_output) # (n_slices, 2, H, W)

        return output



