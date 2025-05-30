import torch
import torch.nn as nn
import torch.nn.functional as F
import fastmri
from models.test_model_1 import ConvMixerStack, UnConvMixerStack, PatchEmbedding, UnPatchEmbedding

"""
The columnwise masked transformer denoiser for singlecoil kspace data.
The forward function takes in a stack of kspace slices (n_slices, 2, H, W) and a mask (1, 1, W, 1), where M-many columns are not masked.

First, an integer position tensor is created that identifies the columns of the kspace data (1, 1, W).
The non-masked columns of the position tensor are then copied to a new tensor (1, 1, M) (filtered positions)
The non-masked columns of the kspace data are then copied to a new tensor (n_slices, 2, H, M) (filtered kspace)
The columns of the filtered kspace are then flattened vertically (n_slices, 2*H, M) where 2*H is the number of channels.
The mask is then flattened vertically (1, 1, W) (filtered mask)
The filtered kspace is then pointwise convoluted to project the number of channels to pre_dims channels. (n_slices, 2*H, M) -> (n_slices, pre_dims, M)
The filtered position tensor is then embedded to a vector of size pre_dims. (1, 1, M) -> (1, pre_dims, M)
The position embedding is then added to the pre_conv output. 
The pre_conv is then passed through pre_layers of pointwise convolutions and activations.
The pre_conv is then passed through a transformer encoder with num_heads heads and hidden_size hidden size. (n_slices, pre_dims, M) -> (n_slices, hidden_size, M)
A decoder input tensor is created with the same shape as the kspace data of hidden_size channels, where 
- the columns of the masked columns are set to the masked token
- the columns of the non-masked columns are set to the encoder output
The position embedding is then added to the decoder input.
The decoder input is then passed through a transformer decoder with num_heads heads and hidden_size hidden size. (n_slices, hidden_size, M) -> (n_slices, hidden_size, M)
The decoder output is then passed through a post_conv to reduce the number of channels to 2. (n_slices, hidden_size, M) -> (n_slices, 2, M)

"""

class PointwiseConstantChannel1DResNet(nn.Module):
    def __init__(
        self,
        n_channels,
        activation: str = 'relu',
        kernel_size: int = 1,
    ):
        super().__init__()

        def create_activation(activation):
            if activation == 'relu':
                return nn.ReLU()
            elif activation == 'gelu':
                return nn.GELU()
            else:
                raise ValueError(f'Unknown activation: {activation}')
            
        padding = kernel_size // 2

        self.layers = nn.Sequential(
            nn.Conv1d(n_channels, n_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(n_channels),
            create_activation(activation),
            nn.Conv1d(n_channels, n_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(n_channels),
            create_activation(activation),
        )

        self.activation = create_activation(activation)

    def forward(self, x):
        return self.activation(x + self.layers(x))

class IFFT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return fastmri.ifft2c(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    
class FFT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return fastmri.fft2c(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class SingleCoilKspaceColumnwiseMaskedTransformerDenoiser(nn.Module):
    def __init__(
            self,
            encoder_num_heads: int = 1,
            decoder_num_heads: int = 1,
            pre_dims: int = 16,
            pre_layers: int = 1,
            hidden_size: int = 128,
            activation: str = 'relu',
            kernel_size: int = 1,
            H: int = 320,
            W: int = 320,
            apply_pre_norm: bool = False,
            apply_dc: bool = False,
    ):
        super().__init__()
        self.encoder_num_heads = encoder_num_heads
        self.decoder_num_heads = decoder_num_heads
        self.pre_dims = pre_dims
        self.pre_layers = pre_layers
        self.hidden_size = hidden_size
        self.activation = activation
        self.H = H
        self.W = W
        self.apply_pre_norm = apply_pre_norm
        self.apply_dc = apply_dc

        # pointwise conv to go from kspace (2*H channels) to pre_dims channels
        self.pre_conv = nn.Conv1d(2*H, pre_dims, kernel_size=1)

        if apply_pre_norm:
            self.pre_norm = nn.BatchNorm1d(self.pre_dims, eps=1e-6)

        # pre_conv position embedding
        self.pre_conv_position_embedding = nn.Embedding(W, pre_dims)

        # pre_conv layers
        if pre_layers > 0:
            self.pre_conv_layers = nn.Sequential(
            *[PointwiseConstantChannel1DResNet(pre_dims, activation, kernel_size) for _ in range(pre_layers)]
            )
        else:
            self.pre_conv_layers = nn.Identity()

        # project pre_conv output to hidden_size
        if pre_dims != hidden_size:
            self.pre_conv_project = nn.Conv1d(pre_dims, hidden_size, kernel_size=1)
        else:
            self.pre_conv_project = nn.Identity()

        # encoder
        self.encoder = nn.MultiheadAttention(
            hidden_size,
            encoder_num_heads,
            batch_first=True,
        )

        # decoder
        self.decoder = nn.MultiheadAttention(
            hidden_size,
            decoder_num_heads,
            batch_first=True,
        )

        # decoder input position embedding
        self.decoder_input_position_embedding = nn.Embedding(W, hidden_size)

        # self.post_conv0 = nn.Sequential(
        #     nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
        #     nn.BatchNorm1d(hidden_size),
        #     nn.ReLU(),
        # )

        # post_conv to go from hidden_size channels to kspace (2*H channels)
        self.post_conv = nn.Conv1d(hidden_size, 2*H, kernel_size=1)

        self.rm = nn.Sequential(
            IFFT(),
            PatchEmbedding(2, 16, patch_size=16),
            ConvMixerStack(16, 4, kernel_size=9),
            UnConvMixerStack(16, 4, kernel_size=9),
            UnPatchEmbedding(16, 2, patch_size=16),
            FFT(),
        )
        
        # masked token of shape (hidden_size)
        self.masked_token = nn.Parameter(torch.randn(hidden_size))

    def forward(self, kspace, col_mask):
        # kspace is of shape (n_slices, 2, H, W)
        # mask is of shape (W)
        n_slices = kspace.shape[0]
        H = kspace.shape[2]
        W = kspace.shape[3]
        M = col_mask.sum()

        # create position tensor of shape (1, 1, W)
        position = torch.arange(W).unsqueeze(0).unsqueeze(0).to(kspace.device)

        # copy non-masked columns of position tensor to new tensor of shape (1, 1, M)
        filtered_positions = position[:, :, col_mask.squeeze()]

        # copy non-masked columns of kspace to new tensor of shape (n_slices, 2, H, M)
        filtered_kspace = kspace[:, :, :, col_mask.squeeze()]

        # flatten filtered kspace to shape (n_slices, 2*H, M)
        filtered_kspace = filtered_kspace.reshape(n_slices, 2*H, M)

        # pointwise conv to go from kspace (2*H channels) to pre_dims channels
        pre_conv = self.pre_conv(filtered_kspace) # (n_slices, 2*H, M) -> (n_slices, pre_dims, M)

        # pre_conv position embedding
        pre_conv_position_embedding = self.pre_conv_position_embedding(filtered_positions) # (1, 1, M) -> (1, 1, M, pre_dims)
        # (1, 1, M, pre_dims) -> (n_slices, 1, M, pre_dims)
        pre_conv_position_embedding = pre_conv_position_embedding.repeat(n_slices, 1, 1, 1)
        pre_conv_position_embedding = pre_conv_position_embedding.squeeze(1).permute(0, 2, 1) # (n_slices, M, pre_dims)

        # add pre_conv position embedding to pre_conv output
        pre_conv = pre_conv + pre_conv_position_embedding # (n_slices, pre_dims, M)
        if self.apply_pre_norm:
            pre_conv = self.pre_norm(pre_conv) # (n_slices, pre_dims, M)

        # pre_conv layers
        pre_conv = self.pre_conv_layers(pre_conv) # (n_slices, pre_dims, M) -> (n_slices, pre_dims, M)

        # project pre_conv output to hidden_size
        pre_conv = self.pre_conv_project(pre_conv) # (n_slices, pre_dims, M) -> (n_slices, hidden_size, M)
        pre_conv = pre_conv.permute(0, 2, 1) # (n_slices, hidden_size, M) -> (n_slices, M, hidden_size)

        # encoder n_slices(batch_size), M (seq_len), hidden_size (embed_dim)
        encoder_output = self.encoder(pre_conv, pre_conv, pre_conv)[0]

        # decoder input n_slices(batch_size), M (seq_len), hidden_size (embed_dim)
        decoder_input = torch.zeros(n_slices, W, self.hidden_size).to(kspace.device) # (n_slices, W, hidden_size)
        decoder_input[:, col_mask.squeeze()] = encoder_output
        # self.masked_token is (hidden_size)
        masked_tokens = self.masked_token.unsqueeze(0).unsqueeze(0).repeat(n_slices, W - M, 1) # (n_slices, W - M, hidden_size)
        decoder_input[:, ~col_mask.squeeze()] = masked_tokens

        decoder_input_position_embedding = self.decoder_input_position_embedding(position) # (1, 1, W, hidden_size)
        decoder_input_position_embedding = decoder_input_position_embedding.repeat(n_slices, 1, 1, 1) # (n_slices, 1, W, hidden_size)
        decoder_input_position_embedding = decoder_input_position_embedding.squeeze(1)#.permute(0, 2, 1) # (n_slices, W, hidden_size)
        decoder_input[:, ~col_mask.squeeze()]  = decoder_input[:, ~col_mask.squeeze()]  + decoder_input_position_embedding[:, ~col_mask.squeeze()]

        decoder_output = self.decoder(decoder_input, decoder_input, decoder_input)[0] # (n_slices, W, hidden_size) -> (n_slices, W, hidden_size)

        decoder_output = decoder_output.permute(0, 2, 1) # (n_slices, hidden_size, W)

        # decoder_output = self.post_conv0(decoder_output)

        output_masked = self.post_conv(decoder_output) # (n_slices, hidden_size, W) -> (n_slices, 2*H, W)
        output_masked = output_masked.reshape(n_slices, 2, H, W) # (n_slices, 2*H, W) -> (n_slices, 2, H, W)

        # output = torch.zeros(n_slices, 2, H, W).to(kspace.device) # (n_slices, 2, H, W)
        # output[:, :, :, ~mask.squeeze()] = output_masked[:, :, :, ~mask.squeeze()]
        # output[:, :, :, mask.squeeze()] = kspace[:, :, :, mask.squeeze()]

        if self.apply_dc:
            mask4d  = col_mask.view(1, 1, 1, W)
            output = mask4d * kspace + (~mask4d) * output_masked
        else:
            output = output_masked

        output = self.rm(output)

        if self.apply_dc:
            output = mask4d * kspace + (~mask4d) * output

        return output
