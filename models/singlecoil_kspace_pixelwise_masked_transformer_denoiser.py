import torch
import torch.nn as nn
import torch.nn.functional as F

"""
The pixelwise masked transformer denoiser for singlecoil kspace data.
The forward function takes in a stack of kspace slices (n_slices, 2, 640, 372) and a mask (1, 1, 372, 1), where M-many columns are not masked.
First, a 2 channel integer position tensor is created that matches the shape of the kspace data (2, 640, 372).
The non-masked columns of the position tensor are then copied to a new tensor (2, 640, M)
The non-masked columns of the kspace data are then copied to a new tensor (n_slices, 2, 640, M)
The non-masked columns of the kspace data are then pointwise convoluted to increase the number of channels to pre_dims. (n_slices, 2, 640, M) -> (n_slices, pre_dims, 640, M)
The non-masked columns of the position tensor is then embedded to a vector of size pre_dims. (2, 640, M) -> (pre_dims, 640, M)
The position embedding is then added to the pre_conv output. 
The pre_conv is then passed through pre_layers of pixelwise convolutions and activations.
The pre_conv is then passed through a transformer encoder with num_heads heads and hidden_size hidden size. (n_slices, pre_dims, 640, M) -> (n_slices, hidden_size, 640, M)
A decoder input tensor is created with the same shape as the kspace data of hidden_size channels, where 
- the pixels of the masked columns are set to the masked token
- the pixels of the non-masked columns are set to the encoder output
The position embedding is then added to the decoder input.
The decoder input is then passed through a transformer decoder with num_heads heads and hidden_size hidden size. (n_slices, hidden_size, 640, 372) -> (n_slices, hidden_size, 640, 372)
The decoder output is then passed through a post_conv to reduce the number of channels to 2. (n_slices, hidden_size, 640, 372) -> (n_slices, 2, 640, 372)
"""

class PixelwiseConstantChannelResNet(nn.Module):
    def __init__(
        self,
        n_channels,
        activation: str = 'relu',
    ):
        super().__init__()

        def create_activation(activation):
            if activation == 'relu':
                return nn.ReLU()
            elif activation == 'gelu':
                return nn.GELU()
            else:
                raise ValueError(f'Unknown activation: {activation}')

        self.layers = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=1),
            nn.BatchNorm2d(n_channels),
            create_activation(activation),
            nn.Conv2d(n_channels, n_channels, kernel_size=1),
            nn.BatchNorm2d(n_channels),
            create_activation(activation),
        )

        self.activation = create_activation(activation)

    def forward(self, x):
        return self.activation(x + self.layers(x))


    

class SingleCoilKspacePixelwiseMaskedTransformerDenoiser(nn.Module):
    def __init__(
            self,
            encoder_num_heads: int = 1,
            decoder_num_heads: int = 1,
            pre_dims: int = 16,
            pre_layers: int = 1,
            hidden_size: int = 128,
            activation: str = 'relu',
            H: int = 640,
            W: int = 372,
            
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

        # pixelwise conv to go from kspace (2 channels) to pre_dims channels
        self.pre_conv = nn.Conv2d(2, pre_dims, kernel_size=1)

        # pre_conv position embedding
        self.pre_conv_position_embedding_H = nn.Embedding(H, pre_dims)
        self.pre_conv_position_embedding_W = nn.Embedding(W, pre_dims)

        # pre_conv layers
        if pre_layers > 0:
            self.pre_conv_layers = nn.Sequential(
            *[PixelwiseConstantChannelResNet(pre_dims, activation) for _ in range(pre_layers)]
            )
        else:
            self.pre_conv_layers = nn.Identity()

        # project pre_conv output to hidden_size
        self.pre_conv_project = nn.Conv2d(pre_dims, hidden_size, kernel_size=1)

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
        self.decoder_input_position_embedding_H = nn.Embedding(H, hidden_size)
        self.decoder_input_position_embedding_W = nn.Embedding(W, hidden_size)

        # post_conv to go from hidden_size channels to kspace (2 channels)
        self.post_conv = nn.Conv2d(hidden_size, 2, kernel_size=1)
        
        # masked token of shape (hidden_size, 1, 1)
        self.masked_token = nn.Parameter(torch.randn(hidden_size, 1, 1))

    def forward(self, kspace, mask):
        # kspace is of shape (n_slices, 2, H, W)
        # mask is of shape (1, 1, W, 1)
        n_slices = kspace.shape[0]
        H = kspace.shape[2]
        W = kspace.shape[3]
        M = mask.sum()

        # create position tensor of shape (2, H, W)
        position = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij'), dim=0)
        position = position.to(kspace.device)

        # copy non-masked columns of position tensor to new tensor of shape (2, H, M)
        position_masked = position[:, :, mask.squeeze()]

        # copy non-masked columns of kspace to new tensor of shape (n_slices, 2, H, M)
        kspace_masked = kspace[:, :, :, mask.squeeze()]

        # pointwise conv to go from kspace (2 channels) to pre_dims channels
        pre_conv = self.pre_conv(kspace_masked)

        # pre_conv position embedding
        pre_conv_position_embedding_H = self.pre_conv_position_embedding_H(position_masked[0, :, :])
        pre_conv_position_embedding_W = self.pre_conv_position_embedding_W(position_masked[1, :, :])
        pre_conv_position_embedding = pre_conv_position_embedding_H + pre_conv_position_embedding_W
        pre_conv_position_embedding = pre_conv_position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(n_slices, 1, 1, 1)

        # add pre_conv position embedding to pre_conv output
        pre_conv = pre_conv + pre_conv_position_embedding

        # pre_conv layers
        pre_conv = self.pre_conv_layers(pre_conv)

        # project pre_conv output to hidden_size
        pre_conv = self.pre_conv_project(pre_conv)

        # encoder
        pre_conv = pre_conv.permute(0, 2, 3, 1).reshape(n_slices, H * M, self.hidden_size)
        encoder_output = self.encoder(pre_conv, pre_conv, pre_conv)[0]

        # decoder input
        decoder_input = torch.zeros(n_slices, H, W, self.hidden_size).to(kspace.device)
        decoder_input[:, :, mask.squeeze()] = encoder_output
        decoder_input[:, :, ~mask.squeeze()] = self.masked_token
        decoder_input = decoder_input.permute(0, 2, 3, 1).reshape(n_slices, H * W, self.hidden_size)
        decoder_input_position_embedding_H = self.decoder_input_position_embedding_H(position[0, :, :])
        decoder_input_position_embedding_W = self.decoder_input_position_embedding_W(position[1, :, :])
        decoder_input_position_embedding = decoder_input_position_embedding_H + decoder_input_position_embedding_W
        decoder_input_position_embedding = decoder_input_position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(n_slices, 1, 1, 1)
        decoder_input = decoder_input + decoder_input_position_embedding

        # decoder
        decoder_output = self.decoder(decoder_input, decoder_input, decoder_input)[0]

        # post_conv to go from hidden_size channels to kspace (2 channels)
        decoder_output = decoder_output.reshape(n_slices, H, W, self.hidden_size).permute(0, 3, 1, 2)
        output = self.post_conv(decoder_output)

        return output

