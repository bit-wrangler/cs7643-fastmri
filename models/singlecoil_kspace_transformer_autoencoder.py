import torch
import torch.nn as nn
import torch.nn.functional as F

"""
A transformer-based autoencoder for singlecoil kspace data.

First, an integer position tensor is created that identifies the columns of the kspace data (1, 1, W).
The columns of the kspace data are then flattened vertically (n_slices, 2*H, W) where 2*H is the number of channels.
The kspace data is then pointwise convoluted to project the number of channels to pre_dims channels. (n_slices, 2*H, W) -> (n_slices, pre_dims, W)
The position tensor is then embedded to a vector of size pre_dims. (1, 1, W) -> (1, pre_dims, W)
The position embedding is then added to the pre_transformer output. 
A summary token is then appended to the pre_transformer output. (n_slices, pre_dims, W) -> (n_slices, pre_dims, W+1)
The pre_transformer is then passed through a transformer encoder with num_heads heads and hidden_size hidden size. (n_slices, pre_dims, W+1) -> (n_slices, hidden_size, W+1)
A decoder input tensor is created with the shape (n_slices, hidden_size, W+1).
- The first column of the decoder input is set to the summary token
- The remaining columns of the decoder input are set to the blank token + position embedding
The decoder input is then passed through a transformer decoder with num_heads heads and hidden_size hidden size. (n_slices, hidden_size, W+1) -> (n_slices, hidden_size, W+1)
The docoder output (sans summary token) is then passed through a post_transformer to reduce the number of channels to 2*H. (n_slices, hidden_size, W) -> (n_slices, 2*H, W)
"""

class SingleCoilKspaceTransformerAutoencoder(nn.Module):
    def __init__(
            self,
            encoder_num_heads: int = 1,
            decoder_num_heads: int = 1,
            transformer_hidden_size: int = 256,
            W: int = 320,
            H: int = 320,
            ):
        super().__init__()
        self.encoder_num_heads = encoder_num_heads
        self.decoder_num_heads = decoder_num_heads
        self.transformer_hidden_size = transformer_hidden_size
        self.W = W
        self.H = H

        # pre_transformer to go from kspace (2*H channels) to transformer_hidden_size channels
        self.pre_transformer = nn.Conv1d(2*H, transformer_hidden_size, kernel_size=1)

        # pre_transformer position embedding
        self.pre_transformer_position_embedding = nn.Embedding(W, transformer_hidden_size)

        # summary token of shape (transformer_hidden_size)
        self.summary_token = nn.Parameter(torch.randn(transformer_hidden_size))

        # encoder
        self.encoder = nn.MultiheadAttention(
            transformer_hidden_size,
            encoder_num_heads,
            batch_first=True,
        )

        # decoder input position embedding
        self.decoder_input_position_embedding = nn.Embedding(W, transformer_hidden_size)

        # decoder blank token of shape (transformer_hidden_size)
        self.blank_token = nn.Parameter(torch.randn(transformer_hidden_size))

        # decoder
        self.decoder = nn.MultiheadAttention(
            transformer_hidden_size,
            decoder_num_heads,
            batch_first=True,
        )

        # post_transformer to go from transformer_hidden_size channels to kspace (2*H channels)
        self.post_transformer = nn.Conv1d(transformer_hidden_size, 2*H, kernel_size=1)

    def forward(self, kspace):
        # kspace is of shape (n_slices, 2, H, W)

        n_slices = kspace.shape[0]
        H = kspace.shape[2]
        W = kspace.shape[3]

        # create position tensor of shape (1, 1, W)
        position = torch.arange(W).unsqueeze(0).unsqueeze(0).to(kspace.device)

        # pointwise conv to go from kspace (2*H channels) to transformer_hidden_size channels
        pre_transformer = self.pre_transformer(kspace.reshape(n_slices, 2*H, W)) # (n_slices, 2*H, W) -> (n_slices, transformer_hidden_size, W)

        # pre_transformer position embedding
        pre_transformer_position_embedding = self.pre_transformer_position_embedding(position) # (1, 1, W) -> (1,1, W, transformer_hidden_size)
        pre_transformer_position_embedding = pre_transformer_position_embedding.repeat(n_slices, 1, 1, 1) # (n_slices, 1, W, transformer_hidden_size)
        pre_transformer_position_embedding = pre_transformer_position_embedding.squeeze(1).permute(0, 2, 1) # (n_slices, W, transformer_hidden_size)

        # add pre_transformer position embedding to pre_transformer output
        pre_transformer = pre_transformer + pre_transformer_position_embedding

        # add summary token to pre_transformer output
        summary_token = self.summary_token.unsqueeze(0).unsqueeze(0).repeat(n_slices, 1, 1) # (n_slices, 1, transformer_hidden_size)
        pre_transformer = pre_transformer.permute(0, 2, 1) # (n_slices, W+1, transformer_hidden_size)

        pre_transformer = torch.cat([summary_token, pre_transformer], dim=1) # (n_slices, W+1, transformer_hidden_size)

        # encoder n_slices(batch_size), W+1 (seq_len), transformer_hidden_size (embed_dim)
        encoder_output = self.encoder(pre_transformer, pre_transformer, pre_transformer)[0] # (n_slices, W+1, transformer_hidden_size)

        # decoder input n_slices(batch_size), W+1 (seq_len), transformer_hidden_size (embed_dim)
        decoder_input = torch.zeros(n_slices, W+1, self.transformer_hidden_size).to(kspace.device) # (n_slices, W+1, transformer_hidden_size)
        decoder_input[:, 0, :] = encoder_output[:, 0, :] # (n_slices, 1, transformer_hidden_size)
        decoder_input[:, 1:, :] = self.blank_token.unsqueeze(0).unsqueeze(0).repeat(n_slices, W, 1) # (n_slices, W, transformer_hidden_size)

        decoder_input_position_embedding = self.decoder_input_position_embedding(position) # (1, 1, W, transformer_hidden_size)
        decoder_input_position_embedding = decoder_input_position_embedding.repeat(n_slices, 1, 1, 1) # (n_slices, 1, W, transformer_hidden_size)
        decoder_input_position_embedding = decoder_input_position_embedding.squeeze(1)#.permute(0, 2, 1) # (n_slices, W, transformer_hidden_size)
        decoder_input[:, 1:, :] = decoder_input[:, 1:, :] + decoder_input_position_embedding

        decoder_output = self.decoder(decoder_input, decoder_input, decoder_input)[0] # (n_slices, W+1, transformer_hidden_size) -> (n_slices, W+1, transformer_hidden_size)
        decoder_output = decoder_output[:, 1:, :] # (n_slices, W, transformer_hidden_size)
        decoder_output = decoder_output.permute(0, 2, 1) # (n_slices, transformer_hidden_size, W)

        output = self.post_transformer(decoder_output) # (n_slices, transformer_hidden_size, W) -> (n_slices, 2*H, W)
        output = output.reshape(n_slices, 2, H, W) # (n_slices, 2*H, W) -> (n_slices, 2, H, W)

        return output


