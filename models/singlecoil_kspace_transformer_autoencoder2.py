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

class PointwiseConstantChannel1DResNet(nn.Module):
    def __init__(
        self,
        n_channels,
        activation: str = 'gelu',
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
            nn.Conv1d(n_channels, n_channels, kernel_size=1),
            nn.BatchNorm1d(n_channels),
            create_activation(activation),
            nn.Conv1d(n_channels, n_channels, kernel_size=1),
            nn.BatchNorm1d(n_channels),
            create_activation(activation),
        )

        self.activation = create_activation(activation)

    def forward(self, x):
        return self.activation(x + self.layers(x))

class SingleCoilKspaceTransformerAutoencoder2(nn.Module):
    def __init__(
            self,
            encoder_num_heads: int = 1,
            n_encoder_layers: int = 1,
            decoder_num_heads: int = 1,
            n_decoder_layers: int = 1,
            transformer_hidden_size: int = 256,
            ff_dim: int = 256,
            n_summary_tokens: int = 1,
            W: int = 320,
            H: int = 320,
            ):
        super().__init__()
        self.encoder_num_heads = encoder_num_heads
        self.n_encoder_layers = n_encoder_layers
        self.decoder_num_heads = decoder_num_heads
        self.n_decoder_layers = n_decoder_layers
        self.transformer_hidden_size = transformer_hidden_size
        self.W = W
        self.H = H
        self.n_summary_tokens = n_summary_tokens

        # pre_transformer to go from kspace (2*H channels) to transformer_hidden_size channels
        self.pre_transformer = nn.Conv1d(2*H, transformer_hidden_size, kernel_size=1)

        # pre_transformer position embedding
        self.pre_transformer_position_embedding = nn.Embedding(W, transformer_hidden_size)

        # summary token of shape (transformer_hidden_size)
        self.summary_token = nn.Parameter(torch.randn(n_summary_tokens, transformer_hidden_size))

        # encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                transformer_hidden_size,
                encoder_num_heads,
                batch_first=True,
                dim_feedforward=ff_dim,
            ),
            n_encoder_layers,
        )

        # decoder input position embedding
        self.decoder_input_position_embedding = nn.Embedding(W, transformer_hidden_size)
        # decoder
        # self.decoder = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(
        #         transformer_hidden_size,
        #         decoder_num_heads,
        #         batch_first=True,
        #         dim_feedforward=ff_dim,
        #     ),
        #     n_decoder_layers,
        # )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                transformer_hidden_size,
                decoder_num_heads,
                batch_first=True,
                dim_feedforward=ff_dim,
            ),
            n_decoder_layers,
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
        summary_token = self.summary_token.unsqueeze(0).repeat(n_slices, 1, 1) # (n_slices, n_summary_tokens, transformer_hidden_size)
        pre_transformer = pre_transformer.permute(0, 2, 1) # (n_slices, W+n_summary_tokens, transformer_hidden_size)

        pre_transformer = torch.cat([summary_token, pre_transformer], dim=1) # (n_slices, W+nummary_tokens, transformer_hidden_size)

        # encoder n_slices(batch_size), W+n_summary_tokens (seq_len), transformer_hidden_size (embed_dim)
        # encoder_output = self.encoder(pre_transformer, pre_transformer, pre_transformer)[0] # (n_slices, W+n_ummary_tokens, transformer_hidden_size)
        encoder_output = self.encoder(pre_transformer) # (n_slices, W+n_ummary_tokens, transformer_hidden_size)

        # decoder input n_slices(batch_size), W+nummary_tokens (seq_len), transformer_hidden_size (embed_dim)
        # decoder_input = torch.zeros(n_slices, W+self.n_summary_tokens, self.transformer_hidden_size).to(kspace.device) # (n_slices, W+n_summary_tokens, transformer_hidden_size)
        # decoder_input[:, :self.n_summary_tokens, :] = encoder_output[:, self.n_summary_tokens, :] # (n_slices, 1, transformer_hidden_size)

        # decoder_input_position_embedding = self.decoder_input_position_embedding(position) # (1, 1, W, transformer_hidden_size)
        # decoder_input_position_embedding = decoder_input_position_embedding.repeat(n_slices, 1, 1, 1) # (n_slices, 1, W, transformer_hidden_size)
        # decoder_input_position_embedding = decoder_input_position_embedding.squeeze(1)#.permute(0, 2, 1) # (n_slices, W, transformer_hidden_size)
        # decoder_input[:, self.n_summary_tokens:, :] = decoder_input_position_embedding

        decoder_memory = encoder_output[:, :self.n_summary_tokens, :] # (n_slices, n_summary_tokens, transformer_hidden_size)
        decoder_target = self.decoder_input_position_embedding(position) # (1, 1, W, transformer_hidden_size)
        decoder_target = decoder_target.repeat(n_slices, 1, 1, 1) # (n_slices, 1, W, transformer_hidden_size)
        decoder_target = decoder_target.squeeze(1) # (n_slices, W, transformer_hidden_size)

        # decoder_output = self.decoder(decoder_input, decoder_input, decoder_input)[0] # (n_slices, W+n_s ummary_tokens, transformer_hidden_size) -> (n_slices, W+n_ummary_tokens, transformer_hidden_size)
        # decoder_output = self.decoder(decoder_input) # (n_slices, W+n_s ummary_tokens, transformer_hidden_size) -> (n_slices, W+n_ummary_tokens, transformer_hidden_size)
        # decoder_output = decoder_output[:, self.n_summary_tokens:, :] # (n_slices, W, transformer_hidden_size)
        # decoder_output = decoder_output.permute(0, 2, 1) # (n_slices, transformer_hidden_size, W)
        decoder_output = self.decoder(decoder_target, decoder_memory) # (n_slices, W, transformer_hidden_size) -> (n_slices, W, transformer_hidden_size)
        decoder_output = decoder_output.permute(0, 2, 1) # (n_slices, transformer_hidden_size, W)

        output = self.post_transformer(decoder_output) # (n_slices, transformer_hidden_size, W) -> (n_slices, 2*H, W)
        output = output.reshape(n_slices, 2, H, W) # (n_slices, 2*H, W) -> (n_slices, 2, H, W)

        return output


