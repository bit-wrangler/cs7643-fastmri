import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedKspaceColumnEmbedding(nn.Module):
    def __init__(
            self,
            dim1: int = 32,
            H: int = 320,
            W: int = 320,
                 ):
        super().__init__()

        # pointwise conv to go from kspace (2*H channels) to pre_dims channels
        self.pre_conv = nn.Conv1d(2*H, dim1, kernel_size=1)

        self.pre_conv_position_embedding = nn.Embedding(W, dim1)

        self.pre_conv_mask_embedding = nn.Embedding(2, dim1)

    def forward(self, kspace, col_mask):
        # kspace is of shape (n_slices, 2, H, W)
        # mask is of shape (W)
        n_slices = kspace.shape[0]
        H = kspace.shape[2]
        W = kspace.shape[3]
        M = col_mask.sum()

        position = torch.arange(W).unsqueeze(0).unsqueeze(0).to(kspace.device)
        position_embedding = self.pre_conv_position_embedding(position) # (1, 1, W) -> (1, 1, W, pre_dims)
        position_embedding = position_embedding.repeat(n_slices, 1, 1, 1) # (n_slices, 1, W, pre_dims)
        position_embedding = position_embedding.squeeze(1).permute(0, 2, 1) # (n_slices, pre_dims, W)

        mask_ids = col_mask.squeeze().long() # (W,)
        mask_embedding = self.pre_conv_mask_embedding(mask_ids) # (W, hidden_size)
        mask_embedding = mask_embedding.t() # (hidden_size, W)
        mask_embedding = mask_embedding.unsqueeze(0).repeat(n_slices, 1, 1) # (n_slices, hidden_size, W)

        column_kspace = kspace.reshape(n_slices, 2*H, W) # (n_slices, 2*H, W)

        projected_column_kspace = self.pre_conv(column_kspace) # (n_slices, pre_dims, W)
        projected_column_kspace += mask_embedding # (n_slices, pre_dims, W)
        projected_column_kspace += position_embedding # (n_slices, pre_dims, W)

        return projected_column_kspace