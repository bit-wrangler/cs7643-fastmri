import dotenv
import os
import torch
import fastmri # Added import
import fastmri.data.transforms as T # Added potentially missing import T
# from models.singlecoil_kspace_columnwise_masked_transformer_denoiser import SingleCoilKspaceColumnwiseMaskedTransformerDenoiser # Removed old model import
from ReconFormer_main.models.Recurrent_Transformer import ReconFormer # Import the new model

from kspace_trainer import KspaceTrainer

# Load environment variables
dotenv.load_dotenv()

# Training and model hyperparameters
configs = [
    # experiment using ReconFormer
    {
        'tags': ['ReconFormer', 'updated_trainer'], # Updated tags
        'notes': 'Training with ReconFormer model', # Updated notes
        # Data parameters
        'center_fractions': [0.04],
        'accelerations': [8],
        'seed': 42,
        'H': 320, # Image height (ensure matches ReconFormer expectations if needed)
        'W': 320, # Image width (ensure matches ReconFormer expectations if needed)

        # Model hyperparameters for ReconFormer (adjust values as needed)
        'reconformer_in_channels': 2, # Typically 2 for complex data (real, imag)
        'reconformer_out_channels': 2,
        #! 'reconformer_num_ch': (96, 48, 24), # Example channel numbers per block
        'reconformer_num_ch': (96, 48, 24), # Example channel numbers per block

        'reconformer_down_scales': (2, 1, 1.5), # Example scales for UC/OC blocks
        # 'reconformer_num_ch':         (48, 24, 24),
        'reconformer_num_iter': 1, # Number of recurrent iterations in ReconFormer
        'reconformer_img_size': 320, # Image size expected by ReconFormer blocks
        # 'reconformer_img_size':       256,
        # 'reconformer_num_heads': (6, 6, 6), # Example head numbers per block
        'reconformer_num_heads':      (6, 6, 6),
        'reconformer_depths': (6, 6, 6), # Example depths per block
        'reconformer_window_sizes': (8, 8, 8), # Example window sizes per block
        'reconformer_resi_connection': '1conv', # Type of residual connection
        'reconformer_mlp_ratio': 2.0,
        #!'reconformer_use_checkpoint': (False,) * 6, # Checkpointing for blocks
        'reconformer_use_checkpoint': (True,) * 6, # Checkpointing for blocks
        

        # Training hyperparameters (kept mostly the same, adjust if needed)
        'batch_size': 1,
        'num_epochs': 250,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'mse_weight': 1.0, # Image domain MSE weight
        'ssim_weight': 1000.0, # Image domain SSIM weight
        'terminate_patience': 10,
        'use_l1': False, # Whether to use L1 for image domain MSE

        'scheduler': {
            'type': 'ReduceLROnPlateau',
            'factor': 0.5,
            'patience': 5,
        },

        # Paths
        'train_path': os.environ.get('SINGLECOIL_TRAIN_PATH'),
        'val_path': os.environ.get('SINGLECOIL_VAL_PATH'),

        # Checkpointing
        'save_checkpoint_every': 5,
        'checkpoint_dir': 'checkpoints_reconformer', # Changed checkpoint dir
    },
]

# configs = [
#     {
#         'tags': ['ReconFormer', 'small_model'],
#         'notes': 'Lightweight ReconFormer for 8 GB GPU',
#         # Data parameters
#         'center_fractions': [0.04],
#         'accelerations': [8],
#         'seed': 42,
#         'H': 192,                      # crop height
#         'W': 192,                      # crop width

#         # ReconFormer hyperparameters
#         'reconformer_in_channels': 2,
#         'reconformer_out_channels': 2,
#         'reconformer_num_ch': (32, 16, 8),        # much smaller channel widths
#         'reconformer_down_scales': (2, 1, 1.5),
#         'reconformer_num_iter': 2,                # fewer recurrent iterations
#         'reconformer_img_size': 192,              # match the crop size
#         'reconformer_num_heads': (4, 4, 4),       # divides evenly into (32,16,8)
#         'reconformer_depths': (3, 3, 3),          # fewer transformer layers
#         'reconformer_window_sizes': (8, 8, 8),
#         'reconformer_resi_connection': '1conv',
#         'reconformer_mlp_ratio': 2.0,
#         'reconformer_use_checkpoint': (True,) * 6,# enable gradient checkpointing

#         # Training hyperparameters
#         'batch_size': 1,
#         'num_epochs': 250,
#         'learning_rate': 1e-4,
#         'weight_decay': 1e-5,
#         'mse_weight': 1.0,
#         'ssim_weight': 1000.0,
#         'terminate_patience': 10,
#         'use_l1': False,

#         'scheduler': {
#             'type': 'ReduceLROnPlateau',
#             'factor': 0.5,
#             'patience': 5,
#         },

#         # Paths (from .env)
#         'train_path': os.environ.get('SINGLECOIL_TRAIN_PATH'),
#         'val_path': os.environ.get('SINGLECOIL_VAL_PATH'),

#         # Checkpointing
#         'save_checkpoint_every': 5,
#         'checkpoint_dir': 'checkpoints_reconformer',
#     },
# ]


# Create checkpoint directory if it doesn't exist
os.makedirs(configs[0]['checkpoint_dir'], exist_ok=True)

# Define the forward function wrapper for ReconFormer
def recon_former_forward(kspace, masked_kspace, mask, image, model):
    """
    Wrapper function to adapt KspaceTrainer data flow for ReconFormer.

    Args:
        kspace (torch.Tensor): Ground truth k-space (N, C, H, W) - Used as Target
        masked_kspace (torch.Tensor): Input masked k-space (N, C, H, W) - Used as k0 and for initial image
        mask (torch.Tensor): Sampling mask (1, 1, W, 1) - Used for DC
        model (nn.Module): The ReconFormer model instance.

    Returns:
        torch.Tensor: The predicted k-space (N, C, H, W) for the loss function.
    """

    # image_complex = fastmri.ifft2c(masked_kspace)
    # image_pred_complex = model(image_complex, k0=masked_kspace, mask=mask)
    # output = fastmri.complex_abs(image_pred_complex)
    masked_k_cl = masked_kspace.permute(0, 2, 3, 1).contiguous()

    # Inverse FFT into image domain, result is (N, H, W, 2)
    img_complex_cl = fastmri.ifft2c(masked_k_cl)

    # Back to channel-first for the model: (N, H, W, 2) → (N, 2, H, W)
    img_input = img_complex_cl.permute(0, 3, 1, 2).contiguous()

    # ReconFormer forward takes (image, k0, mask) in channel-first
    img_pred = model(img_input, k0=masked_kspace, mask=mask)

    # Compute magnitude of the predicted complex image
    # Move channels last for complex_abs: (N, 2, H, W) → (N, H, W, 2)
    img_pred_cl = img_pred.permute(0, 2, 3, 1).contiguous()
    pred_image_abs = fastmri.complex_abs(img_pred_cl)

    # 2) k‐space prediction for L1 k‐space loss:
    kspace_pred_cl = fastmri.fft2c(img_pred_cl)                  # (N, H, W, 2)
    kspace_pred     = kspace_pred_cl.permute(0, 3, 1, 2).contiguous()  # (N, 2, H, W)
    return kspace_pred, pred_image_abs


def train_model():
    for CONFIG in configs:
        # Initialize ReconFormer model with parameters from CONFIG
        model = ReconFormer(
            in_channels=CONFIG['reconformer_in_channels'],
            out_channels=CONFIG['reconformer_out_channels'],
            num_ch=CONFIG['reconformer_num_ch'],
            down_scales=CONFIG['reconformer_down_scales'],
            num_iter=CONFIG['reconformer_num_iter'],
            img_size=CONFIG['reconformer_img_size'],
            num_heads=CONFIG['reconformer_num_heads'],
            depths=CONFIG['reconformer_depths'],
            window_sizes=CONFIG['reconformer_window_sizes'],
            resi_connection=CONFIG['reconformer_resi_connection'],
            mlp_ratio=CONFIG['reconformer_mlp_ratio'],
            use_checkpoint=CONFIG['reconformer_use_checkpoint']
        )

        

        # Create trainer instance, passing the custom forward function
        trainer = KspaceTrainer(CONFIG, model, forward_func=recon_former_forward)

        # Start training
        trainer.train()


if __name__ == "__main__":
    train_model()