import dotenv
import os
import torch
from models.singlecoil_kspace_convmixer_denoiser import SingleCoilKspaceConvmixerDenoiser
from kspace_trainer import KspaceTrainer

# Load environment variables
dotenv.load_dotenv()

# Training and model hyperparameters
# SSIM-Metric: 0.606969
CONFIG = {
    # Data parameters
    'center_fractions': [0.04],
    'accelerations': [8],
    'seed': 42,

    # Model hyperparameters
    'patch_size': 8,
    'conv_dims': 16,
    'pos_embed_dim': 4,
    'conv_blocks': 4,
    'conv_kernel_size': 13,
    'activation': 'gelu',
    'H': 320,
    'W': 320,

    # Training hyperparameters
    'batch_size': 1,  # Reduced batch size to avoid CUDA errors
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'mse_weight': 1.0,
    'ssim_weight': 1000.0,

    # Paths
    'train_path': os.environ.get('SINGLECOIL_TRAIN_PATH'),
    'val_path': os.environ.get('SINGLECOIL_VAL_PATH'),

    # Checkpointing
    'save_checkpoint_every': 5,
    'checkpoint_dir': 'checkpoints',
}

# Create checkpoint directory if it doesn't exist
os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
def train_model():
    # Initialize model with parameters from CONFIG
    model = SingleCoilKspaceConvmixerDenoiser(
        patch_size=CONFIG['patch_size'],
        conv_dims=CONFIG['conv_dims'],
        pos_embed_dim=CONFIG['pos_embed_dim'],
        conv_blocks=CONFIG['conv_blocks'],
        conv_kernel_size=CONFIG['conv_kernel_size'],
        activation=CONFIG['activation'],
        H=CONFIG['H'],
        W=CONFIG['W']
    )

    # Create trainer instance
    trainer = KspaceTrainer(CONFIG, model, forward_func=lambda kspace, masked_kspace, mask, model: model(masked_kspace, mask))

    # Start training
    trainer.train()


if __name__ == "__main__":
    train_model()
