import dotenv
import os
import torch
from kspace_trainer import KspaceTrainer
from models.singlecoil_kspace_transformer_autoencoder5 import SingleCoilKspaceTransformerAutoencoder5

# Load environment variables
dotenv.load_dotenv()

# Training and model hyperparameters
configs = [
{
    'tags': ['autoencoder5'], # ['transformer1', 'loss', 'psnr']
    # Data parameters
    'center_fractions': [0.04,0.04,0.04,0.04],
    'accelerations': [1,2,4,8],
    'seed': 42,
    'H': 320,
    'W': 320,

    # Model hyperparameters
    'model': {
        'decoder1_num_heads': 8,
        'n_decoder1_layers': 2,
        'decoder2_num_heads': 8,
        'n_decoder2_layers': 2,
        'transformer_hidden_size': 256,
        'ff_dim': 512,
        'dropout': 0.0,
        'n_summary_tokens': 32,
        'H': 320,
        'W': 320,
    },

    # Training hyperparameters
    'batch_size': 1,  # Reduced batch size to avoid CUDA errors
    'num_epochs': 400,
    'learning_rate': 2e-4,
    'lr_decay': 0.99999,
    'weight_decay': 1e-5,
    'mse_weight': 1.,
    'ssim_weight': 10.,
    'min_learning_rate': 1e-6,
    'patience': 100,
    'use_l1': True,

    # Paths
    'train_path': os.environ.get('SINGLECOIL_TRAIN_PATH'),
    'val_path': os.environ.get('SINGLECOIL_VAL_PATH'),

    # Checkpointing
    'save_checkpoint_every': 10,
    'checkpoint_dir': 'checkpoints',
},
]

# Create checkpoint directory if it doesn't exist
os.makedirs(configs[0]['checkpoint_dir'], exist_ok=True)
def train_model():
    for CONFIG in configs:
        # Initialize model with parameters from CONFIG
        model = SingleCoilKspaceTransformerAutoencoder5(
            **CONFIG['model']
        )

        # Create trainer instance
        trainer = KspaceTrainer(CONFIG, model, forward_func=lambda kspace, masked_kspace, mask, model: model(kspace,mask))

        # Start training
        trainer.train()


if __name__ == "__main__":
    train_model()
