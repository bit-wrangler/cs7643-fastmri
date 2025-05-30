import dotenv
import os
import torch
from kspace_trainer import KspaceTrainer
from models.singlecoil_kspace_transformer_autoencoder import SingleCoilKspaceTransformerAutoencoder

# Load environment variables
dotenv.load_dotenv()

# Training and model hyperparameters
configs = [
{
    'tags': ['autoencoder1'], # ['transformer1', 'loss', 'psnr']
    # Data parameters
    'center_fractions': [0.04],
    'accelerations': [8],
    'seed': 42,

    # Model hyperparameters
    'encoder_num_heads': 16,
    'decoder_num_heads': 16,
    'hidden_size': 256,
    'H': 320,
    'W': 320,

    # Training hyperparameters
    'batch_size': 1,  # Reduced batch size to avoid CUDA errors
    'num_epochs': 150,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'mse_weight': 1.,
    'ssim_weight': 100.,
    'min_learning_rate': 1e-6,

    # Paths
    'train_path': os.environ.get('SINGLECOIL_TRAIN_PATH'),
    'val_path': os.environ.get('SINGLECOIL_VAL_PATH'),

    # Checkpointing
    'save_checkpoint_every': 5,
    'checkpoint_dir': 'checkpoints',
},
]

# Create checkpoint directory if it doesn't exist
os.makedirs(configs[0]['checkpoint_dir'], exist_ok=True)
def train_model():
    for CONFIG in configs:
        # Initialize model with parameters from CONFIG
        model = SingleCoilKspaceTransformerAutoencoder(
            encoder_num_heads=CONFIG['encoder_num_heads'],
            decoder_num_heads=CONFIG['decoder_num_heads'],
            transformer_hidden_size=CONFIG['hidden_size'],
            H=CONFIG['H'],
            W=CONFIG['W']
        )

        # Create trainer instance
        trainer = KspaceTrainer(CONFIG, model, forward_func=lambda kspace, masked_kspace, mask, model: model(kspace))

        # Start training
        trainer.train()


if __name__ == "__main__":
    train_model()
