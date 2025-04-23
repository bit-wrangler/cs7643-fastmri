import dotenv
import os
import torch
from models.singlecoil_kspace_columnwise_masked_transformer_denoiser import SingleCoilKspaceColumnwiseMaskedTransformerDenoiser
from kspace_trainer import KspaceTrainer

# Load environment variables
dotenv.load_dotenv()

# Training and model hyperparameters
configs = [
    # experiment 1 - 1 layer
{
    'tags': ['full-transformer', 'enc_layers_1-dec_layers_1'],
    'notes': 'replaced multihead with full transformer encoder attention - 1 layer',
    # Data parameters
    'center_fractions': [0.04],
    'accelerations': [8],
    'seed': 42,

    # Model hyperparameters
    'encoder_num_heads': 8,
    'decoder_num_heads': 4,
    'pre_dims': 512,
    'kernel_size': 5,
    'pre_layers': 0,
    'hidden_size': 512,
    'activation': 'relu',
    'H': 320,
    'W': 320,
    'encoder_layers': 1,
    'decoder_layers': 1,

    # Training hyperparameters
    'batch_size': 1,  # Reduced batch size to avoid CUDA errors
    'num_epochs': 250,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'mse_weight': 1.,
    'ssim_weight': 1000.,
    'terminate_patience': 10,
    'use_l1': False,

    'scheduler': {
        'type': 'ReduceLROnPlateau',
        'factor': 0.5,
        'patience': 5,
    },

    # 'scheduler': {
    #     'type': 'CyclicLR',
    #     'base_lr': 1e-6,
    #     'max_lr': 2e-4,
    #     'step_size_up': 250,
    #     'mode': 'exp_range',
    #     'gamma': 0.99999,
    # },

    # Paths
    'train_path': os.environ.get('SINGLECOIL_TRAIN_PATH'),
    'val_path': os.environ.get('SINGLECOIL_VAL_PATH'),

    # Checkpointing
    'save_checkpoint_every': 5,
    'checkpoint_dir': 'checkpoints',
},
    # experiment 2 - 2 layers
{
    'tags': ['full-transformer', 'enc_layers_2-dec_layers_2'],
    'notes': 'replaced multihead with full transformer encoder attention - 2 layers',
    # Data parameters
    'center_fractions': [0.04],
    'accelerations': [8],
    'seed': 42,

    # Model hyperparameters
    'encoder_num_heads': 8,
    'decoder_num_heads': 4,
    'pre_dims': 512,
    'kernel_size': 5,
    'pre_layers': 0,
    'hidden_size': 512,
    'activation': 'relu',
    'H': 320,
    'W': 320,
    'encoder_layers': 2,
    'decoder_layers': 2,

    # Training hyperparameters
    'batch_size': 1,  # Reduced batch size to avoid CUDA errors
    'num_epochs': 250,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'mse_weight': 1.,
    'ssim_weight': 1000.,
    'terminate_patience': 10,
    'use_l1': False,

    'scheduler': {
        'type': 'ReduceLROnPlateau',
        'factor': 0.5,
        'patience': 5,
    },

    # 'scheduler': {
    #     'type': 'CyclicLR',
    #     'base_lr': 1e-6,
    #     'max_lr': 2e-4,
    #     'step_size_up': 250,
    #     'mode': 'exp_range',
    #     'gamma': 0.99999,
    # },

    # Paths
    'train_path': os.environ.get('SINGLECOIL_TRAIN_PATH'),
    'val_path': os.environ.get('SINGLECOIL_VAL_PATH'),

    # Checkpointing
    'save_checkpoint_every': 5,
    'checkpoint_dir': 'checkpoints',
},
    # experiment 2 - 2 layers
{
    'tags': ['full-transformer', 'enc_layers_4-dec_layers_4'],
    'notes': 'replaced multihead with full transformer encoder attention - 4 layers',
    # Data parameters
    'center_fractions': [0.04],
    'accelerations': [8],
    'seed': 42,

    # Model hyperparameters
    'encoder_num_heads': 8,
    'decoder_num_heads': 4,
    'pre_dims': 512,
    'kernel_size': 5,
    'pre_layers': 0,
    'hidden_size': 512,
    'activation': 'relu',
    'H': 320,
    'W': 320,
    'encoder_layers': 4,
    'decoder_layers': 4,

    # Training hyperparameters
    'batch_size': 1,  # Reduced batch size to avoid CUDA errors
    'num_epochs': 250,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'mse_weight': 1.,
    'ssim_weight': 1000.,
    'terminate_patience': 10,
    'use_l1': False,

    'scheduler': {
        'type': 'ReduceLROnPlateau',
        'factor': 0.5,
        'patience': 5,
    },

    # 'scheduler': {
    #     'type': 'CyclicLR',
    #     'base_lr': 1e-6,
    #     'max_lr': 2e-4,
    #     'step_size_up': 250,
    #     'mode': 'exp_range',
    #     'gamma': 0.99999,
    # },

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
        model = SingleCoilKspaceColumnwiseMaskedTransformerDenoiser(
            encoder_num_heads=CONFIG['encoder_num_heads'],
            decoder_num_heads=CONFIG['decoder_num_heads'],
            pre_dims=CONFIG['pre_dims'],
            pre_layers=CONFIG['pre_layers'],
            hidden_size=CONFIG['hidden_size'],
            kernel_size=CONFIG['kernel_size'],
            activation=CONFIG['activation'],
            H=CONFIG['H'],
            W=CONFIG['W'],
            encoder_layers = CONFIG['encoder_layers'] if 'encoder_layers' in CONFIG else 1,
            decoder_layers = CONFIG['decoder_layers'] if 'decoder_layers' in CONFIG else 1
        )

        # Create trainer instance
        trainer = KspaceTrainer(CONFIG, model, forward_func=lambda kspace, masked_kspace, mask, model: model(kspace, mask))

        # Start training
        trainer.train()


if __name__ == "__main__":
    train_model()
