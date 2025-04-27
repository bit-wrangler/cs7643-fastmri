import dotenv
import os
import torch
import fastmri
from models.sc_knee_image_rm import RM1Multi
from kspace_trainer import KspaceTrainer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32  = True

# Load environment variables
dotenv.load_dotenv()

# Training and model hyperparameters
configs = [
    {
    'tags': ['transformer1', 'updated_trainer_4'], # ['transformer1', 'loss', 'psnr']
    'notes': 'control4 - max normalization', # 'control'
    # Data parameters
    'val_center_fractions': [0.04],
    'val_accelerations': [8],
    'train_center_fractions': [0.04],
    'train_accelerations': [8],
    'seed': 42,
    'H': 320,
    'W': 320,

    # Model hyperparameters
    'model': {
        'dim': 64,
        'depth': 1,
        'k': 9,
        'patch_size': 8,
        'n_layers': 4,
    },

    # Training hyperparameters
    'batch_size': 1,  # Reduced batch size to avoid CUDA errors
    'num_epochs': 250,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'mse_weight': 1.,
    'ssim_weight': 1.,
    'psnr_weight': 0.1,
    'terminate_patience': 12,
    'use_l1': True,
    'l1_transition_mse': 0.02,
    'max_norm': 10.0,
    'normalization': 'max',

    'scheduler': {
        'type': 'ReduceLROnPlateau',
        'factor': 0.5,
        'patience': 6,
    },

    # 'scheduler': {
    #     'type': 'CyclicLR',
    #     'base_lr': 1e-6,
    #     'max_lr': 1e-3,
    #     'step_size_up': 250,
    #     'mode': 'exp_range',
    #     'gamma': 0.99993,
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
        model = RM1Multi(**CONFIG['model'])

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{total_params:,} parameters')          # human‑readable with commas
        print(f'{total_params/1e6:.2f} M parameters')  # in millions

        # Create trainer instance with a forward_func that returns image domain predictions
        def forward_func(kspace, masked_kspace, mask, image, model):
            # Get kspace prediction from model
            pred_image_abs = model(masked_kspace, mask)


            return None, pred_image_abs

        trainer = KspaceTrainer(CONFIG, model, forward_func=forward_func)

        # Start training
        trainer.train()


if __name__ == "__main__":
    train_model()
