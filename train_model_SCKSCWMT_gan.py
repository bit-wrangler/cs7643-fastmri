import dotenv
import os
import torch
import fastmri
from models.singlecoil_kspace_columnwise_masked_transformer_denoiser import SingleCoilKspaceColumnwiseMaskedTransformerDenoiser
from kspace_trainer_gan import KspaceTrainerGAN

# Load environment variables
dotenv.load_dotenv()

# Training and model hyperparameters
configs = [
    {
    'tags': ['transformer1', 'gan'], # ['transformer1', 'loss', 'psnr']
    'notes': 'gan test', # 'control'
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
        'encoder_num_heads': 8,
        'decoder_num_heads': 4,
        'pre_dims': 256,
        'kernel_size': 5,
        'pre_layers': 0,
        'hidden_size': 256,
        'activation': 'relu',
        'H': 320,
        'W': 320,
        'apply_pre_norm': False,
    },

    'discriminator': {
        'in_ch': 1,
        'dim': 64,
        'patch': 16,
        'ksize': 9,
        'depth': 8,
    },

    # Training hyperparameters
    'batch_size': 1,  # Reduced batch size to avoid CUDA errors
    'num_epochs': 250,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'mse_weight': 1.,
    'ssim_weight': 1000.,
    'terminate_patience': 10,
    'use_l1': False,
    'max_norm': 10.0,
    'adv_weight'   : 0.01,
    'd_lr'         : 1e-4,
    'gan_start_recon_loss': 501.0,
    'clamp_predicted': False,

    'scheduler': {
        'type': 'ReduceLROnPlateau',
        'factor': 0.5,
        'patience': 5,
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
        model = SingleCoilKspaceColumnwiseMaskedTransformerDenoiser(
            **CONFIG['model']
        )

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{total_params:,} parameters')          # human‑readable with commas
        print(f'{total_params/1e6:.2f} M parameters')  # in millions

        # Create trainer instance with a forward_func that returns image domain predictions
        def forward_func(kspace, masked_kspace, mask, image, model):
            # Get kspace prediction from model
            kspace_pred = model(kspace, mask)

            # Convert to image domain
            kspace_pred_permuted = kspace_pred.permute(0, 2, 3, 1)
            pred_image = fastmri.ifft2c(kspace_pred_permuted)
            pred_image_abs = fastmri.complex_abs(pred_image)

            return pred_image_abs

        trainer = KspaceTrainerGAN(CONFIG, model, forward_func=forward_func)

        # Start training
        trainer.train()


if __name__ == "__main__":
    train_model()
