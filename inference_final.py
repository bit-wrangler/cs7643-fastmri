
# %%import os
import glob
import re
import dotenv
import h5py
import fastmri
import torch
import numpy as np
import matplotlib.pyplot as plt
from fastmri.data.subsample import RandomMaskFunc
import fastmri.data.transforms as T
from ReconFormer_main.models.Recurrent_Transformer import ReconFormer
from models.singlecoil_kspace_columnwise_masked_transformer_denoiser import SingleCoilKspaceColumnwiseMaskedTransformerDenoiser
from models.sc_knee_image_rm2 import RM2
from models.sc_knee_image_rm2b import RM2b
from kspace_trainer import FastMRIDataset, max_normalize
import os
import gc

dotenv.load_dotenv()

def mae_forward_func(kspace, masked_kspace, mask, model):
    # Get kspace prediction from model
    kspace_pred = model(kspace, mask)

    # Convert to image domain
    kspace_pred_permuted = kspace_pred.permute(0, 2, 3, 1)
    pred_image = fastmri.ifft2c(kspace_pred_permuted)
    pred_image_abs = fastmri.complex_abs(pred_image)

    return pred_image_abs

def rm_forward_func(kspace, masked_kspace, mask, model):
    # Get kspace prediction from model
    pred_image_abs = model(masked_kspace, mask)


    return pred_image_abs

def recon_former_forward(kspace, masked_kspace, mask, model):
    """
    Wrapper function to adapt KspaceTrainer data flow for ReconFormer.

    Args:
        kspace (torch.Tensor): Ground truth k-space (N, C, H, W) - Used as Target
        masked_kspace (torch.Tensor): Input masked k-space (N, C, H, W) - Used as k0 and for initial image
        mask (torch.Tensor): Sampling mask (W)
        model (nn.Module): The ReconFormer model instance.

    Returns:
        torch.Tensor: The predicted k-space (N, C, H, W) for the loss function.
    """

    # convert mask from (W) to (1, 1, W)
    mask = mask.unsqueeze(0).unsqueeze(1).unsqueeze(2)

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
    # kspace_pred_cl = fastmri.fft2c(img_pred_cl)                  # (N, H, W, 2)
    # kspace_pred     = kspace_pred_cl.permute(0, 3, 1, 2).contiguous()  # (N, 2, H, W)
    return pred_image_abs

reconformer_config = {
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
        'reconformer_use_checkpoint': (False,) * 6, # Checkpointing for blocks
        

        # Training hyperparameters (kept mostly the same, adjust if needed)
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

        # Paths
        'train_path': os.environ.get('SINGLECOIL_TRAIN_PATH'),
        'val_path': os.environ.get('SINGLECOIL_VAL_PATH'),

        # Checkpointing
        'save_checkpoint_every': 5,
        'checkpoint_dir': 'checkpoints', # Changed checkpoint dir
    }

# %%
models = [
    ('gt', None, None, 'Ground Truth'),
    ('zf', None, None, 'Zero Filled'),
    ('file:./checkpoints/best_model_mae.pt',
        lambda: SingleCoilKspaceColumnwiseMaskedTransformerDenoiser(**{
        'encoder_num_heads': 32,
        'decoder_num_heads': 32,
        'pre_dims': 256,
        'kernel_size': 5,
        'pre_layers': 0,
        'hidden_size': 256,
        'activation': 'relu',
        'H': 320,
        'W': 320,
        'apply_pre_norm': False,
        'apply_dc': True,
    }),
    mae_forward_func,
    'Masked Autoencoder'),
    (
        'file:./checkpoints/best_model_rm2.pt',
        lambda: RM2(**{
        'autoencoder_dim': 32,
        'autoencoder_depth': 4,
        'autoencoder_kernel_size': 9,
        'autoencoder_patch_size': 8,
        'encoder_dim': 64,
        'encoder_depth': 4,
        'encoder_kernel_size': 9,
        'encoder_patch_size': 8,
        'kspace_embedding_dim': 512,
        'transformer_hidden_size': 256,
        'transformer_num_heads': 16,
        'transformer_num_layers': 1,
        'apply_final_dc': True,
        'H': 320,
        'W': 320,
    }),
    rm_forward_func,
    '2-stage w/ k-space'
    ),
    (
        'file:./checkpoints/best_model_rm2b.pt',
        lambda: RM2b(**{
        'autoencoder_dim': 32,
        'autoencoder_depth': 4,
        'autoencoder_kernel_size': 9,
        'autoencoder_patch_size': 8,
        'encoder_dim': 64,
        'encoder_depth': 4,
        'encoder_kernel_size': 9,
        'encoder_patch_size': 8,
        'kspace_embedding_dim': 512,
        'transformer_hidden_size': 256,
        'transformer_num_heads': 16,
        'transformer_num_layers': 1,
        'apply_final_dc': True,
        'H': 320,
        'W': 320,
    }),
    rm_forward_func,
    '2-stage w/o k-space'
    ),
    (
        'file:./checkpoints/best_model_rf.pt',
        lambda: ReconFormer(
            in_channels=reconformer_config['reconformer_in_channels'],
            out_channels=reconformer_config['reconformer_out_channels'],
            num_ch=reconformer_config['reconformer_num_ch'],
            down_scales=reconformer_config['reconformer_down_scales'],
            num_iter=reconformer_config['reconformer_num_iter'],
            img_size=reconformer_config['reconformer_img_size'],
            num_heads=reconformer_config['reconformer_num_heads'],
            depths=reconformer_config['reconformer_depths'],
            window_sizes=reconformer_config['reconformer_window_sizes'],
            resi_connection=reconformer_config['reconformer_resi_connection'],
            mlp_ratio=reconformer_config['reconformer_mlp_ratio'],
            use_checkpoint=reconformer_config['reconformer_use_checkpoint']
        ),
        recon_former_forward,
        'ReconFormer'
    )
]

# %%

TEST_PATH    = os.environ['SINGLECOIL_VAL_PATH']
OUT_FIG      = "comparison_vertical.png"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACCEL        = 8
CENTER_FRAC  = 0.04
TEST_FILE_INDEX = 0
H, W          = 320, 320


test_files = sorted(glob.glob(os.path.join(TEST_PATH, "*.h5")))


# %%

mask_func = RandomMaskFunc(
            center_fractions=[CENTER_FRAC],
            accelerations=[ACCEL],
            seed=42
        )

val_dataset = FastMRIDataset(
            TEST_PATH,
            mask_func,
            target_size=(W, H),
            augment=False, 
        )

# %%
def _process_batch(batch, norm, device):
    """Process a batch and return kspace, masked_kspace, and mask tensors."""
    # {
    #     'kspace': kspace,
    #     'masked_kspace': masked_kspace,
    #     'image': zf_abs,
    #     'target': target,
    #     'zf_max': zf_max,
    #     'mask': mask,
    #     'file_path': file_path,
    # }

    # Get data from batch
    kspace = batch['kspace'].clone().to(device, non_blocking=True)
    masked_kspace = batch['masked_kspace'].clone().to(device, non_blocking=True)
    mask = batch['mask'].clone().to(device, non_blocking=True)

    with torch.no_grad():
        mk_abs = fastmri.complex_abs(fastmri.ifft2c(masked_kspace.permute(0,2,3,1)))

        if norm == 'max':
            # zf_max = zf_abs.view(zf_abs.size(0), -1).amax(dim=1).clamp_min_(1e-6)
            # zf_abs = max_normalize(zf_abs, zf_max)
            # target = max_normalize(target, zf_max)
            mk_max = mk_abs.view(mk_abs.size(0), -1).amax(dim=1).clamp_min_(1e-6)
            kspace = max_normalize(kspace, mk_max)
            masked_kspace = max_normalize(masked_kspace, mk_max)
            normalization = {
                'type': 'max',
                'zf_max': mk_max
            }
        elif norm == 'meanstd':
            # zf_abs, mean, std = T.normalize_instance(zf_abs, eps=1e-6)
            # target = meanstd_normalize(target, mean, std, eps=1e-6)
            mk_abs, mean, std = T.normalize_instance(mk_abs, eps=1e-6)
            kspace = meanstd_normalize(kspace, mean, std, eps=1e-6)
            masked_kspace = meanstd_normalize(masked_kspace, mean, std, eps=1e-6)
            normalization = {
                'type': 'meanstd',
                'mean': mean,
                'std': std
            }
        zf_abs = fastmri.complex_abs(fastmri.ifft2c(masked_kspace.permute(0,2,3,1)))
        target = fastmri.complex_abs(fastmri.ifft2c(kspace.permute(0,2,3,1)))

    return kspace, masked_kspace, mask, zf_abs, target, normalization

# %%
batch = val_dataset[TEST_FILE_INDEX]

# %%

def run_model(checkpoint, model_create, forward_func, label, batch, norm, device):
    kspace, masked_kspace, mask, zf_abs, target, normalization = _process_batch(batch, norm, device)
    if checkpoint.startswith('file:'):
        ckpt_path = checkpoint[5:]
        model = model_create().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state_dict'])
        model.eval()

        with torch.inference_mode():                # no autograd graph
            pred = forward_func(kspace, masked_kspace, mask, model)

        # keep only a CPU copy of what you need for later plotting
        pred_cpu = pred.cpu()

        # drop GPU references **before** the next model is built
        del model, pred
        torch.cuda.empty_cache()    # release cached blocks
        torch.cuda.ipc_collect()    # free IPC-shared memory
        gc.collect()                # run Python GC
    else:
        if checkpoint == 'gt':
            pred_cpu = target.cpu()
        elif checkpoint == 'zf':
            pred_cpu = zf_abs.cpu()
        else:
            raise ValueError(f'Unknown checkpoint: {checkpoint}')
    return pred_cpu

# %%

outputs = [run_model(
    checkpoint, model, forward_func, label, batch, 'max', DEVICE
) for checkpoint, model, forward_func, label in models]
# %%

fig, axs = plt.subplots(len(models), 1, figsize=(4, 3*len(models)))

for i, ax in enumerate(axs):
    (checkpoint, model, forward_func, label) = models[i]
    output = outputs[i]
    idx = output.shape[0] // 2
    img = output[idx].numpy()
    title = label
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    ax.set_title(title, fontsize=12)

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
print('done')