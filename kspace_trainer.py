import h5py
import os
import glob
import fastmri
import fastmri.data.transforms as T
from fastmri.evaluate import ssim, psnr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from fastmri.data.subsample import RandomMaskFunc
from ssim_loss import ssim_loss
from psnr_loss import PSNRLoss
import wandb
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal
import math, random
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32  = True

def clipped_interp(x, w_min, w_max, x_min, x_max):
    x = np.asarray(x)
    y = w_max + (w_min - w_max) * (x - x_min) / (x_max - x_min)
    y = np.where(x <= x_min, w_max, np.where(x >= x_max, w_min, y))
    return y

def minmax_normalize(image, min_val, scale, min_clip=0, max_clip=1):
    return ((image - min_val) / scale)#.clamp(min_clip, max_clip)

def max_normalize(image, max_value):
    while max_value.ndim < image.ndim:
        max_value = max_value.unsqueeze(-1)
    return image / max_value

def meanstd_normalize(image, mean, std, eps=1e-6):
    while mean.ndim < image.ndim:   # add singleton dims at the end
        mean = mean.unsqueeze(-1)
        std  = std .unsqueeze(-1)

    return ((image - mean) / (std + eps)).clamp(-6, 6)

class FastMRIDataset(Dataset):
    def __init__(self, data_path, mask_func=None, target_size=None, augment=False):
        """
        Initialize the FastMRI dataset.

        Args:
            data_path: Path to the data directory
            mask_func: Mask function to apply
            target_size: Target size for cropping (W, H). If None, no cropping is performed.
        """
        self.data_path = data_path
        self.mask_func = mask_func
        self.target_size = target_size  # Target size for W and H dimensions
        self.file_list = glob.glob(os.path.join(data_path, '*.h5'))
        self.augment = augment
        print(f"Found {len(self.file_list)} files in {data_path}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        with h5py.File(file_path, 'r') as hf:
            volume_kspace = hf['kspace'][()]

            kspace = T.to_tensor(volume_kspace)    # (N, H, W, 2)

            if self.target_size is not None:
                kspace = T.complex_center_crop(kspace, self.target_size)

            # ---------------- image-domain aug -----------------
            if self.augment:
                img = fastmri.ifft2c(kspace).permute(0, 3, 1, 2) # (N, H, W, 2) -> (N, 2, H, W)

                # +/-5 px translation
                dx, dy = random.randint(-5, 5), random.randint(-5, 5)
                img = torch.roll(img, shifts=(dy, dx), dims=(-2, -1))

                # flip image horizontally
                if random.random() < 0.5:
                    img = torch.flip(img, dims=(-1,))

                # +/-3° rotation
                angle = random.uniform(-3.0, 3.0)
                if abs(angle) > 1e-3:
                    theta = torch.tensor([[ math.cos(math.radians(angle)), -math.sin(math.radians(angle)), 0],
                                    [ math.sin(math.radians(angle)),  math.cos(math.radians(angle)), 0]],
                                    dtype=img.dtype, device=img.device)
                    grid = F.affine_grid(theta.unsqueeze(0).repeat(img.size(0),1,1),
                                        img.size(), align_corners=False)
                    img = F.grid_sample(img, grid, mode='bilinear',
                                        padding_mode='zeros', align_corners=False)

                kspace = fastmri.fft2c(img.permute(0, 2, 3, 1))
                                  # back to (S,2,H,W)
            # ---------------------------------------------------

            # (re)-apply sampling mask **after** augmentation
            if self.mask_func is not None:
                masked_kspace, mask, _ = T.apply_mask(kspace, self.mask_func)
            else:
                masked_kspace, mask = kspace, torch.ones_like(kspace[..., :1])

            kspace = kspace.permute(0, 3, 1, 2)
            masked_kspace = masked_kspace.permute(0, 3, 1, 2)

            mask = mask.to(torch.bool).squeeze()  

            return {
                'kspace': kspace,
                'masked_kspace': masked_kspace,
                'mask': mask,
                'file_path': file_path,
            }


def image_losses(pred_image_abs, target_image_abs, use_l1=False):
    """
    Calculate both MSE and SSIM losses in image domain
    Returns both losses separately

    Args:
        pred_image_abs: Predicted image magnitude (already in image domain)
        target_image_abs: Target image magnitude (already in image domain)
        use_l1: Whether to use L1 loss instead of MSE
    """
    pred_image_abs_norm   = pred_image_abs 
    target_image_abs_norm = target_image_abs
    if use_l1:
        mse_loss = nn.L1Loss()(pred_image_abs_norm, target_image_abs_norm)
    else:
        mse_loss = nn.MSELoss()(pred_image_abs_norm, target_image_abs_norm)

    data_range = torch.tensor(3.0).to(pred_image_abs.device).repeat(pred_image_abs.shape[0])

    # Calculate SSIM loss
    # For SSIM loss, we need to keep the dimensions
    # SSIM module expects inputs of shape [batch, channel, height, width]
    # Add channel dimension since we have a single channel (magnitude image)
    pred_image_abs_ssim = pred_image_abs.unsqueeze(1)
    target_image_abs_ssim = target_image_abs.unsqueeze(1)
    ssim_loss_val = ssim_loss(target_image_abs_ssim, pred_image_abs_ssim, data_range)

    return mse_loss, ssim_loss_val


def combined_loss(kspace_pred, kspace_target,pred_image_abs, target_image_abs, psnr_loss, ave_mse=1e6, mse_weight=1.0, ssim_weight=1.0, psnr_weight=1.0, use_l1=False):
    """
    Combined loss using both MSE and SSIM losses in the image domain
    Returns the combined loss and individual losses for tracking

    Args:
        pred_image_abs: Predicted image magnitude (already in image domain)
        target_image_abs: Target image magnitude (already in image domain)
        mse_weight: Weight for MSE loss
        ssim_weight: Weight for SSIM loss
        use_l1: Whether to use L1 loss instead of MSE
    """
    mse_loss, ssim_loss_val = image_losses(pred_image_abs, target_image_abs, use_l1=use_l1)
    psnr_loss_val = psnr_loss(pred_image_abs, target_image_abs)
    # psnr_loss_val = psnr_loss(kspace_pred, kspace_target)

    if kspace_pred is not None:
        k_loss = nn.L1Loss()(kspace_pred, kspace_target)

    log_ave_mse = np.log10(ave_mse) # 1000 -> 3

    interp_ssim_weight = clipped_interp(log_ave_mse, 0.01, ssim_weight, -1, 1).item()
    interp_psnr_weight = clipped_interp(log_ave_mse, 0.001, psnr_weight, -1, 1).item()

    # Combine losses with their respective weights
    total_loss = mse_weight * mse_loss + interp_ssim_weight * ssim_loss_val + interp_psnr_weight * psnr_loss_val
    if kspace_pred is not None:
        total_loss += k_loss * 0.5

    return total_loss, mse_loss, ssim_loss_val, psnr_loss_val


def centered_crop(tensor, target_W, target_H):
    """
    Centered crop of a tensor to target dimensions
    """
    _,_, H, W= tensor.shape
    start_H = (H - target_H) // 2
    start_W = (W - target_W) // 2
    return tensor[:, :, start_H:start_H+target_H, start_W:start_W+target_W]


def centered_crop_1d(tensor, target_W):
    """
    Centered crop of a tensor to target dimensions
    """
    _, _, W, _ = tensor.shape
    start_W = (W - target_W) // 2
    return tensor[:, :, start_W:start_W+target_W, :]

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class KspaceTrainer:
    def __init__(self, config, model, forward_func=None):
        """
        Initialize the KspaceTrainer.

        Args:
            config: Dictionary containing training configuration parameters
            model: PyTorch model to train
            forward_func: Optional callable that takes (kspace, masked_kspace, mask, model) and returns outputs.
                          If None, model(kspace, mask) will be used.
        """
        self.config = config
        self.model = model
        self.forward_func = forward_func


        self.plot_idx = None
        self.plot_file = None

        self.use_l1 = self.config.get('use_l1', False)

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = torch.amp.GradScaler(enabled=self.device.type == "cuda")
        print(f"Using device: {self.device}")

        # Move model to device
        self.model = self.model.to(self.device)

        tags = None
        notes = None

        if 'tags' in config:
            tags = config['tags']

        if 'notes' in config:
            notes = config['notes']

        # Initialize wandb
        project_name = os.environ.get('WANDB_PROJECT_NAME', 'cs7643-fastmri')
        self.run = wandb.init(project=project_name, config=config, tags=tags, notes=notes)

        # Log model information
        self.run.name = f"{type(model).__name__}_{self.run.id}"
        self.run.save()

        # Log model architecture
        self.run.config.update({
            "model_name": type(model).__name__,
            "model_structure": str(model)
        })

        # Create mask function
        self.val_mask_func = RandomMaskFunc(
            center_fractions=self.config.get('val_center_fractions', [0.04]),
            accelerations=self.config.get('val_accelerations', [8]),
            seed=self.config.get('seed', 42)
        )

        self.train_mask_func = RandomMaskFunc(
            center_fractions=self.config.get('train_center_fractions', [0.04]),
            accelerations=self.config.get('train_accelerations', [8]),
            seed=self.config.get('seed', 42)
        )

        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.config.get('checkpoint_dir', 'checkpoints'), exist_ok=True)

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )

        config_contains_scheduler = 'scheduler' in config and isinstance(config['scheduler'], dict)

        scheduler_type = 'ReduceLROnPlateau'
        scheduler_args = {
            'factor': 0.5,
            'patience': 5,
        }

        if config_contains_scheduler:
            scheduler_type = config['scheduler']['type']
            scheduler_args = {k: v for k, v in config['scheduler'].items() if k != 'type'}

        if scheduler_type == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_args.get('factor', 0.5),
                patience=scheduler_args.get('patience', 5),
                verbose=True
            )

        elif scheduler_type == 'CyclicLR':
            self.scheduler = optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=scheduler_args.get('base_lr', 1e-6),
                max_lr=scheduler_args.get('max_lr', 2e-4),
                step_size_up=scheduler_args.get('step_size_up', 250),
                mode=scheduler_args.get('mode', 'exp_range'),
                gamma=scheduler_args.get('gamma', 0.99994),
            )

        self.psnr_loss = PSNRLoss()

        print(f"Using scheduler: {scheduler_type}")
        print(f"Scheduler config: {scheduler_args}")

    def _get_dataloaders(self):
        """Create and return train and validation dataloaders."""
        # Create datasets with target size for cropping
        target_size = (self.config['W'], self.config['H']) if 'W' in self.config and 'H' in self.config else None

        train_dataset = FastMRIDataset(
            self.config['train_path'],
            self.train_mask_func,
            target_size=target_size,
            augment=self.config.get('augment', False)
        )

        val_dataset = FastMRIDataset(
            self.config['val_path'],
            self.val_mask_func,
            target_size=target_size,
            augment=False
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,  # Process one file at a time
            shuffle=True,
            num_workers=self.config.get('num_workers', 4)
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,  # Process one file at a time
            shuffle=False,
            num_workers=self.config.get('num_workers', 4)
        )

        return train_loader, val_loader

    def _process_batch(self, batch, norm:Literal['max','meanstd']):
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
        kspace = batch['kspace'][0].to(self.device, non_blocking=True)
        masked_kspace = batch['masked_kspace'][0].to(self.device, non_blocking=True)
        mask = batch['mask'][0].to(self.device, non_blocking=True)

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

    def _forward_pass(self, kspace, masked_kspace, mask, image):
        """Perform forward pass using model or custom forward function.

        The forward_func is expected to return an image prediction (magnitude image),
        not a kspace prediction.
        """
        if self.forward_func is not None:
            # If a custom forward function is provided, use it
            # The forward function should handle the model and inputs
            # It should return an image prediction (magnitude image)
            k_space_pred, pred_image_abs = self.forward_func(kspace, masked_kspace, mask, image, self.model)
        else:
            # Otherwise use the model directly and convert to image domain
            kspace_pred = self.model(kspace, mask)
            kspace_dc   = mask * kspace + (~mask) * kspace_pred
            # Convert to image domain
            kspace_pred_permuted = kspace_dc.permute(0, 2, 3, 1)
            pred_image = fastmri.ifft2c(kspace_pred_permuted)
            pred_image_abs = fastmri.complex_abs(pred_image)

        return k_space_pred, pred_image_abs

    def train_epoch(self, dataloader, epoch, prev_ave_mse, norm:Literal['max','meanstd']='max'):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        running_kspace_loss = 0.0  # Kept for compatibility
        running_image_loss = 0.0  # Used for MSE loss
        running_ssim_loss = 0.0  # For SSIM loss
        running_psnr_loss = 0.0  # For PSNR loss
        total_slices = 0

        running_lr = 0.0

        # Create progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")

        n_batches = len(dataloader)
        batch_offset = epoch * n_batches

        for batch_idx, batch in enumerate(pbar):
            # Process batch
            kspace, masked_kspace, mask, image, target, normalization = self._process_batch(batch, norm)

            # Get number of slices
            n_slices = kspace.shape[0]
            total_slices += n_slices

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass - returns image domain prediction
            with torch.amp.autocast(self.device.type):
                k_space_pred, pred_image_abs = self._forward_pass(kspace, masked_kspace, mask, image)
                # if normalization['type'] == 'max':
                #     pred_image_abs = max_normalize(pred_image_abs, normalization['zf_max'])
                # elif normalization['type'] == 'meanstd':
                #     pred_image_abs = meanstd_normalize(pred_image_abs, normalization['mean'], normalization['std'], eps=1e-6)

                # Convert target kspace to image domain
                target_image_abs = target

                use_l1 = self.use_l1

                # Calculate loss
                loss, mse_loss, ssim_loss_val, psnr_loss_val = combined_loss(
                    k_space_pred,
                    kspace,
                    pred_image_abs,
                    target_image_abs,
                    self.psnr_loss,
                    ave_mse=prev_ave_mse,
                    mse_weight=self.config.get('mse_weight', 1.0),
                    ssim_weight=self.config.get('ssim_weight', 1000.0),
                    psnr_weight=self.config.get('psnr_weight', 1.0),
                    use_l1=use_l1
                )

            # Scale factor (for compatibility with original code)
            scale_factor = 1.0
            loss = loss * scale_factor

            # Backward pass and optimize
            # loss.backward()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.get('max_norm', 1.0))
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            # self.optimizer.step()

            # Update running losses - multiply by number of slices to calculate per-slice average later
            running_loss += (loss.item() / scale_factor) * n_slices  # Unscale for reporting
            running_kspace_loss += 0  # No longer using kspace loss
            running_image_loss += mse_loss.item() * n_slices  # Track MSE loss
            running_ssim_loss += ssim_loss_val.item() * n_slices  # Track SSIM loss
            running_psnr_loss += psnr_loss_val.item() * n_slices  # Track PSNR loss

            current_lr = self.optimizer.param_groups[0]['lr']
            running_lr += current_lr

            # Update progress bar - show per-slice metrics
            current_metrics = {
                'loss': running_loss / total_slices,
                'mse_loss': running_image_loss / total_slices,
                'ssim_loss': running_ssim_loss / total_slices,
                'psnr_loss': running_psnr_loss / total_slices,
                'lr': current_lr
            }
            pbar.set_postfix(current_metrics)
            if type(self.scheduler) == optim.lr_scheduler.CyclicLR:
                self.scheduler.step()

        # Calculate average losses per slice
        avg_loss = running_loss / total_slices if total_slices > 0 else 0
        avg_mse_loss = running_image_loss / total_slices if total_slices > 0 else 0
        avg_ssim_loss = running_ssim_loss / total_slices if total_slices > 0 else 0
        avg_psnr_loss = running_psnr_loss / total_slices if total_slices > 0 else 0
        avg_lr = running_lr / n_batches

        # Log epoch training metrics to wandb
        self.run.log({
            "train_loss": avg_loss,
            "train_mse_loss": avg_mse_loss,
            "train_ssim_loss": avg_ssim_loss,
            "train_psnr_loss": avg_psnr_loss,
            'learning_rate': avg_lr
        }, commit=False)

        return avg_loss, avg_mse_loss, avg_ssim_loss

    def validate(self, dataloader, prev_ave_mse, norm:Literal['max','meanstd']='max', plot=False):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        running_mse_loss = 0.0
        running_ssim_loss = 0.0
        running_psnr_loss = 0.0
        running_ssim = 0.0
        running_zf_ssim = 0.0
        running_psnr = 0.0
        running_zf_psnr = 0.0
        total_slices = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                # Process batch
                kspace, masked_kspace, mask, image, target, normalization = self._process_batch(batch, norm)

                # Get number of slices
                n_slices = kspace.shape[0]
                total_slices += n_slices

                # Forward pass - returns image domain prediction
                with torch.amp.autocast(self.device.type):
                    k_space_pred, pred_image_abs = self._forward_pass(kspace, masked_kspace, mask, image)
                # if normalization['type'] == 'max':
                #     pred_image_abs = max_normalize(pred_image_abs, normalization['zf_max'])
                # elif normalization['type'] == 'meanstd':
                #     pred_image_abs = meanstd_normalize(pred_image_abs, normalization['mean'], normalization['std'], eps=1e-6)

                    # Convert target kspace to image domain
                    target_image_abs = target

                    use_l1 = self.use_l1

                    # Calculate loss
                    loss, mse_loss, ssim_loss_val, psnr_loss_val = combined_loss(
                        k_space_pred,
                        kspace,
                        pred_image_abs,
                        target_image_abs,
                        self.psnr_loss,
                        ave_mse=prev_ave_mse,
                        mse_weight=self.config.get('mse_weight', 1.0),
                        ssim_weight=self.config.get('ssim_weight', 1000.0),
                        psnr_weight=self.config.get('psnr_weight', 1.0),
                        use_l1=use_l1
                    )

                # Convert to numpy for SSIM and PSNR calculation
                pred_image_abs_np = pred_image_abs.cpu().numpy()
                target_image_abs_np = target_image_abs.cpu().numpy()
                zf_abs_np = image.cpu().numpy()

                # Calculate SSIM for all slices at once
                # The ssim function expects 3D arrays and returns the average SSIM
                slice_ssim = ssim(target_image_abs_np, pred_image_abs_np)
                slice_ssim_zf = ssim(target_image_abs_np, zf_abs_np)
                running_ssim += slice_ssim * n_slices  # Multiply by n_slices since we're averaging later
                running_zf_ssim += slice_ssim_zf * n_slices

                # Calculate PSNR for all slices at once
                for i in range(n_slices):
                    slice_psnr = psnr(target_image_abs_np[i], pred_image_abs_np[i])
                    slice_psnr_zf = psnr(target_image_abs_np[i], zf_abs_np[i])
                    running_zf_psnr += slice_psnr_zf
                    running_psnr += slice_psnr


                # Update running losses
                running_loss += loss.item() * n_slices
                running_mse_loss += mse_loss.item() * n_slices
                running_ssim_loss += ssim_loss_val.item() * n_slices
                running_psnr_loss += psnr_loss_val.item() * n_slices

                if plot:
                    if self.plot_file is None:
                        self.plot_file = batch['file_path']
                    if self.plot_file != batch['file_path']:
                        continue
                    if self.plot_idx is None:
                        self.plot_idx = n_slices // 2
                    idx = self.plot_idx
                    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
                    ax[0].imshow(target[idx].cpu(), cmap="gray"); ax[0].set_title("target")
                    ax[1].imshow(image[idx].cpu(), cmap="gray"); ax[1].set_title("zf")
                    ax[2].imshow(pred_image_abs[idx].clamp(-6,6).cpu(), cmap="gray"); ax[2].set_title("pred")
                    diff = (target[idx] - pred_image_abs[idx]).abs()
                    ax[3].imshow(diff.cpu(), cmap="hot"); ax[3].set_title("|diff|")
                    for a in ax: a.axis("off")
                    self.run.log({"qualitative": wandb.Image(fig)}, commit=False)
                    plt.close(fig)

        # Calculate average losses
        avg_loss = running_loss / total_slices if total_slices > 0 else 0
        avg_mse_loss = running_mse_loss / total_slices if total_slices > 0 else 0
        avg_ssim_loss = running_ssim_loss / total_slices if total_slices > 0 else 0
        avg_psnr_loss = running_psnr_loss / total_slices if total_slices > 0 else 0
        avg_ssim = running_ssim / total_slices if total_slices > 0 else 0
        avg_zf_ssim = running_zf_ssim / total_slices if total_slices > 0 else 0
        avg_psnr = running_psnr / total_slices if total_slices > 0 else 0
        avg_zf_psnr = running_zf_psnr / total_slices if total_slices > 0 else 0

        # Log validation metrics to wandb
        self.run.log({
            "val_loss": avg_loss,
            "val_mse_loss": avg_mse_loss,
            "val_ssim_loss": avg_ssim_loss,
            "val_psnr_loss": avg_psnr_loss,
            "val_ssim": avg_ssim.item(),
            "val_ssim_gain": avg_ssim.item()-avg_zf_ssim.item(),
            "val_psnr": avg_psnr.item(),
            "val_psnr_gain": avg_psnr.item()-avg_zf_psnr.item()
        }, commit=False)

        return avg_loss, avg_mse_loss, avg_ssim_loss, avg_ssim.item(), avg_psnr.item(), avg_zf_ssim.item(), avg_zf_psnr.item()

    def train(self):
        """Main training loop."""
        # Get dataloaders
        train_loader, val_loader = self._get_dataloaders()

        # Training loop
        best_val_loss = float('inf')
        early_stopping = EarlyStopping(patience=self.config.get('terminate_patience', 10))

        norm = self.config.get('normalization', 'max')
        prev_ave_mse = 1e6
        mse_reset_boundaries = [10, 1, 0.5, 0.25, 0.1]

        # torch.autograd.set_detect_anomaly(True)

        for epoch in range(self.config['num_epochs']):
            current_lr = self.optimizer.param_groups[0]['lr']
            # Train for one epoch
            train_loss, train_mse_loss, train_ssim_loss = self.train_epoch(train_loader, epoch, prev_ave_mse, norm)

            # Validate
            val_loss, val_mse_loss, val_ssim_loss, val_ssim, val_psnr, val_zf_ssim, val_zf_psnr = self.validate(val_loader, prev_ave_mse, norm, plot=epoch % 10 == 0)
            if val_mse_loss < prev_ave_mse:
                prev_ave_mse = val_mse_loss
            if len(mse_reset_boundaries) > 0 and val_mse_loss < mse_reset_boundaries[0]:
                # while len(mse_reset_boundaries) > 0 and val_mse_loss < mse_reset_boundaries[0]:
                boundary = mse_reset_boundaries.pop(0)
                print(f'Detected crossed boundary MSE {boundary}, resetting early stopping and scheduler.')
                best_val_loss = float('inf')
                early_stopping.counter = 0
                early_stopping.best_loss = None
                if type(self.scheduler) == optim.lr_scheduler.ReduceLROnPlateau:
                    self.scheduler.num_bad_epochs = 0
                    self.scheduler.best = 1e6

            # Update learning rate
            if type(self.scheduler) == optim.lr_scheduler.ReduceLROnPlateau:
                self.scheduler.step(val_loss)

            # Print metrics with current learning rate
            print(f"Epoch {epoch+1}/{self.config['num_epochs']} - LR: {current_lr:.2e} - "
                  f"Train Loss: {train_loss:.6f} (MSE: {train_mse_loss:.6f}, SSIM-Loss: {train_ssim_loss:.6f}) - "
                  f"Val Loss: {val_loss:.6f} (MSE: {val_mse_loss:.6f}, SSIM-Loss: {val_ssim_loss:.6f}, SSIM: {val_ssim:.6f} ({val_zf_ssim:.6f}), PSNR: {val_psnr:.6f}({val_zf_psnr:.6f}) )")

            # Log learning rate to wandb
            # self.run.log({"learning_rate": current_lr})
            self.run.log({'epoch': epoch + 1})

            # Save checkpoint if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint('best_model.pt', epoch, train_loss, val_loss, val_ssim, val_psnr)
                print(f"Saved best model checkpoint with validation loss: {val_loss:.6f}, SSIM: {val_ssim:.6f}, PSNR: {val_psnr:.6f}")

            # Save checkpoint every N epochs
            if (epoch + 1) % self.config.get('save_checkpoint_every', 5) == 0:
                self._save_checkpoint(f'model_epoch_{epoch+1}.pt', epoch, train_loss, val_loss, val_ssim, val_psnr)
                print(f"Saved checkpoint at epoch {epoch+1}")

            if val_mse_loss < self.config.get('l1_transition_mse', 0.01):
                self.use_l1 = True
                early_stopping.counter = 0
                early_stopping.best_loss = None
                if type(self.scheduler) == optim.lr_scheduler.ReduceLROnPlateau:
                    self.scheduler.num_bad_epochs = 0
                    self.scheduler.best = 1e6

            if early_stopping(val_loss):
                print(f"Early stopping triggered, stopping training. Best val loss: {best_val_loss:.6f}")
                break


        print("Training complete.")
        self.run.finish()

    def _save_checkpoint(self, filename, epoch, train_loss, val_loss, val_ssim, val_psnr):
        """Save a checkpoint."""
        checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
        checkpoint_path = os.path.join(checkpoint_dir, filename)

        # Save the checkpoint locally
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_ssim': val_ssim,
            'val_psnr': val_psnr,
            'config': self.config
        }
        torch.save(checkpoint, checkpoint_path)

        # Log best model metrics
        if 'best' in filename:
            self.run.summary.update({
                "best_epoch": epoch + 1,
                "best_val_loss": val_loss,
                "best_val_ssim": val_ssim,
                "best_val_psnr": val_psnr
            })

    def __del__(self):
        """Cleanup when the trainer is deleted."""
        try:
            self.run.finish()
        except:
            pass
