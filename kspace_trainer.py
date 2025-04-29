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
import wandb
import matplotlib.pyplot as plt


def minmax_normalize(image, min_val, scale, min_clip=0, max_clip=1):
    return ((image - min_val) / scale)#.clamp(min_clip, max_clip)

def max_normalize(image, max_value):
    return (image / max_value.view(-1, 1, 1))

class FastMRIDataset(Dataset):
    def __init__(self, data_path, mask_func=None, target_size=None):
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
        print(f"Found {len(self.file_list)} files in {data_path}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        with h5py.File(file_path, 'r') as hf:
            volume_kspace = hf['kspace'][()]

            kspace = T.to_tensor(volume_kspace)
            if self.target_size is not None:
                kspace = T.complex_center_crop(kspace, self.target_size)

            if self.mask_func is not None:
                masked_kspace, mask, _ = T.apply_mask(kspace, self.mask_func)
            else:
                masked_kspace, mask = kspace, torch.ones_like(kspace[..., :1])

            kspace = kspace.permute(0, 3, 1, 2)
            masked_kspace = masked_kspace.permute(0, 3, 1, 2)

            mask = mask.to(torch.bool)[:, 0:1, :, 0:1]

            return {
                'kspace': kspace,
                'masked_kspace': masked_kspace,
                'mask': mask,
                'file_path': file_path,
            }


def image_losses(pred_image_abs, target_image_abs, zf_max, use_l1=False):
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

    # Calculate SSIM loss
    # For SSIM loss, we need to keep the dimensions
    # SSIM module expects inputs of shape [batch, channel, height, width]
    # Add channel dimension since we have a single channel (magnitude image)
    # pred_image_abs_ssim = pred_image_abs.unsqueeze(1)
    # target_image_abs_ssim = target_image_abs.unsqueeze(1)
    pred_image_abs_ssim = pred_image_abs
    target_image_abs_ssim = target_image_abs
    ssim_loss_val = ssim_loss(target_image_abs_ssim, pred_image_abs_ssim, torch.ones_like(zf_max, device=zf_max.device))

    return mse_loss, ssim_loss_val


def combined_loss(pred_image_abs, target_image_abs, zf_max, mse_weight=1.0, ssim_weight=1000.0, use_l1=False):
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
    mse_loss, ssim_loss_val = image_losses(pred_image_abs, target_image_abs, zf_max, use_l1=use_l1)

    # Combine losses with their respective weights
    total_loss = mse_weight * mse_loss + ssim_weight * ssim_loss_val

    return total_loss, mse_loss, ssim_loss_val


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

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                patience=scheduler_args.get('patience', 5)
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

        print(f"Using scheduler: {scheduler_type}")
        print(f"Scheduler config: {scheduler_args}")

    def _get_dataloaders(self):
        """Create and return train and validation dataloaders."""
        # Create datasets with target size for cropping
        target_size = (self.config['W'], self.config['H']) if 'W' in self.config and 'H' in self.config else None

        train_dataset = FastMRIDataset(
            self.config['train_path'],
            self.train_mask_func,
            target_size=target_size
        )

        val_dataset = FastMRIDataset(
            self.config['val_path'],
            self.val_mask_func,
            target_size=target_size
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

    def _process_batch(self, batch):
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
            zf_abs = fastmri.complex_abs(fastmri.ifft2c(masked_kspace.permute(0,2,3,1)))
            target = fastmri.complex_abs(fastmri.ifft2c(kspace.permute(0,2,3,1)))
            zf_max = zf_abs.view(zf_abs.size(0), -1).amax(dim=1).clamp_min_(1e-6)
            zf_abs = max_normalize(zf_abs, zf_max)
            target = max_normalize(target, zf_max)

        return kspace, masked_kspace, mask, zf_abs, target, zf_max

    def _forward_pass(self, kspace, masked_kspace, mask, image):
        """Perform forward pass using model or custom forward function.

        The forward_func is expected to return an image prediction (magnitude image),
        not a kspace prediction.
        """
        if self.forward_func is not None:
            # If a custom forward function is provided, use it
            # The forward function should handle the model and inputs
            # It should return an image prediction (magnitude image)
            pred_image_abs = self.forward_func(kspace, masked_kspace, mask, image, self.model)
        else:
            # Otherwise use the model directly and convert to image domain
            kspace_pred = self.model(kspace, mask)
            # Convert to image domain
            kspace_pred_permuted = kspace_pred.permute(0, 2, 3, 1)
            pred_image = fastmri.ifft2c(kspace_pred_permuted)
            pred_image_abs = fastmri.complex_abs(pred_image)

        return pred_image_abs

    def train_epoch(self, dataloader, epoch):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        running_kspace_loss = 0.0  # Kept for compatibility
        running_image_loss = 0.0  # Used for MSE loss
        running_ssim_loss = 0.0  # For SSIM loss
        total_slices = 0

        running_lr = 0.0

        # Create progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")

        n_batches = len(dataloader)
        batch_offset = epoch * n_batches

        for batch_idx, batch in enumerate(pbar):
            # Process batch
            kspace, masked_kspace, mask, image, target, zf_max = self._process_batch(batch)

            # Get number of slices
            n_slices = kspace.shape[0]
            total_slices += n_slices

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass - returns image domain prediction
            # print('Image shape in train_epoch - pred 1 %s', image.shape)
            pred_image_abs = self._forward_pass(kspace, masked_kspace, mask, image)
            # print('Image shape in train_epoch - pred 2 %s', pred_image_abs.shape)
            # pred_image_abs = max_normalize(pred_image_abs, zf_max)
            # print('Image shape in train_epoch - pred 3 %s', pred_image_abs.shape)

            # Convert target kspace to image domain
            target_image_abs = target

            if target_image_abs.ndim == 3:
                target_image_abs = target_image_abs.unsqueeze(1)

            # print('Image shape in train_epoch - pred %s', pred_image_abs.shape)
            # print('Image shape in train_epoch - target %s', target_image_abs.shape)

            use_l1 = self.config.get('use_l1', False)

            # Calculate loss
            loss, mse_loss, ssim_loss_val = combined_loss(
                pred_image_abs,
                target_image_abs,
                zf_max,
                mse_weight=self.config.get('mse_weight', 1.0),
                ssim_weight=self.config.get('ssim_weight', 1000.0),
                use_l1=use_l1
            )

            # Scale factor (for compatibility with original code)
            scale_factor = 1.0
            loss = loss * scale_factor

            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.get('max_norm', 1.0))
            self.optimizer.step()

            # Update running losses - multiply by number of slices to calculate per-slice average later
            running_loss += (loss.item() / scale_factor) * n_slices  # Unscale for reporting
            running_kspace_loss += 0  # No longer using kspace loss
            running_image_loss += mse_loss.item() * n_slices  # Track MSE loss
            running_ssim_loss += ssim_loss_val.item() * n_slices  # Track SSIM loss

            current_lr = self.optimizer.param_groups[0]['lr']
            running_lr += current_lr

            # Update progress bar - show per-slice metrics
            current_metrics = {
                'loss': running_loss / total_slices,
                'mse_loss': running_image_loss / total_slices,
                'ssim_loss': running_ssim_loss / total_slices,
                'slices': total_slices,
                'lr': current_lr
            }
            pbar.set_postfix(current_metrics)
            if type(self.scheduler) == optim.lr_scheduler.CyclicLR:
                self.scheduler.step()

        # Calculate average losses per slice
        avg_loss = running_loss / total_slices if total_slices > 0 else 0
        avg_mse_loss = running_image_loss / total_slices if total_slices > 0 else 0
        avg_ssim_loss = running_ssim_loss / total_slices if total_slices > 0 else 0
        avg_lr = running_lr / n_batches

        # Log epoch training metrics to wandb
        self.run.log({
            "train_loss": avg_loss,
            "train_mse_loss": avg_mse_loss,
            "train_ssim_loss": avg_ssim_loss,
            'learning_rate': avg_lr
        }, commit=False)

        return avg_loss, avg_mse_loss, avg_ssim_loss

    def validate(self, dataloader, plot=False):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        running_mse_loss = 0.0
        running_ssim_loss = 0.0
        running_ssim = 0.0
        running_psnr = 0.0
        total_slices = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                # Process batch
                kspace, masked_kspace, mask, image, target, zf_max = self._process_batch(batch)

                # Get number of slices
                n_slices = kspace.shape[0]
                total_slices += n_slices

                # Forward pass - returns image domain prediction
                pred_image_abs = self._forward_pass(kspace, masked_kspace, mask, image)
                pred_image_abs = max_normalize(pred_image_abs, zf_max)

                # Convert target kspace to image domain
                target_image_abs = target

                use_l1 = self.config.get('use_l1', False)

                # Calculate loss
                loss, mse_loss, ssim_loss_val = combined_loss(
                    pred_image_abs,
                    target_image_abs,
                    zf_max,
                    mse_weight=self.config.get('mse_weight', 1.0),
                    ssim_weight=self.config.get('ssim_weight', 1000.0),
                    use_l1=use_l1
                )

                # Convert to numpy for SSIM and PSNR calculation
                pred_image_abs_np = pred_image_abs.cpu().numpy()
                target_image_abs_np = target_image_abs.cpu().numpy()

                # Calculate SSIM for all slices at once
                # The ssim function expects 3D arrays and returns the average SSIM
                slice_ssim = ssim(target_image_abs_np, pred_image_abs_np)
                running_ssim += slice_ssim * n_slices  # Multiply by n_slices since we're averaging later

                # Calculate PSNR for all slices at once
                for i in range(n_slices):
                    slice_psnr = psnr(target_image_abs_np[i], pred_image_abs_np[i])
                    running_psnr += slice_psnr


                # Update running losses
                running_loss += loss.item() * n_slices
                running_mse_loss += mse_loss.item() * n_slices
                running_ssim_loss += ssim_loss_val.item() * n_slices

                if plot:
                    if self.plot_file is None:
                        self.plot_file = batch['file_path']
                    if self.plot_file != batch['file_path']:
                        continue
                    if self.plot_idx is None:
                        self.plot_idx = n_slices // 2
                    idx = self.plot_idx
                    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
                    ax[0].imshow(target[idx].cpu(), cmap="gray"); ax[0].set_title("target")
                    ax[1].imshow(pred_image_abs[idx].cpu(), cmap="gray"); ax[1].set_title("pred")
                    diff = (target[idx] - pred_image_abs[idx]).abs()
                    ax[2].imshow(diff.cpu(), cmap="hot"); ax[2].set_title("|diff|")
                    for a in ax: a.axis("off")
                    self.run.log({"qualitative": wandb.Image(fig)}, commit=False)
                    plt.close(fig)

        # Calculate average losses
        avg_loss = running_loss / total_slices if total_slices > 0 else 0
        avg_mse_loss = running_mse_loss / total_slices if total_slices > 0 else 0
        avg_ssim_loss = running_ssim_loss / total_slices if total_slices > 0 else 0
        avg_ssim = running_ssim / total_slices if total_slices > 0 else 0
        avg_psnr = running_psnr / total_slices if total_slices > 0 else 0

        # Log validation metrics to wandb
        self.run.log({
            "val_loss": avg_loss,
            "val_mse_loss": avg_mse_loss,
            "val_ssim_loss": avg_ssim_loss,
            "val_ssim": avg_ssim.item(),
            "val_psnr": avg_psnr.item()
        }, commit=False)

        return avg_loss, avg_mse_loss, avg_ssim_loss, avg_ssim.item(), avg_psnr.item()

    def train(self):
        """Main training loop."""
        # Get dataloaders
        train_loader, val_loader = self._get_dataloaders()

        # Training loop
        best_val_loss = float('inf')
        early_stopping = EarlyStopping(patience=self.config.get('terminate_patience', 10))

        # torch.autograd.set_detect_anomaly(True)

        for epoch in range(self.config['num_epochs']):

            torch.cuda.empty_cache()

            current_lr = self.optimizer.param_groups[0]['lr']
            # Train for one epoch
            train_loss, train_mse_loss, train_ssim_loss = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_mse_loss, val_ssim_loss, val_ssim, val_psnr = self.validate(val_loader, epoch % 10 == 0)

            # Update learning rate
            if type(self.scheduler) == optim.lr_scheduler.ReduceLROnPlateau:
                self.scheduler.step(val_loss)

            # Print metrics with current learning rate
            print(f"Epoch {epoch+1}/{self.config['num_epochs']} - LR: {current_lr:.2e} - "
                  f"Train Loss: {train_loss:.6f} (MSE: {train_mse_loss:.6f}, SSIM-Loss: {train_ssim_loss:.6f}) - "
                  f"Val Loss: {val_loss:.6f} (MSE: {val_mse_loss:.6f}, SSIM-Loss: {val_ssim_loss:.6f}, SSIM: {val_ssim:.6f}, PSNR: {val_psnr:.6f})")

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
