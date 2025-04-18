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

            # Convert to tensor
            volume = T.to_tensor(volume_kspace)

            # Get current dimensions
            _, _, W, _ = volume.shape

            # Create mask based on the actual dimensions
            if self.mask_func:
                mask, _ = self.mask_func((1,) * len(volume.shape[:-3]) + tuple(volume.shape[-3:]), None)
                mask = mask.type(torch.bool)
            else:
                # If no mask function is provided, create a mask of all ones
                mask = torch.ones((1, 1, W, 1), dtype=torch.bool)

            # Permute volume to match model input format
            volume_input = volume.permute(0, 3, 1, 2)  # (n_slices, 2, H, W)

            # Center crop if target_size is provided
            if self.target_size is not None:
                target_W, target_H = self.target_size
                volume_input = centered_crop(volume_input, target_W, target_H)
                mask = centered_crop_1d(mask, target_W)

            # Create masked kspace by zeroing out columns where mask is 0
            masked_kspace = volume_input.clone()
            masked_kspace[:, :, :, ~mask.squeeze()] = 0

            # Store the original dimensions
            original_shape = volume_input.shape

            return {
                'kspace': volume_input,
                'masked_kspace': masked_kspace,
                'mask': mask,
                'file_path': file_path,
                'original_shape': original_shape
            }


def image_domain_losses(pred, target, use_l1=False):
    """
    Calculate both MSE and SSIM losses in image domain after converting from k-space
    Returns both losses separately
    """
    # Permute to match fastmri.ifft2c input format
    pred_permuted = pred.permute(0, 2, 3, 1)
    target_permuted = target.permute(0, 2, 3, 1)

    # Convert to image domain
    pred_image = fastmri.ifft2c(pred_permuted)
    target_image = fastmri.ifft2c(target_permuted)

    # Calculate absolute values
    pred_image_abs = fastmri.complex_abs(pred_image)
    target_image_abs = fastmri.complex_abs(target_image)

    # Get the maximum value for normalization and SSIM calculation
    target_max = torch.max(target_image_abs.view(target_image_abs.size(0), -1), dim=1)[0]

    # Calculate MSE loss
    # For MSE, normalize the images
    pred_image_abs_norm = pred_image_abs / target_max.view(-1, 1, 1, 1)
    target_image_abs_norm = target_image_abs / target_max.view(-1, 1, 1, 1)
    if use_l1:
        mse_loss = nn.L1Loss()(pred_image_abs_norm, target_image_abs_norm)
    else:
        mse_loss = nn.MSELoss()(pred_image_abs_norm, target_image_abs_norm)

    # Calculate SSIM loss
    # For SSIM loss, we need to keep the dimensions
    # SSIM module expects inputs of shape [batch, channel, height, width]
    # Add channel dimension since we have a single channel (magnitude image)
    pred_image_abs_ssim = pred_image_abs.unsqueeze(1)
    target_image_abs_ssim = target_image_abs.unsqueeze(1)
    ssim_loss_val = ssim_loss(target_image_abs_ssim, pred_image_abs_ssim, target_max)

    return mse_loss, ssim_loss_val


def combined_loss(pred, target, mse_weight=1.0, ssim_weight=1000.0, use_l1=False):
    """
    Combined loss using both MSE and SSIM losses in the image domain
    Returns the combined loss and individual losses for tracking
    """
    mse_loss, ssim_loss_val = image_domain_losses(pred, target, use_l1=use_l1)

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

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Move model to device
        self.model = self.model.to(self.device)

        tags = None

        if 'tags' in config:
            tags = config['tags']

        # Initialize wandb
        project_name = os.environ.get('WANDB_PROJECT_NAME', 'cs7643-fastmri')
        self.run = wandb.init(project=project_name, config=config, tags=tags)

        # Log model information
        self.run.name = f"{type(model).__name__}_{self.run.id}"
        self.run.save()

        # Log model architecture
        self.run.config.update({
            "model_name": type(model).__name__,
            "model_structure": str(model)
        })

        # Create mask function
        self.mask_func = RandomMaskFunc(
            center_fractions=self.config.get('center_fractions', [0.04]),
            accelerations=self.config.get('accelerations', [8]),
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

        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        self.min_learning_rate = self.config.get('min_learning_rate', 1e-6)

    def _get_dataloaders(self):
        """Create and return train and validation dataloaders."""
        # Create datasets with target size for cropping
        target_size = (self.config['W'], self.config['H']) if 'W' in self.config and 'H' in self.config else None

        train_dataset = FastMRIDataset(
            self.config['train_path'],
            self.mask_func,
            target_size=target_size
        )

        val_dataset = FastMRIDataset(
            self.config['val_path'],
            self.mask_func,
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
        # Get data from batch
        kspace = batch['kspace'][0].to(self.device)          # [0] because batch_size=1
        masked_kspace = batch['masked_kspace'][0].to(self.device)  # [0] because batch_size=1
        mask = batch['mask'][0].to(self.device)              # [0] because batch_size=1

        return kspace, masked_kspace, mask

    def _forward_pass(self, kspace, masked_kspace, mask):
        """Perform forward pass using model or custom forward function."""
        if self.forward_func is not None:
            # If a custom forward function is provided, use it
            # The forward function should handle the model and inputs
            outputs = self.forward_func(kspace, masked_kspace, mask, self.model)
        else:
            # Otherwise use the model directly
            outputs = self.model(kspace, mask)

        return outputs

    def train_epoch(self, dataloader, epoch):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        running_kspace_loss = 0.0  # Kept for compatibility
        running_image_loss = 0.0  # Used for MSE loss
        running_ssim_loss = 0.0  # For SSIM loss
        total_slices = 0

        # Create progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")

        for batch_idx, batch in enumerate(pbar):
            # Process batch
            kspace, masked_kspace, mask = self._process_batch(batch)

            # Get number of slices
            n_slices = kspace.shape[0]
            total_slices += n_slices

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self._forward_pass(kspace, masked_kspace, mask)

            use_l1 = self.config.get('use_l1', False)

            # Calculate loss
            loss, mse_loss, ssim_loss_val = combined_loss(
                outputs,
                kspace,
                mse_weight=self.config.get('mse_weight', 1.0),
                ssim_weight=self.config.get('ssim_weight', 1000.0),
                use_l1=use_l1
            )

            # Scale factor (for compatibility with original code)
            scale_factor = 1.0
            loss = loss * scale_factor

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Update running losses
            running_loss += loss.item() / scale_factor  # Unscale for reporting
            running_kspace_loss += 0  # No longer using kspace loss
            running_image_loss += mse_loss.item()  # Track MSE loss
            running_ssim_loss += ssim_loss_val.item()  # Track SSIM loss

            # Update progress bar
            current_metrics = {
                'loss': running_loss / (batch_idx + 1),
                'mse_loss': running_image_loss / (batch_idx + 1),
                'ssim_loss': running_ssim_loss / (batch_idx + 1),
                'slices': total_slices
            }
            pbar.set_postfix(current_metrics)

        # Calculate average losses
        avg_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0
        avg_mse_loss = running_image_loss / len(dataloader) if len(dataloader) > 0 else 0
        avg_ssim_loss = running_ssim_loss / len(dataloader) if len(dataloader) > 0 else 0

        # Log epoch training metrics to wandb
        self.run.log({
            "train_loss": avg_loss,
            "train_mse_loss": avg_mse_loss,
            "train_ssim_loss": avg_ssim_loss,
            "epoch": epoch + 1
        }, commit=False)

        return avg_loss, avg_mse_loss, avg_ssim_loss

    def validate(self, dataloader):
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
                kspace, masked_kspace, mask = self._process_batch(batch)

                # Get number of slices
                n_slices = kspace.shape[0]
                total_slices += n_slices

                # Forward pass
                outputs = self._forward_pass(kspace, masked_kspace, mask)

                use_l1 = self.config.get('use_l1', False)

                # Calculate loss
                loss, mse_loss, ssim_loss_val = combined_loss(
                    outputs,
                    kspace,
                    mse_weight=self.config.get('mse_weight', 1.0),
                    ssim_weight=self.config.get('ssim_weight', 1000.0),
                    use_l1=use_l1
                )

                # Calculate SSIM
                # Convert to image domain for SSIM calculation
                pred_permuted = outputs.permute(0, 2, 3, 1)
                target_permuted = kspace.permute(0, 2, 3, 1)

                # Convert to image domain
                pred_image = fastmri.ifft2c(pred_permuted)
                target_image = fastmri.ifft2c(target_permuted)

                # Calculate absolute values
                pred_image_abs = fastmri.complex_abs(pred_image).cpu().numpy()
                target_image_abs = fastmri.complex_abs(target_image).cpu().numpy()

                # Calculate SSIM for all slices at once
                # The ssim function expects 3D arrays and returns the average SSIM
                slice_ssim = ssim(target_image_abs, pred_image_abs)
                running_ssim += slice_ssim * n_slices  # Multiply by n_slices since we're averaging later

                # Calculate PSNR for all slices at once
                for i in range(n_slices):
                    slice_psnr = psnr(target_image_abs[i], pred_image_abs[i])
                    running_psnr += slice_psnr


                # Update running losses
                running_loss += loss.item() * n_slices
                running_mse_loss += mse_loss.item() * n_slices
                running_ssim_loss += ssim_loss_val.item() * n_slices

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

        for epoch in range(self.config['num_epochs']):
            current_lr = self.optimizer.param_groups[0]['lr']
            # Train for one epoch
            train_loss, train_mse_loss, train_ssim_loss = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_mse_loss, val_ssim_loss, val_ssim, val_psnr = self.validate(val_loader)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Print metrics with current learning rate
            print(f"Epoch {epoch+1}/{self.config['num_epochs']} - LR: {current_lr:.2e} - "
                  f"Train Loss: {train_loss:.6f} (MSE: {train_mse_loss:.6f}, SSIM-Loss: {train_ssim_loss:.6f}) - "
                  f"Val Loss: {val_loss:.6f} (MSE: {val_mse_loss:.6f}, SSIM-Loss: {val_ssim_loss:.6f}, SSIM: {val_ssim:.6f}, PSNR: {val_psnr:.6f})")

            # Log learning rate to wandb
            self.run.log({"learning_rate": current_lr})

            # Save checkpoint if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint('best_model.pt', epoch, train_loss, val_loss, val_ssim, val_psnr)
                print(f"Saved best model checkpoint with validation loss: {val_loss:.6f}, SSIM: {val_ssim:.6f}, PSNR: {val_psnr:.6f}")

            # Save checkpoint every N epochs
            if (epoch + 1) % self.config.get('save_checkpoint_every', 5) == 0:
                self._save_checkpoint(f'model_epoch_{epoch+1}.pt', epoch, train_loss, val_loss, val_ssim, val_psnr)
                print(f"Saved checkpoint at epoch {epoch+1}")

            if current_lr <= self.min_learning_rate:
                print(f"Reached minimum learning rate of {self.min_learning_rate}, stopping training.")
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
