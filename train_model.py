import h5py
import dotenv
import os
import glob
import fastmri
import fastmri.data.transforms as T
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from models.singlecoil_kspace_columnwise_masked_transformer_denoiser import SingleCoilKspaceColumnwiseMaskedTransformerDenoiser
from fastmri.data.subsample import RandomMaskFunc

# Load environment variables
dotenv.load_dotenv()

# Training and model hyperparameters
CONFIG = {
    # Data parameters
    'center_fractions': [0.04],
    'accelerations': [8],
    'seed': 42,

    # Model hyperparameters
    'encoder_num_heads': 1,
    'decoder_num_heads': 1,
    'pre_dims': 16,
    'pre_layers': 1,
    'hidden_size': 128,
    'activation': 'relu',
    'H': 320,
    'W': 320,

    # Training hyperparameters
    'batch_size': 1,  # Reduced batch size to avoid CUDA errors
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'kspace_loss_weight': 0.5,
    'image_loss_weight': 0.5,

    # Paths
    'train_path': os.environ.get('SINGLECOIL_TRAIN_PATH'),
    'val_path': os.environ.get('SINGLECOIL_VAL_PATH'),

    # Checkpointing
    'save_checkpoint_every': 5,
    'checkpoint_dir': 'checkpoints',
}

# Create checkpoint directory if it doesn't exist
os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

# Dataset class for FastMRI data
class FastMRIDataset(Dataset):
    def __init__(self, data_path, mask_func=None, target_size=(640, 372)):
        self.data_path = data_path
        self.mask_func = mask_func
        self.target_size = target_size  # Target size for H and W dimensions
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
                mask, _ = self.mask_func((1,) * len(volume.shape[:-3]) + tuple(volume.shape[-3:]), None, CONFIG['seed'])
                mask = mask.type(torch.bool)
            else:
                # If no mask function is provided, create a mask of all ones
                mask = torch.ones((1, 1, W, 1), dtype=torch.bool)

            # Permute volume to match model input format
            volume_input = volume.permute(0, 3, 1, 2)  # (n_slices, 2, H, W)

            # Store the original dimensions
            original_shape = volume_input.shape

            return {
                'kspace': volume_input,
                'mask': mask,
                'file_path': file_path,
                'original_shape': original_shape
            }

# Loss functions
def kspace_loss(pred, target):
    """
    Calculate MSE loss in k-space domain
    """
    return nn.MSELoss()(pred, target)

def image_domain_loss(pred, target):
    """
    Calculate MSE loss in image domain after converting from k-space
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

    return nn.MSELoss()(pred_image_abs, target_image_abs)

def combined_loss(pred, target, kspace_weight=0.5, image_weight=0.5):
    """
    Combined loss from k-space and image domain
    """
    k_loss = kspace_loss(pred, target)
    img_loss = image_domain_loss(pred, target)

    return kspace_weight * k_loss + image_weight * img_loss, k_loss, img_loss

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

# Training function
def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    running_kspace_loss = 0.0
    running_image_loss = 0.0
    total_slices = 0

    # Create progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")

    for batch_idx, batch in enumerate(pbar):
        # Get data from batch
        kspace = batch['kspace'][0].to(device)  # [0] because batch_size=1
        mask = batch['mask'][0].to(device)      # [0] because batch_size=1

        # center crop to target dimensions
        kspace = centered_crop(kspace, CONFIG['W'], CONFIG['H'])
        mask = centered_crop_1d(mask, CONFIG['W'])

        # Process in smaller chunks if needed
        batch_size = CONFIG['batch_size']
        n_slices = kspace.shape[0]
        total_slices += n_slices

        # Print shapes for debugging
        # print(f"Volume shape: {kspace.shape}, Mask shape: {mask.shape}")

        # Process the volume in chunks of batch_size slices
        # for i in range(0, n_slices, batch_size):
        #     try:
        #         # Get a chunk of slices
        #         end_idx = min(i + batch_size, n_slices)
        #         kspace_chunk = kspace[i:end_idx]

                # print(f"Processing chunk {i}:{end_idx} with shape {kspace_chunk.shape}")

                # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(kspace, mask)

        # Calculate loss
        loss, k_loss, img_loss = combined_loss(
            outputs,
            kspace,
            kspace_weight=CONFIG['kspace_loss_weight'],
            image_weight=CONFIG['image_loss_weight']
        )
            # except RuntimeError as e:
            #     print(f"Error processing chunk {i}:{end_idx}: {e}")
            #     print(f"Skipping this chunk and continuing...")
            #     continue

            # Scale the loss by the ratio of slices in this chunk to the batch size
            # This ensures that smaller chunks don't get disproportionate weight
        scale_factor = 1.0
        loss = loss * scale_factor

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update running losses
        running_loss += loss.item() / scale_factor  # Unscale for reporting
        running_kspace_loss += k_loss.item()
        running_image_loss += img_loss.item()

        # Update progress bar after processing the entire volume
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'k_loss': running_kspace_loss / (batch_idx + 1),
            'img_loss': running_image_loss / (batch_idx + 1),
            'slices': total_slices
        })

    # Calculate average losses (weighted by number of slices)
    avg_loss = running_loss / total_slices if total_slices > 0 else 0
    avg_kspace_loss = running_kspace_loss / total_slices if total_slices > 0 else 0
    avg_image_loss = running_image_loss / total_slices if total_slices > 0 else 0

    return avg_loss, avg_kspace_loss, avg_image_loss

# Validation function
def validate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    running_kspace_loss = 0.0
    running_image_loss = 0.0
    total_slices = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Get data from batch
            kspace = batch['kspace'][0].to(device)  # [0] because batch_size=1
            mask = batch['mask'][0].to(device)      # [0] because batch_size=1

            # center crop to target dimensions
            kspace = centered_crop(kspace, CONFIG['W'], CONFIG['H'])
            mask = centered_crop_1d(mask, CONFIG['W'])

            # Print shapes for debugging
            # print(f"Validation - Volume shape: {kspace.shape}, Mask shape: {mask.shape}")

            n_slices = kspace.shape[0]
            total_slices += n_slices

            # Process the volume in chunks of batch_size slices
            # for i in range(0, n_slices, batch_size):
                # try:
                    # Get a chunk of slices
                    # end_idx = min(i + batch_size, n_slices)
                    # kspace_chunk = kspace[i:end_idx]

                    # print(f"Validation - Processing chunk {i}:{end_idx} with shape {kspace_chunk.shape}")

                    # Forward pass
            outputs = model(kspace, mask)

                    # Calculate loss
            loss, k_loss, img_loss = combined_loss(
                outputs,
                kspace,
                kspace_weight=CONFIG['kspace_loss_weight'],
                image_weight=CONFIG['image_loss_weight']
            )
                # except RuntimeError as e:
                #     print(f"Error in validation for chunk {i}:{end_idx}: {e}")
                #     print(f"Skipping this chunk and continuing...")
                #     continue

                # Update running losses
            running_loss += loss.item() * n_slices
            running_kspace_loss += k_loss.item() * n_slices
            running_image_loss += img_loss.item() * n_slices

    # Calculate average losses (weighted by number of slices)
    avg_loss = running_loss / total_slices
    avg_kspace_loss = running_kspace_loss / total_slices
    avg_image_loss = running_image_loss / total_slices

    return avg_loss, avg_kspace_loss, avg_image_loss

# Function to test if the model can handle a sample input
def test_model_with_sample(model, device):
    print("Testing model with sample input...")
    try:
        # Create a sample input with the expected dimensions
        sample_kspace = torch.randn(1, 2, CONFIG['H'], CONFIG['W']).to(device)  # 1 slice
        sample_mask = torch.ones(1, 1, CONFIG['W'], 1, dtype=torch.bool).to(device)

        # Try a forward pass
        with torch.no_grad():
            output = model(sample_kspace, sample_mask)

        print(f"Model successfully processed sample input with shape {sample_kspace.shape}")
        print(f"Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"Error testing model with sample input: {e}")
        return False

# Main training function
def train_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create mask function
    mask_func = RandomMaskFunc(
        center_fractions=CONFIG['center_fractions'],
        accelerations=CONFIG['accelerations']
    )

    # Create datasets and dataloaders
    train_dataset = FastMRIDataset(CONFIG['train_path'], mask_func)
    val_dataset = FastMRIDataset(CONFIG['val_path'], mask_func)

    # Use batch_size=1 to process one file at a time
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # Process one file at a time
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Process one file at a time
        shuffle=False,
        num_workers=4
    )

    # Initialize model with smaller batch size for testing
    # First try with default parameters
    model = SingleCoilKspaceColumnwiseMaskedTransformerDenoiser(
        encoder_num_heads=CONFIG['encoder_num_heads'],
        decoder_num_heads=CONFIG['decoder_num_heads'],
        pre_dims=CONFIG['pre_dims'],
        pre_layers=CONFIG['pre_layers'],
        hidden_size=CONFIG['hidden_size'],
        activation=CONFIG['activation'],
        H=CONFIG['H'],
        W=CONFIG['W']
    ).to(device)

    print(f"Model initialized with parameters: H={CONFIG['H']}, W={CONFIG['W']}")

    # Test if the model can handle a sample input
    if not test_model_with_sample(model, device):
        raise RuntimeError("Model failed to process sample input with default parameters")

    
    print(f"Note: The model will handle different input sizes by processing in chunks.")

    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

    # Initialize learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(CONFIG['num_epochs']):
        # Train for one epoch
        train_loss, train_kspace_loss, train_image_loss = train_epoch(
            model, train_loader, optimizer, device, epoch
        )

        # Validate
        val_loss, val_kspace_loss, val_image_loss = validate(model, val_loader, device)

        # Update learning rate
        scheduler.step(val_loss)

        # Print metrics
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']} - "
            f"Train Loss: {train_loss:.6f} (K: {train_kspace_loss:.6f}, Img: {train_image_loss:.6f}) - "
            f"Val Loss: {val_loss:.6f} (K: {val_kspace_loss:.6f}, Img: {val_image_loss:.6f})")

        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': CONFIG
            }, os.path.join(CONFIG['checkpoint_dir'], 'best_model.pt'))
            print(f"Saved best model checkpoint with validation loss: {val_loss:.6f}")

        # Save checkpoint every N epochs
        if (epoch + 1) % CONFIG['save_checkpoint_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': CONFIG
            }, os.path.join(CONFIG['checkpoint_dir'], f'model_epoch_{epoch+1}.pt'))
            print(f"Saved checkpoint at epoch {epoch+1}")



if __name__ == "__main__":
    train_model()
