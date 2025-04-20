import sys
import logging
import pathlib
import random
import shutil
import time
import functools
import numpy as np
import argparse
import os
from dotenv import load_dotenv

import torch
import torchvision
# Removed: from tensorboardX import SummaryWriter # Replaced by wandb
import wandb # Added wandb
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

# Ensure these imports are correct for your project structure
from dataset import SliceData, SliceDisplayDataDev
from models import OCUCFormer
import torchvision
from torch import nn
from torch.autograd import Variable
from torch import optim
from tqdm import tqdm

# --- Configuration ---
# Added wandb specific configurations
# Make sure to set WANDB_PROJECT and WANDB_ENTITY environment variables
# or change the project/entity values here.
CONFIG = {
    # == WandB Project Configuration ==
    'project': os.getenv('WANDB_PROJECT', 'cs7643-fastmri-ocucformer'), # Wandb project name
    'entity': os.getenv('WANDB_ENTITY'), # Wandb entity (username or team)
    'tags': ['ocucformer', 'singlecoil', 'baseline'], # Example tags
    'notes': 'Training OCUCFormer baseline model on single-coil knee data.', # Example notes

    # == Experiment Directory & Checkpointing ==
    'checkpoint': 'checkpoints_sc',
    'resume': False,
    'resume_checkpoint_path': None,
    'exp_dir': 'checkpoints_sc', # Directory for checkpoints and logs

    # == Data & Mask Parameters ==
    'dataset_type': 'knee_singlecoil',
    'mask_type': 'cartesian',
    'acceleration_factor': "4x", # Example, ensure this matches your data/masks

    # == Training Hyperparameters ==
    'batch_size': 1, # Default: 2 - Adjusted based on previous config
    'num_epochs': 2,
    'lr': 0.001,
    'lr_step_size': 40,
    'lr_gamma': 0.1,
    'weight_decay': 0.0,
    'seed': 42,
    'report_interval': 100, # Log loss within epoch interval

    # == Model Hyperparameters ==
    'num_pools': 2, # Default: 4 (Used in OCUCFormer if applicable)
    'drop_prob': 0.0, # Used in OCUCFormer if applicable
    'num_chans': 2, # Default: 32 (Used in OCUCFormer if applicable)
    'timesteps': 2, # Default: 5 (Used in OCUCFormer)

    # == Other Settings ==
    'data_parallel': False,
    'device': 'cuda',
}


def create_datasets(args):
    # This function expects args.train_path and args.validation_path to be Path objects
    acc_factors = args.acceleration_factor.split(',')
    mask_types = args.mask_type.split(',')
    dataset_types = args.dataset_type.split(',')
    usmask_path = args.usmask_path

    train_data = SliceData(args.train_path, acc_factors, dataset_types, mask_types, 'train', usmask_path)
    dev_data = SliceData(args.validation_path, acc_factors, dataset_types, mask_types, 'validation', usmask_path)
    display1_data = SliceDisplayDataDev(args.validation_path, dataset_types[0], mask_types[0], acc_factors[0], usmask_path)
    return dev_data, train_data, display1_data


def create_data_loaders(args):
    dev_data, train_data, display1_data = create_datasets(args)
    display_indices = range(0, len(display1_data), max(1, len(display1_data) // 16))
    display1_subset = torch.utils.data.Subset(display1_data, display_indices)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        # num_workers=4, # Consider adding based on system capabilities
        # pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        # num_workers=4,
        # pin_memory=True,
    )
    display_loader1 = DataLoader(
        dataset=display1_subset,
        batch_size=min(args.batch_size, 16),
        shuffle=False,
        # num_workers=4,
        # pin_memory=True,
    )
    return train_loader, dev_loader, display_loader1


def train_epoch(args, epoch, model, data_loader, optimizer): # Removed writer

    model.train()
    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    loop = tqdm(data_loader, desc=f"Epoch {epoch} Train")
    current_lr = optimizer.param_groups[0]['lr'] # Get current learning rate

    for iter, data in enumerate(loop):
        input_img, input_kspace, target_img, mask = data # Renamed for clarity

        input_img = input_img.unsqueeze(1).to(args.device).float()
        input_kspace = input_kspace.to(args.device).float() # Assuming complex stored as 2 channels
        target_img = target_img.unsqueeze(1).to(args.device).float()
        mask = mask.to(args.device).float()

        output_img = model(input_img, input_kspace, mask)
        target_h, target_w = target_img.shape[2], target_img.shape[3] # Should be 320, 320

        # Check if output shape differs from target shape
        if output_img.shape[2] != target_h or output_img.shape[3] != target_w:
             # Use torchvision's functional center_crop
             # This assumes output_img is larger or equal in size to target_img
             # If output could be smaller, add checks or use resizing instead.
             try:
                 output_img_cropped = TF.center_crop(output_img, (target_h, target_w))
                 # Optional: Log a warning only once to avoid spamming logs
                 # if iter == 0 and epoch == start_epoch: # Assuming start_epoch is available
                 #    logging.warning(f"Model output shape {output_img.shape} differs from target shape {target_img.shape}. Cropping output for loss calculation.")
             except Exception as crop_error:
                 logging.error(f"Error cropping output tensor: {crop_error}. Output shape: {output_img.shape}, Target shape: {target_img.shape}")
                 # Handle error appropriately, e.g., skip batch or raise error
                 continue # Skip this batch if cropping fails
        else:
             output_img_cropped = output_img # No cropping needed if shapes match

        if output_img_cropped.shape != target_img.shape:
             logging.error(f"Shape mismatch AFTER cropping! Output: {output_img_cropped.shape}, Target: {target_img.shape}. Skipping loss calculation.")
             continue # Skip if shapes still don't match

        loss = F.l1_loss(output_img_cropped, target_img) # Calculate loss on cropped output
        # --- END FIX ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()

        # --- Wandb Logging within epoch ---
        if iter % args.report_interval == 0:
            wandb.log({
                'train/batch_loss': loss.item(),
                'train/avg_loss': avg_loss,
                'train/epoch': epoch,
                'train/batch_step': global_step + iter,
                'train/lr': current_lr
            })
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'LR = {current_lr:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
        loop.set_postfix({'AvgLoss': avg_loss, 'LR': current_lr})

    return avg_loss, time.perf_counter() - start_epoch, current_lr


def evaluate(args, epoch, model, data_loader): # Removed writer

    model.eval()
    losses = [] # Track L1 loss
    # Add other metrics if needed (e.g., PSNR, SSIM)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader, desc=f"Epoch {epoch} Eval")):
            input_img, input_kspace, target_img, mask = data

            input_img = input_img.unsqueeze(1).to(args.device).float()
            input_kspace = input_kspace.to(args.device).float()
            target_img = target_img.unsqueeze(1).to(args.device).float()
            mask = mask.to(args.device).float()

            output_img = model(input_img, input_kspace, mask)

            loss = F.l1_loss(output_img, target_img)
            losses.append(loss.item())
            # TODO: Calculate PSNR and SSIM here if desired

        avg_dev_loss = np.mean(losses)
        # avg_psnr = np.mean(psnr_scores) # Calculate if implemented
        # avg_ssim = np.mean(ssim_scores) # Calculate if implemented

        # --- Wandb Logging for epoch evaluation ---
        wandb.log({
            'val/epoch_loss': avg_dev_loss,
            # 'val/psnr': avg_psnr, # Log if calculated
            # 'val/ssim': avg_ssim, # Log if calculated
            'val/epoch': epoch,
        })

    return avg_dev_loss, time.perf_counter() - start


def visualize(args, epoch, model, data_loader, datasettype_string): # Removed writer

    def log_wandb_image(image_tensor, caption):
        # Ensure tensor is detached, on CPU, and normalized for wandb
        image_tensor = image_tensor.detach().cpu()
        # Normalize each image in the batch individually for better visualization
        for i in range(image_tensor.shape[0]):
             img = image_tensor[i]
             img = (img - img.min()) / (img.max() - img.min() + 1e-6)
             image_tensor[i] = img

        grid = torchvision.utils.make_grid(image_tensor, nrow=4, pad_value=1)
        # Log image to wandb - use step=epoch
        wandb.log({f"val/images/{caption}": wandb.Image(grid, caption=f"{caption} - Epoch {epoch}")}, step=epoch)


    model.eval()
    with torch.no_grad():
        # Only visualize one batch
        for iter, data in enumerate(tqdm(data_loader, desc=f"Epoch {epoch} Visualize")):
            input_img, input_kspace, target_img, mask = data
            input_img = input_img.unsqueeze(1).to(args.device).float()
            input_kspace = input_kspace.to(args.device).float()
            target_img = target_img.unsqueeze(1).to(args.device).float()
            mask = mask.to(args.device).float()

            output_img = model(input_img, input_kspace, mask)

            log_wandb_image(input_img, f'Input_{datasettype_string}')
            log_wandb_image(target_img, f'Target_{datasettype_string}')
            log_wandb_image(output_img, f'Reconstruction_{datasettype_string}')
            log_wandb_image(torch.abs(target_img - output_img), f'Error_{datasettype_string}')
            break # Only log images from the first batch

def save_model(exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best, args_dict):
    # Added args_dict to save config
    if not isinstance(exp_dir, pathlib.Path):
        exp_dir = pathlib.Path(exp_dir)

    save_path = exp_dir / 'model.pt'
    best_save_path = exp_dir / 'best_model.pt'

    # Prepare model state dict (handle DataParallel)
    if isinstance(model, torch.nn.DataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    torch.save(
        {
            'epoch': epoch,
            'args': args_dict, # Save args/config dictionary
            'model': model_state_dict,
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': str(exp_dir)
        },
        f=save_path
    )
    logging.info(f"Saved checkpoint to {save_path} (Epoch {epoch})")


    if is_new_best:
        shutil.copyfile(save_path, best_save_path)
        logging.info(f"Saved new best model to {best_save_path} (Epoch {epoch})")
        # --- Wandb: Save best model ---
        # wandb.save(str(best_save_path)) # Option 1: Save directly
        # Option 2: Use artifacts (more robust)
        try:
            artifact = wandb.Artifact(f'{wandb.run.name}-best-model', type='model')
            artifact.add_file(str(best_save_path))
            wandb.log_artifact(artifact, aliases=['best', f'epoch-{epoch}'])
            logging.info(f"Logged best model artifact to wandb (Epoch {epoch})")
        except Exception as e:
            logging.error(f"Failed to log best model artifact to wandb: {e}")


def build_model(args_ns): # Renamed arg to avoid conflict
    # Pass args Namespace to the model constructor
    # Ensure model definition matches the one used (OCUCFormer)
    model = OCUCFormer(args_ns, timesteps=args_ns.timesteps).to(args_ns.device)
    return model

def load_model(checkpoint_file):
    if not isinstance(checkpoint_file, pathlib.Path):
        checkpoint_file = pathlib.Path(checkpoint_file)

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    checkpoint = torch.load(checkpoint_file, map_location='cpu')

    # Recreate Namespace object from saved dictionary
    # Fallback to current CONFIG if 'args' not in checkpoint (older format)
    args_dict = checkpoint.get('args', CONFIG) # Use CONFIG as fallback
    args_ns = argparse.Namespace(**args_dict)

    # Update args device to current device if necessary
    args_ns.device = torch.device(CONFIG['device'] if torch.cuda.is_available() else 'cpu') # Use current config's device setting

    model = build_model(args_ns) # build_model uses the recreated args Namespace

    # Handle DataParallel state dict loading
    state_dict = checkpoint['model']
    # Check if the model was saved *with* the 'module.' prefix
    data_parallel_saved = any(k.startswith('module.') for k in state_dict.keys())

    if data_parallel_saved:
         # If saved with 'module.', remove it before loading if current model is not DP
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    else:
        # If saved without 'module.', load directly
         model.load_state_dict(state_dict)

    # Move model to device *before* potentially wrapping with DataParallel
    model.to(args_ns.device)

    # Wrap with DataParallel if specified in the *current* config
    if CONFIG['data_parallel']: # Check current config
        model = torch.nn.DataParallel(model)
        logging.info("Applying DataParallel wrapper during loading.")


    optimizer = build_optim(args_ns, model.parameters())
    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info("Loaded optimizer state from checkpoint.")
    except Exception as e:
        print(f"Could not load optimizer state_dict: {e}. Initializing new optimizer.")
        logging.warning(f"Could not load optimizer state_dict: {e}. Initializing new optimizer.")


    return checkpoint, model, optimizer, args_ns # Return updated args Namespace


def build_optim(args_ns, params): # Renamed arg
    # Use hyperparameters from the args Namespace
    optimizer = torch.optim.Adam(params, args_ns.lr, weight_decay=args_ns.weight_decay)
    return optimizer


def main(config_dict): # Accept config dictionary
    try:
        wandb.init(...) # Call init
        logging.info(...)
        wandb_active = True # Set flag if successful
    except Exception as e:
        logging.error(f"Failed to initialize Wandb: {e}. Proceeding without wandb logging.")
        wandb_active = False # Set flag if failed
    # Convert dict to Namespace for compatibility with existing functions
    args = argparse.Namespace(**config_dict)

    # --- Initialize Wandb ---
    try:
        wandb.init(
            project=args.project,
            entity=args.entity,
            config=config_dict, # Log the original config dictionary
            tags=args.tags,
            notes=args.notes,
            resume="allow", # Allow resuming if run_id is found
            # id=wandb_run_id, # Set this if you want to force resume a specific run
        )
        logging.info(f"Wandb initialized: project='{args.project}', entity='{args.entity}', run_id='{wandb.run.id}'")
        wandb_active = True
    except Exception as e:
        logging.error(f"Failed to initialize Wandb: {e}. Proceeding without wandb logging.")
        wandb_active = False


    # --- Load .env file ---
    load_dotenv()

    # --- Get paths from .env, fallback to args or raise error ---
    train_path_env = os.getenv('SINGLECOIL_TRAIN_PATH')
    val_path_env = os.getenv('SINGLECOIL_VAL_PATH')
    usmask_path_env = os.getenv('USMASK_PATH')

    args.train_path = pathlib.Path(train_path_env) if train_path_env else getattr(args, 'train_path', None)
    args.validation_path = pathlib.Path(val_path_env) if val_path_env else getattr(args, 'validation_path', None)
    args.usmask_path = pathlib.Path(usmask_path_env) if usmask_path_env else getattr(args, 'usmask_path', None)

    if not args.train_path: raise ValueError("Train path missing (.env: SINGLECOIL_TRAIN_PATH)")
    if not args.validation_path: raise ValueError("Validation path missing (.env: SINGLECOIL_VAL_PATH)")
    if not args.usmask_path: raise ValueError("Mask path missing (.env: USMASK_PATH)")

    if isinstance(args.train_path, str): args.train_path = pathlib.Path(args.train_path)
    if isinstance(args.validation_path, str): args.validation_path = pathlib.Path(args.validation_path)
    if isinstance(args.usmask_path, str): args.usmask_path = pathlib.Path(args.usmask_path)


    print(f"Using train path: {args.train_path}")
    print(f"Using validation path: {args.validation_path}")
    print(f"Using usmask path: {args.usmask_path}")


    if not args.train_path.exists(): raise FileNotFoundError(f"Train path does not exist: {args.train_path}")
    if not args.validation_path.exists(): raise FileNotFoundError(f"Validation path does not exist: {args.validation_path}")
    if not args.usmask_path.exists(): raise FileNotFoundError(f"Undersampling mask path does not exist: {args.usmask_path}")

    # --- Ensure exp_dir is Path object and exists ---
    args.exp_dir = pathlib.Path(args.exp_dir)
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    if wandb_active: # Save logs inside wandb run dir if possible
        log_dir = pathlib.Path(wandb.run.dir)
    else:
        log_dir = args.exp_dir
    log_file_path = log_dir / 'train_ocucformer_sc.log' # Consistent log file name
    # Setup logging (consider rotating file handler for long runs)
    logging.basicConfig(filename=log_file_path, filemode='a' if args.resume else 'w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Add console handler as well
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger('').addHandler(console_handler)


    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {args.device}")
    logging.info(f"Run Config: {config_dict}") # Log the config

    # Handle checkpoint path for resume
    checkpoint_path_to_load = None
    if args.resume:
        if args.resume_checkpoint_path: # Explicit path provided
            checkpoint_path_to_load = pathlib.Path(args.resume_checkpoint_path)
        else: # Default to model.pt in exp_dir
            checkpoint_path_to_load = args.exp_dir / 'model.pt'

        if not checkpoint_path_to_load or not checkpoint_path_to_load.exists():
            logging.warning(f"Resume specified but checkpoint file not found: {checkpoint_path_to_load}. Starting from scratch.")
            args.resume = False # Disable resume if file not found
        else:
             logging.info(f"Attempting to resume from checkpoint: {checkpoint_path_to_load}")


    if args.resume and checkpoint_path_to_load:
        try:
            checkpoint, model, optimizer, args_loaded = load_model(checkpoint_path_to_load) # args_loaded contains args from checkpoint
            best_dev_loss = checkpoint['best_dev_loss']
            start_epoch = checkpoint['epoch'] + 1
            logging.info(f"Resumed from epoch {start_epoch}, best dev loss: {best_dev_loss:.4g}")
            # Optionally update current args with loaded args if needed, be careful about conflicts
            # args = args_loaded
            del checkpoint
        except Exception as e:
            logging.error(f"Failed to load checkpoint {checkpoint_path_to_load}: {e}. Starting from scratch.")
            args.resume = False # Disable resume on load failure
            model = build_model(args) # Use current args
            optimizer = build_optim(args, model.parameters())
            best_dev_loss = 1e9
            start_epoch = 0
    else:
        model = build_model(args)
        optimizer = build_optim(args, model.parameters())
        best_dev_loss = 1e9
        start_epoch = 0
        logging.info("Starting training from scratch.")

    # Apply DataParallel if specified in current config *after* potential loading
    if args.data_parallel and not isinstance(model, torch.nn.DataParallel):
        logging.info("Applying DataParallel wrapper.")
        model = torch.nn.DataParallel(model).to(args.device)
    elif not args.data_parallel and isinstance(model, torch.nn.DataParallel):
         # This case happens if loaded model was DP but current config is not
         logging.info("Removing DataParallel wrapper as per current config.")
         model = model.module # Unwrap

    # Log model architecture (optional, can be large)
    # logging.info(model)
    if wandb_active:
        wandb.watch(model, log='gradients', log_freq=max(100, args.report_interval * 5)) # Watch model


    train_loader, dev_loader, display_loader1 = create_data_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    logging.info(f"Starting training loop from epoch {start_epoch}...")
    for epoch in range(start_epoch, args.num_epochs):

        train_loss, train_time, current_lr = train_epoch(args, epoch, model, train_loader, optimizer)
        dev_loss, dev_time = evaluate(args, epoch, model, dev_loader)

        # Log epoch-level summaries to Wandb
        if wandb_active:
            wandb.log({
                'train/epoch_loss': train_loss,
                'val/epoch_loss': dev_loss, # Already logged in evaluate, but can log again here for summary
                'epoch': epoch,
                'lr': current_lr # Log LR used in this epoch
            }, step=epoch) # Use epoch as the step

        # Visualize
        display_dataset_type = args.dataset_type.split(',')[0] if args.dataset_type else 'display'
        visualize(args, epoch, model, display_loader1, display_dataset_type)

        is_new_best = dev_loss < best_dev_loss
        if is_new_best:
            best_dev_loss = dev_loss
            logging.info(f"*** New Best Dev Loss: {best_dev_loss:.4g} at Epoch {epoch} ***")
            # Log best metrics to wandb summary
            if wandb_active:
                 wandb.summary['best_val_loss'] = best_dev_loss
                 wandb.summary['best_epoch'] = epoch

        # Save model checkpoint (passing config_dict for saving)
        save_model(args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best, config_dict)

        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'DevLoss= {dev_loss:.4g} BestDevLoss= {best_dev_loss:.4g} '
            f'LR = {current_lr:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s '
            f'{"*** New Best ***" if is_new_best else ""}'
        )

        # Step the scheduler *after* evaluation and saving
        scheduler.step()


    logging.info("Training finished.")
    if wandb_active:
        wandb.finish()
        logging.info("Wandb run finished.")


def create_arg_parser():
    # This function remains useful if you want to override CONFIG defaults via command line
    # However, the current script runs directly from the CONFIG dictionary.
    parser = argparse.ArgumentParser(description='Train setup for OCUCFormer Single-Coil with WandB')
    # Add arguments corresponding to keys in CONFIG if command-line overrides are desired
    # Example:
    # parser.add_argument('--lr', type=float, help='Override learning rate from CONFIG')
    # ... other arguments ...
    return parser


if __name__ == '__main__':
    # --- Seed everything ---
    random.seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])
    if CONFIG['device'].startswith('cuda'):
        torch.cuda.manual_seed(CONFIG['seed'])
        torch.cuda.manual_seed_all(CONFIG['seed']) # For multi-GPU
        # Optional: For reproducibility, uncomment these, may slow down training
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    print("--- Starting Training with Configuration ---")
    # Print config nicely
    for key, value in CONFIG.items():
        print(f"{key}: {value}")
    print("-----------------------------------------")

    main(CONFIG) # Pass the configuration dictionary directly