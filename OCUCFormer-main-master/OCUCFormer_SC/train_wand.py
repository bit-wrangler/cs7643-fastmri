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
import wandb # Use wandb for logging
import torchvision.transforms.functional as TF # For center_crop
from torch.nn import functional as F
from torch.utils.data import DataLoader

# Ensure these imports point to your actual dataset and model files
# Make sure these files exist and are correctly implemented
try:
    from dataset import SliceData, SliceDisplayDataDev
    from models import OCUCFormer
except ImportError as e:
    print(f"Error importing dataset or models: {e}")
    print("Please ensure dataset.py and models.py (containing OCUCFormer) are in the correct path.")
    sys.exit(1)

import torchvision
from torch import nn
from torch.autograd import Variable
from torch import optim
from tqdm import tqdm

# --- Configuration ---
# Make sure to set WANDB_PROJECT and WANDB_ENTITY environment variables
# or change the project/entity values here.
CONFIG = {
    # == WandB Project Configuration ==
    'project': os.getenv('WANDB_PROJECT', 'cs7643-fastmri-ocucformer'), # Wandb project name
    'entity': os.getenv('WANDB_ENTITY'), # Wandb entity (username or team), None if not set
    'tags': ['ocucformer', 'singlecoil', 'baseline'], # Example tags
    'notes': 'Training OCUCFormer baseline model on single-coil knee data with wandb logging.', # Example notes

    # == Experiment Directory & Checkpointing ==
    'checkpoint': 'checkpoints_sc', # Base directory for checkpoints
    'resume': False, # Set to True to resume training
    'resume_checkpoint_path': None, # Specific checkpoint file to resume from (if None, tries 'model.pt' in exp_dir)
    'exp_dir': 'checkpoints_sc', # Directory for experiment output (logs, checkpoints)

    # == Data & Mask Parameters ==
    'dataset_type': 'knee_singlecoil',
    'mask_type': 'cartesian',
    'acceleration_factor': "4x", # Example, ensure this matches your data/masks

    # == Training Hyperparameters ==
    'batch_size': 1, # Adjusted based on potential memory issues
    'num_epochs': 150,
    'lr': 0.001,
    'lr_step_size': 40,
    'lr_gamma': 0.1,
    'weight_decay': 0.0,
    'seed': 42,
    'report_interval': 100, # Log loss within epoch interval

    # == Model Hyperparameters ==
    # Ensure these match the intended OCUCFormer configuration
    'num_pools': 4, # Default U-Net pools (adjust based on model def)
    'drop_prob': 0.0, # Dropout probability
    'num_chans': 32, # Base number of channels (adjust based on model def)
    'timesteps': 5, # Timesteps for OCUCFormer

    # == Other Settings ==
    'data_parallel': False, # Use True for multi-GPU DataParallel
    'device': 'cuda', # 'cuda' or 'cpu'
}

# Global flag to track wandb status
wandb_active = False

def create_datasets(args):
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
        num_workers=4, # Adjust based on system capabilities
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
    )
    display_loader1 = DataLoader(
        dataset=display1_subset,
        batch_size=min(args.batch_size, 16),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, dev_loader, display_loader1

def train_epoch(args, epoch, model, data_loader, optimizer, start_epoch_for_logging):
    global wandb_active # Use global flag

    model.train()
    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    loop = tqdm(data_loader, desc=f"Epoch {epoch} Train")
    current_lr = optimizer.param_groups[0]['lr']

    for iter, data in enumerate(loop):
        try:
            input_img, input_kspace, target_img, mask = data
        except ValueError as e:
            logging.error(f"Error unpacking data batch: {e}. Check SliceData return format.")
            continue # Skip batch

        input_img = input_img.unsqueeze(1).to(args.device).float()
        input_kspace = input_kspace.to(args.device).float()
        target_img = target_img.unsqueeze(1).to(args.device).float() # Shape: [B, 1, 320, 320] (Expected)
        mask = mask.to(args.device).float()

        # --- Model Forward Pass ---
        try:
            output_img = model(input_img, input_kspace, mask) # Expected shape may differ
        except Exception as model_error:
            logging.error(f"Error during model forward pass: {model_error}")
            continue # Skip batch

        # --- FIX: Center Crop Output to Match Target ---
        target_h, target_w = target_img.shape[2], target_img.shape[3]
        if output_img.shape[2] != target_h or output_img.shape[3] != target_w:
            try:
                output_img_cropped = TF.center_crop(output_img, (target_h, target_w))
                if iter == 0 and epoch == start_epoch_for_logging: # Log warning only once per run start
                   logging.warning(f"Model output shape {output_img.shape} differs from target shape {target_img.shape}. Cropping output for loss calculation.")
            except Exception as crop_error:
                logging.error(f"Error cropping output tensor: {crop_error}. Output shape: {output_img.shape}, Target shape: {target_img.shape}")
                continue
        else:
            output_img_cropped = output_img

        if output_img_cropped.shape != target_img.shape:
             logging.error(f"Shape mismatch AFTER cropping! Output: {output_img_cropped.shape}, Target: {target_img.shape}. Skipping loss.")
             continue

        # --- Loss Calculation & Optimization ---
        try:
            loss = F.l1_loss(output_img_cropped, target_img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        except Exception as loss_error:
            logging.error(f"Error during loss calculation or optimization: {loss_error}")
            continue # Skip batch

        current_loss = loss.item()
        avg_loss = 0.99 * avg_loss + 0.01 * current_loss if iter > 0 else current_loss

        # --- Wandb Logging within epoch ---
        if iter % args.report_interval == 0:
            if wandb_active: # Check if wandb initialized
                wandb.log({
                    'train/batch_loss': current_loss,
                    'train/avg_loss': avg_loss,
                    'train/epoch': epoch,
                    'train/batch_step': global_step + iter,
                    'train/lr': current_lr
                })
            # Standard logging continues regardless
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {current_loss:.4g} Avg Loss = {avg_loss:.4g} '
                f'LR = {current_lr:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
        loop.set_postfix({'AvgLoss': avg_loss, 'LR': current_lr})

    return avg_loss, time.perf_counter() - start_epoch, current_lr

def evaluate(args, epoch, model, data_loader):
    global wandb_active # Use global flag

    model.eval()
    losses = []
    # TODO: Initialize lists for PSNR and SSIM if calculating
    # psnr_scores = []
    # ssim_scores = []
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader, desc=f"Epoch {epoch} Eval")):
            try:
                input_img, input_kspace, target_img, mask = data
            except ValueError as e:
                logging.error(f"Error unpacking data batch in eval: {e}. Check SliceData return format.")
                continue # Skip batch

            input_img = input_img.unsqueeze(1).to(args.device).float()
            input_kspace = input_kspace.to(args.device).float()
            target_img = target_img.unsqueeze(1).to(args.device).float()
            mask = mask.to(args.device).float()

            try:
                _,_,H_img,W_img = input_img.shape
                _,H_ksp,W_ksp,_ = input_kspace.shape

                if W_img != W_ksp:
                    pad_w = W_ksp - W_img        # = 48 if mask width 368
                    pad_left  = pad_w // 2
                    pad_right = pad_w - pad_left
                    # F.pad expects (W_left, W_right, H_top, H_bottom)
                    input_img = F.pad(input_img, (pad_left, pad_right, 0, 0), "constant", 0)

                output_img = model(input_img, input_kspace, mask)
                output_img = TF.center_crop(output_img, (320, 320))
            except Exception as model_error:
                logging.error(f"Error during model forward pass in eval: {model_error}")
                import traceback
                logging.error(traceback.format_exc())
                continue # Skip batch

            # --- FIX: Center Crop Output to Match Target ---
            target_h, target_w = target_img.shape[2], target_img.shape[3]
            if output_img.shape[2] != target_h or output_img.shape[3] != target_w:
                try:
                    output_img_cropped = TF.center_crop(output_img, (target_h, target_w))
                except Exception as crop_error:
                    logging.error(f"Error cropping output tensor in eval: {crop_error}.")
                    continue
            else:
                output_img_cropped = output_img

            if output_img_cropped.shape != target_img.shape:
                 logging.error(f"Shape mismatch AFTER cropping in eval! Output: {output_img_cropped.shape}, Target: {target_img.shape}.")
                 continue

            # --- Calculate Metrics ---
            try:
                loss = F.l1_loss(output_img_cropped, target_img)
                losses.append(loss.item())
                # TODO: Calculate PSNR and SSIM between output_img_cropped and target_img
                # psnr_val = calculate_psnr(output_img_cropped, target_img)
                # ssim_val = calculate_ssim(output_img_cropped, target_img)
                # psnr_scores.append(psnr_val)
                # ssim_scores.append(ssim_val)
            except Exception as metric_error:
                logging.error(f"Error calculating metrics in eval: {metric_error}")
                continue # Skip batch


        avg_dev_loss = np.mean(losses) if losses else 0
        # avg_psnr = np.mean(psnr_scores) if psnr_scores else 0
        # avg_ssim = np.mean(ssim_scores) if ssim_scores else 0

        # --- Wandb Logging for epoch evaluation ---
        if wandb_active: # Check if wandb initialized
            log_data = {'val/epoch_loss': avg_dev_loss, 'val/epoch': epoch}
            # if psnr_scores: log_data['val/psnr'] = avg_psnr
            # if ssim_scores: log_data['val/ssim'] = avg_ssim
            wandb.log(log_data)

    return avg_dev_loss, time.perf_counter() - start # Return other metrics if calculated

def visualize(args, epoch, model, data_loader, datasettype_string):
    global wandb_active # Use global flag

    def log_wandb_image(image_tensor, caption):
        if not wandb_active: # Check if wandb initialized
             return

        try:
            # Ensure tensor is detached, on CPU, and normalized for wandb
            image_tensor = image_tensor.detach().cpu()
            # Normalize each image in the batch individually
            processed_images = []
            for i in range(image_tensor.shape[0]):
                 img = image_tensor[i]
                 img_min = img.min()
                 img_max = img.max()
                 # Handle potential division by zero if min == max
                 denominator = img_max - img_min + 1e-6
                 img_normalized = (img - img_min) / denominator
                 processed_images.append(img_normalized)
            image_tensor_normalized = torch.stack(processed_images)


            grid = torchvision.utils.make_grid(image_tensor_normalized, nrow=4, pad_value=1)
            # Log image to wandb - use step=epoch for timeline consistency
            wandb.log({f"val/images/{caption}": wandb.Image(grid, caption=f"{caption} - Epoch {epoch}")}, step=epoch)
        except Exception as e:
            logging.error(f"Failed to log image '{caption}' to wandb: {e}")


    model.eval()
    with torch.no_grad():
        # Only visualize one batch
        try:
            data_iter = iter(data_loader)
            data = next(data_iter)
        except StopIteration:
            logging.warning("Visualization data loader is empty.")
            return
        except Exception as e:
            logging.error(f"Error getting data from visualization loader: {e}")
            return

        try:
            input_img, input_kspace, target_img, mask = data
        except ValueError as e:
            logging.error(f"Error unpacking data batch in visualize: {e}. Check SliceDisplayDataDev return format.")
            return

        input_img = input_img.unsqueeze(1).to(args.device).float()
        input_kspace = input_kspace.to(args.device).float()
        target_img = target_img.unsqueeze(1).to(args.device).float()
        mask = mask.to(args.device).float()

        try:
            output_img = model(input_img, input_kspace, mask)
        except Exception as model_error:
            logging.error(f"Error during model forward pass in visualize: {model_error}")
            return

        # --- FIX: Center Crop Output to Match Target ---
        target_h, target_w = target_img.shape[2], target_img.shape[3]
        if output_img.shape[2] != target_h or output_img.shape[3] != target_w:
            try:
                output_img_cropped = TF.center_crop(output_img, (target_h, target_w))
            except Exception as crop_error:
                 logging.error(f"Error cropping output tensor in visualize: {crop_error}.")
                 output_img_cropped = output_img # Fallback to uncropped for visualization if crop fails
        else:
            output_img_cropped = output_img

        # Log images using the potentially cropped output for reconstruction/error
        log_wandb_image(input_img, f'Input_{datasettype_string}')
        log_wandb_image(target_img, f'Target_{datasettype_string}')
        log_wandb_image(output_img_cropped, f'Reconstruction_{datasettype_string}')
        if output_img_cropped.shape == target_img.shape:
            log_wandb_image(torch.abs(target_img - output_img_cropped), f'Error_{datasettype_string}')
        else:
            logging.warning("Skipping error map visualization due to shape mismatch after fallback.")


def save_model(exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best, args_dict):
    global wandb_active # Use global flag

    if not isinstance(exp_dir, pathlib.Path):
        exp_dir = pathlib.Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True) # Ensure dir exists

    save_path = exp_dir / 'model.pt'
    best_save_path = exp_dir / 'best_model.pt'

    try:
        # Prepare model state dict (handle DataParallel)
        if isinstance(model, torch.nn.DataParallel):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        torch.save(
            {
                'epoch': epoch,
                'args': args_dict, # Save config dictionary
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
            # --- Wandb: Save best model artifact ---
            if wandb_active: # Check if wandb initialized
                try:
                    # Use wandb.run.name which is unique for the run
                    artifact_name = f'{wandb.run.name}-best-model' if wandb.run else f'best-model-epoch-{epoch}'
                    artifact = wandb.Artifact(artifact_name, type='model', metadata={'epoch': epoch, 'best_dev_loss': best_dev_loss})
                    artifact.add_file(str(best_save_path))
                    wandb.log_artifact(artifact, aliases=['best', f'epoch-{epoch}'])
                    logging.info(f"Logged best model artifact to wandb (Epoch {epoch})")
                except Exception as e:
                    logging.error(f"Failed to log best model artifact to wandb: {e}")
    except Exception as e:
        logging.error(f"Failed to save checkpoint at epoch {epoch}: {e}")


def build_model(args_ns):
    # Pass args Namespace to the model constructor
    model = OCUCFormer(args_ns, timesteps=args_ns.timesteps).to(args_ns.device)
    return model

def load_model(checkpoint_file):
    if not isinstance(checkpoint_file, pathlib.Path):
        checkpoint_file = pathlib.Path(checkpoint_file)

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    logging.info(f"Loading checkpoint: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location='cpu') # Load to CPU first

    # Recreate Namespace object from saved dictionary
    # Use current CONFIG as default if 'args' isn't in the checkpoint
    args_dict = checkpoint.get('args', CONFIG)
    args_ns = argparse.Namespace(**args_dict)

    # CRITICAL: Override device from loaded args with the CURRENT config's device setting
    args_ns.device = torch.device(CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    logging.info(f"Checkpoint loaded. Using device from current config: {args_ns.device}")

    model = build_model(args_ns)

    # Handle DataParallel state dict loading carefully
    state_dict = checkpoint['model']
    data_parallel_saved = any(k.startswith('module.') for k in state_dict.keys())

    if data_parallel_saved:
        logging.info("Checkpoint saved with DataParallel ('module.' prefix found).")
        # Need to remove 'module.' prefix if current model is *not* DP
        if not CONFIG['data_parallel']:
            logging.info("Current config doesn't use DataParallel. Removing 'module.' prefix.")
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
        else:
            # Loading DP model into a DP model (should work directly)
            logging.info("Current config uses DataParallel. Loading state dict directly.")
            # The model will be wrapped later in main() if needed
            model.load_state_dict(state_dict)
    else:
        logging.info("Checkpoint saved without DataParallel ('module.' prefix not found).")
        # Loading non-DP model. State dict can be loaded directly.
        # If current config *requires* DP, wrapping happens later in main().
        model.load_state_dict(state_dict)

    # Move model to device *before* potential DP wrapping in main()
    model.to(args_ns.device)

    optimizer = build_optim(args_ns, model.parameters())
    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info("Loaded optimizer state from checkpoint.")
    except Exception as e:
        logging.warning(f"Could not load optimizer state_dict: {e}. Initializing new optimizer.")

    return checkpoint, model, optimizer, args_ns # Return args Namespace from CHECKPOINT

def build_optim(args_ns, params):
    optimizer = torch.optim.Adam(params, args_ns.lr, weight_decay=args_ns.weight_decay)
    return optimizer

def main(config_dict):
    global wandb_active # Modify global flag

    # Convert config dict to Namespace for convenience
    args = argparse.Namespace(**config_dict)

    # --- Initialize Wandb ---
    try:
        # Attempt to initialize wandb
        run = wandb.init(
            project=args.project,
            entity=args.entity, # Can be None
            config=config_dict,
            tags=args.tags,
            notes=args.notes,
            resume="allow", # Allows resuming if WANDB_RUN_ID env var is set
            # name= "my-run-name", # Optional: Set a custom run name
            # id= "run-id-to-resume", # Optional: Force resuming a specific run ID
        )
        if run:
            logging.info(f"Wandb initialized successfully: project='{args.project}', entity='{args.entity or 'default'}', run_id='{wandb.run.id}'")
            wandb_active = True
            log_dir = pathlib.Path(wandb.run.dir) # Use wandb dir for logs
        else:
             # This case might happen if wandb.init() returns None unexpectedly
             logging.error("wandb.init() returned None. Proceeding without wandb logging.")
             wandb_active = False
             log_dir = pathlib.Path(args.exp_dir)
    except Exception as e:
        logging.error(f"Failed to initialize Wandb: {e}. Check API key, network, project/entity names. Proceeding without wandb logging.")
        wandb_active = False
        log_dir = pathlib.Path(args.exp_dir)


    # --- Setup Logging ---
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / 'train_ocucformer_sc.log'
    # Configure root logger
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # File handler
    file_handler = logging.FileHandler(log_file_path, mode='a' if args.resume and log_file_path.exists() else 'w')
    file_handler.setFormatter(log_formatter)
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    # Add handlers to the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Remove existing handlers if any to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.info(f"Logging to: {log_file_path}")


    # --- Load .env and Paths ---
    load_dotenv()
    train_path_env = os.getenv('SINGLECOIL_TRAIN_PATH')
    val_path_env = os.getenv('SINGLECOIL_VAL_PATH')
    usmask_path_env = os.getenv('USMASK_PATH')

    args.train_path = pathlib.Path(train_path_env) if train_path_env else getattr(args, 'train_path', None)
    args.validation_path = pathlib.Path(val_path_env) if val_path_env else getattr(args, 'validation_path', None)
    args.usmask_path = pathlib.Path(usmask_path_env) if usmask_path_env else getattr(args, 'usmask_path', None)

    # Validate paths
    path_errors = []
    if not args.train_path: path_errors.append("Train path missing (.env: SINGLECOIL_TRAIN_PATH)")
    elif not pathlib.Path(args.train_path).exists(): path_errors.append(f"Train path does not exist: {args.train_path}")
    if not args.validation_path: path_errors.append("Validation path missing (.env: SINGLECOIL_VAL_PATH)")
    elif not pathlib.Path(args.validation_path).exists(): path_errors.append(f"Validation path does not exist: {args.validation_path}")
    if not args.usmask_path: path_errors.append("Mask path missing (.env: USMASK_PATH)")
    elif not pathlib.Path(args.usmask_path).exists(): path_errors.append(f"Mask path does not exist: {args.usmask_path}")

    if path_errors:
        for err in path_errors: logging.error(err)
        raise ValueError("One or more required paths are missing or invalid.")

    # Ensure paths are Path objects
    args.train_path = pathlib.Path(args.train_path)
    args.validation_path = pathlib.Path(args.validation_path)
    args.usmask_path = pathlib.Path(args.usmask_path)

    logging.info(f"Using train path: {args.train_path}")
    logging.info(f"Using validation path: {args.validation_path}")
    logging.info(f"Using usmask path: {args.usmask_path}")


    # --- Device Setup ---
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {args.device}")
    logging.info(f"Run Config: {config_dict}")


    # --- Model & Optimizer Initialization / Resuming ---
    args.exp_dir = pathlib.Path(args.exp_dir) # Ensure exp_dir is Path
    checkpoint_path_to_load = None
    if args.resume:
        if args.resume_checkpoint_path:
            checkpoint_path_to_load = pathlib.Path(args.resume_checkpoint_path)
        else:
            checkpoint_path_to_load = args.exp_dir / 'model.pt' # Default resume target

        if not checkpoint_path_to_load or not checkpoint_path_to_load.exists():
            logging.warning(f"Resume specified but checkpoint file not found: {checkpoint_path_to_load}. Starting from scratch.")
            args.resume = False # Disable resume if file not found
        else:
             logging.info(f"Attempting to resume from checkpoint: {checkpoint_path_to_load}")

    start_epoch = 0
    best_dev_loss = 1e9
    model = None
    optimizer = None

    if args.resume and checkpoint_path_to_load:
        try:
            checkpoint, model, optimizer, args_loaded = load_model(checkpoint_path_to_load)
            # CRITICAL: Use the epoch and best_dev_loss from checkpoint
            start_epoch = checkpoint.get('epoch', -1) + 1 # Get epoch, default to 0 if key missing
            best_dev_loss = checkpoint.get('best_dev_loss', 1e9) # Get loss, default high if key missing
            logging.info(f"Resumed from epoch {start_epoch}, best dev loss: {best_dev_loss:.4g}")
            # Optionally, update current config `args` with `args_loaded` if needed
            # Be cautious about overriding current run settings (e.g., device)
            del checkpoint # Free memory
        except Exception as e:
            logging.error(f"Failed to load checkpoint {checkpoint_path_to_load}: {e}. Starting from scratch.")
            args.resume = False # Ensure resume is off if loading failed
            model = build_model(args) # Build new model with current args
            optimizer = build_optim(args, model.parameters()) # Build new optimizer
    else:
        logging.info("Starting training from scratch.")
        model = build_model(args)
        optimizer = build_optim(args, model.parameters())

    # Apply DataParallel based on CURRENT config, AFTER loading/building
    if args.data_parallel:
        if isinstance(model, torch.nn.DataParallel):
            logging.info("Model already wrapped in DataParallel.")
        else:
            logging.info("Applying DataParallel wrapper.")
            model = torch.nn.DataParallel(model).to(args.device)
    elif isinstance(model, torch.nn.DataParallel):
         # Model loaded was DP, but current config is not DP -> unwrap
         logging.info("Removing DataParallel wrapper loaded from checkpoint as per current config.")
         model = model.module # Unwrap

    # Ensure final model is on the correct device
    model.to(args.device)


    # --- Wandb Watch Model ---
    if wandb_active:
        try:
            wandb.watch(model, log='gradients', log_freq=max(100, args.report_interval * 5))
        except Exception as e:
            logging.error(f"Failed to initiate wandb.watch: {e}")


    # --- Data Loaders ---
    try:
        train_loader, dev_loader, display_loader1 = create_data_loaders(args)
        logging.info("Data loaders created successfully.")
    except Exception as e:
        logging.error(f"Failed to create data loaders: {e}")
        if wandb_active: wandb.finish(exit_code=1) # Ensure wandb run finishes on critical error
        sys.exit(1) # Exit if data cannot be loaded

    # --- Scheduler ---
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)
    # Resume scheduler state if resuming (optional, depends if scheduler state was saved)
    # if args.resume and 'scheduler' in checkpoint:
    #     scheduler.load_state_dict(checkpoint['scheduler'])

    # --- Training Loop ---
    logging.info(f"Starting training loop from epoch {start_epoch}...")
    for epoch in range(start_epoch, args.num_epochs):
        # Pass start_epoch for conditional logging inside train_epoch
        train_loss, train_time, current_lr = train_epoch(args, epoch, model, train_loader, optimizer, start_epoch)
        dev_loss, dev_time = evaluate(args, epoch, model, dev_loader)

        # Log epoch-level summaries to Wandb
        if wandb_active:
            try:
                log_data = {
                    'train/epoch_loss': train_loss,
                    'val/epoch_loss': dev_loss,
                    'epoch': epoch,
                    'lr': current_lr
                    # Add avg PSNR/SSIM here if calculated in evaluate()
                    # 'val/avg_psnr': avg_psnr,
                    # 'val/avg_ssim': avg_ssim,
                }
                wandb.log(log_data, step=epoch) # Use epoch as the step
            except Exception as e:
                 logging.error(f"Failed to log epoch summary to wandb: {e}")


        # Visualize
        display_dataset_type = args.dataset_type.split(',')[0] if args.dataset_type else 'display'
        visualize(args, epoch, model, display_loader1, display_dataset_type)

        # Checkpoint Saving
        is_new_best = dev_loss < best_dev_loss
        if is_new_best:
            best_dev_loss = dev_loss
            logging.info(f"*** New Best Dev Loss: {best_dev_loss:.4g} at Epoch {epoch} ***")
            if wandb_active: # Check if wandb initialized
                 try:
                     wandb.summary['best_val_loss'] = best_dev_loss
                     wandb.summary['best_epoch'] = epoch
                 except Exception as e:
                      logging.error(f"Failed to update wandb summary: {e}")

        # Save checkpoint (passing config_dict for saving args)
        save_model(args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best, config_dict)

        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'DevLoss= {dev_loss:.4g} BestDevLoss= {best_dev_loss:.4g} '
            f'LR = {current_lr:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s '
            f'{"*** New Best ***" if is_new_best else ""}'
        )

        # Step the scheduler
        scheduler.step()

    # --- End of Training ---
    logging.info("Training finished.")
    if wandb_active:
        try:
            wandb.finish()
            logging.info("Wandb run finished.")
        except Exception as e:
             logging.error(f"Error during wandb.finish(): {e}")


# --- Main Execution ---
if __name__ == '__main__':
    # --- Seed everything ---
    seed = CONFIG.get('seed', 42) # Use seed from config, default 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if CONFIG.get('device', 'cpu').startswith('cuda'):
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU
        # Optional: For reproducibility, uncomment these, may slow down training
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # print("CUDA seeds set.")

    print("--- Starting Training with Configuration ---")
    # Print config nicely
    for key, value in CONFIG.items():
        print(f"{key}: {value}")
    print("-----------------------------------------")

    main(CONFIG) # Pass the configuration dictionary directly