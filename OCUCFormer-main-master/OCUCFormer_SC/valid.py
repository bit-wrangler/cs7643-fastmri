import pathlib
import sys
from collections import defaultdict
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import SliceDataDev # Make sure this uses path correctly
from models import OCUCFormer # Assuming OCUCFormer is the correct model for SC too
import h5py
from tqdm import tqdm
import os # <-- Added import
from dotenv import load_dotenv # <-- Added import


def save_reconstructions(reconstructions, out_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.
    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
    # Ensure out_dir is a Path object and exists
    if not isinstance(out_dir, pathlib.Path):
        out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname, recons_array in reconstructions.items():
        # Input should be a dict: fname -> np.array(slices, H, W)
        if not isinstance(recons_array, np.ndarray):
             raise TypeError(f"Expected numpy array for reconstruction '{fname}', got {type(recons_array)}")

        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons_array)


def create_data_loaders(args):
    # This function expects args.data_path and args.usmask_path to be Path objects
    data = SliceDataDev(args.data_path, args.acceleration_factor, args.dataset_type, args.mask_type, args.usmask_path)
    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=1, # Consider increasing based on system
        pin_memory=True,
        shuffle=False # No need to shuffle validation data
    )
    return data_loader


def load_model(checkpoint_file):
    # Ensure checkpoint_file is a Path object
    if not isinstance(checkpoint_file, pathlib.Path):
        checkpoint_file = pathlib.Path(checkpoint_file)

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    checkpoint = torch.load(checkpoint_file, map_location='cpu') # Load to CPU first

    # Recreate Namespace object from saved dictionary
    if 'args' not in checkpoint or not isinstance(checkpoint['args'], dict):
        raise ValueError("Checkpoint does not contain valid 'args' dictionary.")
    args_dict = checkpoint['args']
    args = argparse.Namespace(**args_dict)

    # Update args device to current device if necessary
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Build model using recreated args Namespace
    model = OCUCFormer(args, timesteps=args.timesteps).to(args.device)

    # Handle DataParallel state dict loading if necessary
    state_dict = checkpoint['model']
    data_parallel_saved = any(k.startswith('module.') for k in state_dict.keys())
    data_parallel_current = isinstance(model, torch.nn.DataParallel) # Check if model is already wrapped

    if data_parallel_saved and not data_parallel_current:
        # Model was saved with DataParallel, loading without it
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        model.to(args.device) # Move model to device
    elif data_parallel_saved and data_parallel_current:
         # Both saved and current model use DataParallel (unlikely if build_model doesn't wrap)
         model.load_state_dict(state_dict)
    elif not data_parallel_saved and data_parallel_current:
        # Saved without DP, loading with DP (wrap it)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict) # Load state dict after wrapping
        model.to(args.device)
    else:
        # Neither uses DataParallel
        model.load_state_dict(state_dict)
        model.to(args.device) # Move model to device

    return model, args # Return loaded args as well


def run_unet(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(list)
    with torch.no_grad():
        for (iter,data) in enumerate(tqdm(data_loader, desc="Validation")):

            # Ensure dataset returns 6 items: input, kspace, target, mask, fnames, slices
            if len(data) != 6:
                 raise ValueError(f"Validation DataLoader expected 6 items, but got {len(data)}. Check SliceDataDev.")

            input_img, input_kspace, target, mask, fnames, slices = data

            input_img = input_img.unsqueeze(1).to(args.device).float()
            input_kspace = input_kspace.to(args.device).float() # K-space likely complex float
            # target = target.unsqueeze(1).to(args.device).float() # Target not needed for inference
            mask = mask.to(args.device).float()

            recons = model(input_img, input_kspace, mask) # Model call

            recons = recons.to('cpu').squeeze(1) # Move to CPU, remove channel dim

            current_batch_size = recons.shape[0]
            for i in range(current_batch_size):
                # Store slice index and reconstruction numpy array
                # Ensure fnames and slices are indexed correctly for the batch
                fname = fnames[i]
                slice_idx = slices[i].item() # Get Python number from tensor
                recon_np = recons[i].numpy()
                reconstructions[fname].append((slice_idx, recon_np))

    # Process reconstructions: sort by slice index and stack
    final_reconstructions = {}
    for fname, slice_preds in reconstructions.items():
        # Sort based on slice index (the first element in the tuple)
        sorted_preds = sorted(slice_preds, key=lambda x: x[0])
        # Stack the numpy arrays (the second element in the tuple)
        final_reconstructions[fname] = np.stack([pred for _, pred in sorted_preds])

    return final_reconstructions


def main(args):
    # --- Load .env file ---
    load_dotenv()

    # --- Get validation path from .env, fallback to args ---
    val_path_env = os.getenv('SINGLECOIL_VAL_PATH')
    usmask_path_env = os.getenv('USMASK_PATH') # Check if usmask_path is in .env

    # Handle data path
    if val_path_env:
        args.data_path = pathlib.Path(val_path_env)
        print(f"Using validation data path from .env: {args.data_path}")
    else:
        if args.data_path:
            args.data_path = pathlib.Path(args.data_path)
            print(f"Using validation data path from command line: {args.data_path}")
        else:
            raise ValueError("Validation data path not specified either in .env (SINGLECOIL_VAL_PATH) or via --data-path argument.")

    # Handle usmask path
    if usmask_path_env:
         args.usmask_path = pathlib.Path(usmask_path_env)
         print(f"Using usmask path from .env: {args.usmask_path}")
    else:
        if args.usmask_path:
            args.usmask_path = pathlib.Path(args.usmask_path)
            print(f"Using usmask path from command line: {args.usmask_path}")
        else:
             # Mask path is critical, raise error if missing
             raise ValueError("Undersampling mask path not specified either in .env (USMASK_PATH) or via --usmask_path argument.")


    # --- Check if paths exist ---
    if not args.data_path.exists():
        raise FileNotFoundError(f"Validation data path does not exist: {args.data_path}")
    if not args.usmask_path.exists():
        raise FileNotFoundError(f"Undersampling mask path does not exist: {args.usmask_path}")
    if not args.checkpoint.exists():
         raise FileNotFoundError(f"Checkpoint file does not exist: {args.checkpoint}")

    # --- Ensure out_dir is Path ---
    args.out_dir = pathlib.Path(args.out_dir)


    # --- Set device from command line ---
    # For validation, typically the command-line arg takes precedence
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {args.device}")

    # --- Load model and create loader ---
    model, loaded_args = load_model(args.checkpoint) # Get loaded args
    # Update current args with device from loaded args only if necessary
    # Generally, keep the device specified for the validation run
    # args.device = loaded_args.device

    data_loader = create_data_loaders(args) # Pass current args

    # --- Run inference and save ---
    reconstructions = run_unet(args, model, data_loader)
    save_reconstructions(reconstructions, args.out_dir)
    print(f"Reconstructions saved to {args.out_dir}")


def create_arg_parser():

    parser = argparse.ArgumentParser(description="Validation script for OCUCFormer Single-Coil")
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--out-dir', type=pathlib.Path, required=True,
                        help='Path to save the reconstructions to')
    parser.add_argument('--batch-size', default=16, type=int, help='Mini-batch size for validation')
    parser.add_argument('--device', type=str, default='cuda:0', help='Which device to run on (e.g., cuda:0, cpu)')
    parser.add_argument('--data-path',type=str, default=None, # Added default
                        help='Path to validation dataset (reads from .env SINGLECOIL_VAL_PATH if set)')
    parser.add_argument('--acceleration_factor',type=str, required=True, help='Acceleration factor (e.g., 4x)')
    parser.add_argument('--dataset_type',type=str, required=True, help='Dataset type (e.g., mrbrain_t1)')
    parser.add_argument('--usmask_path',type=str, default=None, # Added default
                        help='Path to undersampling masks directory (reads from .env USMASK_PATH if set)')
    parser.add_argument('--mask_type',type=str, required=True, help='Mask type (e.g., cartesian)')

    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args() # Parse arguments from command line
    main(args)