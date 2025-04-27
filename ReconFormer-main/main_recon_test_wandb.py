#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ReconFormer Evaluation with Weights & Biases Logging (Updated + Debug Prints)
"""
import warnings
warnings.simplefilter("ignore", UserWarning)

import os
import pathlib
import torch
import wandb
from torch.utils.data import DataLoader
import argparse # Added for type hinting if needed

from utils.options import args_parser
# Assuming test_recon_save is defined in models.evaluation or similar
# Make sure this function returns a dictionary of metrics {metric_name: value}
from models.evaluation import test_recon_save
# DataTransform is likely in mri_data, ensure path is correct if needed
from data.mri_data import SliceData, DataTransform
from data.subsample import create_mask_for_mask_type
from models.Recurrent_Transformer import ReconFormer
try:
    from ssim_loss import ssim as calculate_ssim_metric
except ImportError:
    print("Warning: Could not import ssim function from ssim_loss.py. SSIM metric will not be calculated.")
    calculate_ssim_metric = None



import dotenv

dotenv.load_dotenv()

def _create_dataset(data_path,data_transform, data_partition, sequence, bs, shuffle, sample_rate=None, challenge=None, display=False):
        # torch.cuda.empty_cache()

        sample_rate = sample_rate or sample_rate
        dataset = SliceData(
            root=data_path / data_partition,
            transform=data_transform,
            sample_rate=sample_rate,
            challenge=challenge,
            sequence=sequence
        )
        return DataLoader(dataset, batch_size=bs, shuffle=shuffle, pin_memory=False, num_workers=8)

def create_dataloader(root_path, phase, transform, sequence, batch_size, shuffle, sample_rate, challenge):
    """
    Build DataLoader for MRI slices.
    """
    dataset = SliceData(
        root=root_path / phase,
        transform=transform,
        sample_rate=sample_rate, # Use passed sample_rate
        challenge=challenge,     # Use passed challenge
        sequence=sequence
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False,
        num_workers=0 # Setting to 0 for simplicity/debugging memory issues
    )


def main():
        # disable HDF5 locking on network filesystems
    os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

    # parse command-line arguments
    args, parser = args_parser()
    print('--- Parsed Command Line Args ---')
    print(args)
    print('--------------------------------')

    # --- Set resolution BEFORE using it in config ---
    print("Attempting to set args.resolution...")
    if not hasattr(args, 'resolution') or args.resolution is None:
        try:
            resolution_dict = {'F': 320}
            if hasattr(args, 'test_dataset'):
                 args.resolution = resolution_dict.get(args.test_dataset, 320)
                 print(f"Inferred/defaulted resolution based on test_dataset '{args.test_dataset}': {args.resolution}")
            else:
                 args.resolution = 320
                 print(f"Warning: args.test_dataset not found. Defaulting resolution to {args.resolution}")
        except Exception as e:
            print(f"Error setting resolution: {e}. Defaulting resolution to 320.")
            args.resolution = 320
    else:
        print(f"args.resolution already existed with value: {args.resolution}")

    print('--- Args After Setting Resolution ---')
    print(args)
    print('-------------------------------------')
    # --- End Set resolution ---


    # --- W&B Initialization Setup (Updated) ---
    print("Defining TEST_CONFIG dictionary...")
    current_resolution = getattr(args, 'resolution', None)
    if current_resolution is None:
        print("CRITICAL ERROR: args.resolution is None before creating TEST_CONFIG. Exiting.")
        exit(1)

    TEST_CONFIG = {
        # Data parameters from args
        'test_dataset': args.test_dataset,
        'sequence': args.sequence,
        'challenge': args.challenge, # Keep challenge in logged config
        'resolution': current_resolution,
        'center_fractions': args.center_fractions,
        'accelerations': args.accelerations,
        'mask_type': args.mask_type,
        'sample_rate': args.sample_rate,
        # Model parameters from args / script defaults
        'model': args.model,
        # Testing parameters
        'checkpoint': args.checkpoint,
        'batch_size': args.bs,
    }
    print("TEST_CONFIG defined.")

    # Merge all command-line args into the config for complete tracking
    TEST_CONFIG.update(vars(args))

    # Determine project name
    project_name = os.getenv('WANDB_PROJECT_NAME', 'cs7643-fastmri')

    # Create a descriptive run name
    run_name = f"Test_{args.test_dataset}_{args.sequence}_Acc{args.accelerations[0]}_Frac{args.center_fractions[0]}"
    if args.checkpoint:
         chkpt_name = pathlib.Path(args.checkpoint).stem
         run_name += f"_{chkpt_name}"

    # Define tags and notes
    tags = ['evaluation', args.model, args.test_dataset]
    if hasattr(args, 'tags') and args.tags:
         tags.extend(args.tags)
    notes = f"Evaluation run for checkpoint: {args.checkpoint}"
    if hasattr(args, 'notes') and args.notes:
        notes = args.notes

    # Initialize W&B run
    print("Initializing W&B run...")
    run = wandb.init(
        project=project_name,
        config=TEST_CONFIG,
        name=run_name,
        tags=tags,
        notes=notes,
        job_type='test'
    )
    print(f"W&B Run initialized: {run.name} (ID: {run.id})")
    # --- End W&B Initialization Setup ---


    # device setup
    use_cuda = torch.cuda.is_available() and args.gpu and args.gpu[0] >= 0
    device = torch.device(f"cuda:{args.gpu[0]}" if use_cuda else "cpu")
    args.device = device
    print(f"Using device: {device}")

    # Clear cache before model/data loading (optional)
    if use_cuda:
        torch.cuda.empty_cache()

    print("Creating DataLoader...")
    test_data_path = pathlib.Path(args.F_path)
    path_dict = {'F': pathlib.Path(args.F_path)}

    # load dataset and split users
    if args.challenge == 'singlecoil':
        mask = create_mask_for_mask_type(args.mask_type, args.center_fractions,
                                         args.accelerations)
        val_data_transform = DataTransform(args.resolution, args.challenge, mask, use_seed=True)

        if args.phase == 'test':
                                                                # data_path,data_transform, data_partition, sequence, bs, shuffle, sample_rate=None, challenge=None, display=Fals
            dataset_val = _create_dataset(path_dict[args.test_dataset]/args.sequence,val_data_transform, 'val', args.sequence, args.bs, False, 1.0, args.challenge)
    else:
        exit('Error: unrecognized dataset')

    # create test DataLoader
     

    if not test_data_path.exists():
        raise FileNotFoundError(f"Data path not found: {test_data_path}")

    data_root_for_loader = test_data_path


    # instantiate model
    print("Instantiating model...")
    net = ReconFormer(in_channels=2, out_channels=2, num_ch=(96, 48, 24),num_iter=5,   #5,
        down_scales=(2,1,1.5), img_size=args.resolution, num_heads=(6,6,6), depths=(2,1,1),
        window_sizes=(8,8,8), mlp_ratio=2., resi_connection ='1conv',
        use_checkpoint=(False, False, False, False, False, False)
        # use_checkpoint=(True, True, True, True, True, True)

        ).to(args.device)
    wandb.watch(net, log="all", log_graph=True, log_freq=10) 

    # load pretrained weights
    print(f"Loading checkpoint: {args.checkpoint}")
    if not args.checkpoint or not pathlib.Path(args.checkpoint).exists():
         raise FileNotFoundError(f"Checkpoint not found or not specified: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    if all(key.startswith('module.') for key in state_dict):
        print("Removing 'module.' prefix from checkpoint keys...")
        state_dict = {k.partition('module.')[2]: v for k, v in state_dict.items()}

    try:
        net.load_state_dict(state_dict)
        print(f"Loaded state_dict from checkpoint: {args.checkpoint}")
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        print("Attempting to load with strict=False...")
        try:
            net.load_state_dict(state_dict, strict=False)
            print("Loaded state_dict with strict=False.")
        except Exception as final_e:
            print(f"Failed to load state_dict even with strict=False: {final_e}")
            if run: wandb.finish(exit_code=1)
            exit(1)


    # run evaluation
    print("Running evaluation...")
    net.eval()
    print("Running test_recon_save...")
    with torch.no_grad():
        metrics = test_recon_save(net, dataset_val, args) or {}
    print("Metrics from test_recon_save:", metrics)

    # Compute SSIM if available
    if calculate_ssim_metric is not None:
        print("Computing SSIM loss over dataset...")
        ssim_losses = []
        for batch in dataset_val:
            inp, tgt, mean, std, norm, fname, slice_idx, maxv, m, masked_k = batch
            out = net(inp.to(device), masked_k.to(device), m.to(device))
            # magnitude
            pred = torch.abs(out[...,0] + 1j*out[...,1]).unsqueeze(1)
            truth = torch.abs(tgt.to(device)[...,0] + 1j*tgt.to(device)[...,1]).unsqueeze(1)
            data_range = torch.tensor([maxv], device=device)
            loss_ssim = 1.0 - calculate_ssim_metric(pred, truth, data_range)
            ssim_losses.append(loss_ssim.item())
        avg_ssim = sum(ssim_losses)/len(ssim_losses)
        metrics['ssim_loss'] = avg_ssim
        print(f"Average SSIM loss: {avg_ssim}")

    # Log all metrics
    metric_keys = ['val_loss', 'nmse', 'ssim', 'psnr']  # Define keys for the metrics tuple
    to_log = {f"Test/{k}": v for k, v in zip(metric_keys, metrics)}
    wandb.log(to_log)
    wandb.run.summary.update(to_log)
    print("Logged to W&B:", to_log)

    wandb.finish()
    print("Evaluation completed.")



if __name__ == "__main__":
    main()