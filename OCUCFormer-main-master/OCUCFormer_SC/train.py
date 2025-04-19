import sys
import logging
import pathlib
import random
import shutil
import time
import functools
import numpy as np
import argparse
import os  # <-- Added import
from dotenv import load_dotenv # <-- Added import

import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataset import SliceData,SliceDisplayDataDev # <-- Make sure these use paths correctly
from models import OCUCFormer # Assuming OCUCFormer is the correct model for SC too
import torchvision
from torch import nn
from torch.autograd import Variable
from torch import optim
from tqdm import tqdm

def create_datasets(args):
    # This function expects args.train_path and args.validation_path to be Path objects
    acc_factors = args.acceleration_factor.split(',')
    mask_types = args.mask_type.split(',')
    dataset_types = args.dataset_type.split(',')

    # Ensure args.usmask_path is also a Path object if used here
    # args.usmask_path is handled in main() to prioritize .env
    usmask_path = args.usmask_path # Already a Path object from main()

    train_data = SliceData(args.train_path, acc_factors, dataset_types, mask_types, 'train', usmask_path)
    dev_data = SliceData(args.validation_path, acc_factors, dataset_types, mask_types, 'validation', usmask_path)

    # Use the first combination for display data - ensure args are Path objects
    display1_data = SliceDisplayDataDev(args.validation_path, dataset_types[0], mask_types[0], acc_factors[0], usmask_path)
    return dev_data, train_data, display1_data


def create_data_loaders(args):
    dev_data, train_data, display1_data = create_datasets(args)

    # Consider making the display sampling size configurable
    display_indices = range(0, len(display1_data), max(1, len(display1_data) // 16))
    display1_subset = torch.utils.data.Subset(display1_data, display_indices)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        # num_workers=4, # Consider adding num_workers
        # pin_memory=True, # Consider adding pin_memory
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        # num_workers=4,
        # pin_memory=True,
    )
    display_loader1 = DataLoader(
        dataset=display1_subset, # Use the subset
        batch_size=min(args.batch_size, 16), # Adjust batch size for display
        shuffle=False, # No need to shuffle display data
        # num_workers=4,
        # pin_memory=True,
    )
    return train_loader, dev_loader, display_loader1


def train_epoch(args, epoch, model,data_loader, optimizer, writer):

    model.train()
    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    loop = tqdm(data_loader, desc=f"Epoch {epoch} Train")

    for iter, data in enumerate(loop):
        input, input_kspace, target, mask = data # Make sure dataset returns 4 items

        input = input.unsqueeze(1).to(args.device).float()
        input_kspace = input_kspace.to(args.device).float() # K-space likely complex float
        target = target.unsqueeze(1).to(args.device).float()
        mask = mask.to(args.device).float() # Mask is usually float

        output = model(input, input_kspace, mask)

        loss = F.l1_loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        writer.add_scalar('TrainLoss', loss.item(), global_step + iter)

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
        loop.set_postfix({'Loss': avg_loss})

    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_loader, writer):

    model.eval()
    losses = []
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader, desc=f"Epoch {epoch} Eval")):
            input, input_kspace, target, mask = data

            input = input.unsqueeze(1).to(args.device).float()
            input_kspace = input_kspace.to(args.device).float()
            target = target.unsqueeze(1).to(args.device).float()
            mask = mask.to(args.device).float()

            output = model(input, input_kspace, mask)

            loss = F.l1_loss(output, target)
            losses.append(loss.item())

        writer.add_scalar('Dev_Loss', np.mean(losses), epoch)

    return np.mean(losses), time.perf_counter() - start


def visualize(args, epoch, model, data_loader, writer, datasettype_string):

    def save_image(image, tag):
        # Ensure image is on CPU and detach before manipulating
        image = image.detach().cpu()
        image -= image.min()
        image /= (image.max() + 1e-6) # Add epsilon for safety
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader, desc=f"Epoch {epoch} Visualize")):
            input, input_kspace, target, mask = data
            input = input.unsqueeze(1).to(args.device).float()
            input_kspace = input_kspace.to(args.device).float()
            target = target.unsqueeze(1).to(args.device).float()
            mask = mask.to(args.device).float()

            output = model(input, input_kspace, mask)

            save_image(input, f'Input_{datasettype_string}')
            save_image(target, f'Target_{datasettype_string}')
            save_image(output, f'Reconstruction_{datasettype_string}')
            save_image(torch.abs(target - output), f'Error_{datasettype_string}')
            break # Visualize only one batch

def save_model(args, exp_dir, epoch, model, optimizer,best_dev_loss,is_new_best):
    # Ensure exp_dir is a Path object
    if not isinstance(exp_dir, pathlib.Path):
        exp_dir = pathlib.Path(exp_dir)

    save_path = exp_dir / 'model.pt'
    best_save_path = exp_dir / 'best_model.pt'

    # Detach args Namespace before saving if it contains tensors/objects not meant for saving
    # Or better: save args as a dictionary
    args_dict = vars(args)

    torch.save(
        {
            'epoch': epoch,
            'args': args_dict, # Save args as dictionary
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': str(exp_dir) # Save as string if needed elsewhere
        },
        f=save_path
    )

    if is_new_best:
        shutil.copyfile(save_path, best_save_path)
        logging.info(f"Saved new best model to {best_save_path}")


def build_model(args):
    # Pass args Namespace to the model constructor
    model = OCUCFormer(args, timesteps=args.timesteps).to(args.device)
    return model

def load_model(checkpoint_file):
    # Ensure checkpoint_file is a Path object
    if not isinstance(checkpoint_file, pathlib.Path):
        checkpoint_file = pathlib.Path(checkpoint_file)

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    checkpoint = torch.load(checkpoint_file, map_location='cpu') # Load to CPU first

    # Recreate Namespace object from saved dictionary
    args_dict = checkpoint['args']
    args = argparse.Namespace(**args_dict)

    # Update args device to current device if necessary
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = build_model(args) # build_model uses the recreated args Namespace

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


    optimizer = build_optim(args, model.parameters())
    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
    except:
        print("Could not load optimizer state_dict, initializing optimizer.")
        logging.warning("Could not load optimizer state_dict, initializing optimizer.")


    return checkpoint, model, optimizer, args # Return updated args


def build_optim(args, params):
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    return optimizer


def main(args):
    # --- Load .env file ---
    load_dotenv() # Loads variables from .env into environment

    # --- Get paths from .env, fallback to args or raise error ---
    train_path_env = os.getenv('SINGLECOIL_TRAIN_PATH')
    val_path_env = os.getenv('SINGLECOIL_VAL_PATH')
    usmask_path_env = os.getenv('USMASK_PATH') # Check if usmask_path is in .env too

    # Prioritize .env paths if they exist
    if train_path_env:
        args.train_path = pathlib.Path(train_path_env)
        print(f"Using train path from .env: {args.train_path}")
    else:
        if args.train_path:
             args.train_path = pathlib.Path(args.train_path)
             print(f"Using train path from command line: {args.train_path}")
        else:
             raise ValueError("Train path not specified either in .env (SINGLECOIL_TRAIN_PATH) or via --train-path argument.")

    if val_path_env:
        args.validation_path = pathlib.Path(val_path_env)
        print(f"Using validation path from .env: {args.validation_path}")
    else:
        if args.validation_path:
             args.validation_path = pathlib.Path(args.validation_path)
             print(f"Using validation path from command line: {args.validation_path}")
        else:
            raise ValueError("Validation path not specified either in .env (SINGLECOIL_VAL_PATH) or via --validation-path argument.")

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
    if not args.train_path.exists():
        raise FileNotFoundError(f"Train path does not exist: {args.train_path}")
    if not args.validation_path.exists():
        raise FileNotFoundError(f"Validation path does not exist: {args.validation_path}")
    if not args.usmask_path.exists():
        raise FileNotFoundError(f"Undersampling mask path does not exist: {args.usmask_path}")

    # --- Ensure exp_dir is Path object ---
    args.exp_dir = pathlib.Path(args.exp_dir)
    args.exp_dir.mkdir(parents=True, exist_ok=True)

    # Setup SummaryWriter and Logging
    writer = SummaryWriter(log_dir=str(args.exp_dir / 'summary'))
    log_file_path = str(args.exp_dir / 'train_ocucrn_sc.log') # Changed log file name slightly
    logging.basicConfig(filename=log_file_path, filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__) # Assign logger if used later

    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {args.device}")

    if args.resume:
        print('Resuming model...')
        logging.info('Resuming model...')
        # Make sure checkpoint path is Path object
        checkpoint_path = pathlib.Path(args.checkpoint) if args.checkpoint else None
        if not checkpoint_path or not checkpoint_path.exists():
             raise FileNotFoundError(f"Resume specified but checkpoint file not found or not specified: {args.checkpoint}")

        checkpoint, model, optimizer, args = load_model(checkpoint_path) # Get updated args
        best_dev_loss= checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}, best dev loss: {best_dev_loss:.4g}")
        logging.info(f"Resuming from epoch {start_epoch}, best dev loss: {best_dev_loss:.4g}")
        del checkpoint
    else:
        model = build_model(args)
        if args.data_parallel:
             # Wrap model AFTER building if data_parallel is true
            model = torch.nn.DataParallel(model).to(args.device)
        optimizer = build_optim(args, model.parameters())
        best_dev_loss = 1e9
        start_epoch = 0
        logging.info("Starting training from scratch.")

    # Apply DataParallel if specified in args and not resuming, or if loaded model wasn't wrapped
    if args.data_parallel and not isinstance(model, torch.nn.DataParallel):
        print("Applying DataParallel wrapper.")
        model = torch.nn.DataParallel(model).to(args.device)


    logging.info(args)
    logging.info(model)

    train_loader, dev_loader, display_loader1 = create_data_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    for epoch in range(start_epoch, args.num_epochs):

        train_loss,train_time = train_epoch(args, epoch, model, train_loader,optimizer,writer)
        dev_loss,dev_time = evaluate(args, epoch, model, dev_loader, writer)

        # Determine the dataset type string for visualization dynamically if needed
        display_dataset_type = args.dataset_type.split(',')[0] if args.dataset_type else 'display'
        visualize(args, epoch, model, display_loader1, writer, display_dataset_type)

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss,dev_loss)
        save_model(args, args.exp_dir, epoch, model, optimizer,best_dev_loss,is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'DevLoss= {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s '
            f'{"*** New Best ***" if is_new_best else ""}'
        )
        # Step the scheduler after optimizer.step()
        scheduler.step()


    writer.close()


def create_arg_parser():

    parser = argparse.ArgumentParser(description='Train setup for OCUCFormer Single-Coil')
    parser.add_argument('--seed',default=42,type=int,help='Seed for random number generators')
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers (if applicable in model)')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability (if applicable in model)')
    parser.add_argument('--num-chans', type=int, default=32, help='Number of base channels (if applicable in model)')
    parser.add_argument('--batch-size', default=2, type=int,  help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--data-parallel', action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-dir', type=str, default='checkpoints_sc', # Changed default
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str, default=None, # Added default
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--train-path',type=str, default=None, # Added default
                        help='Path to train h5 files (reads from .env SINGLECOIL_TRAIN_PATH if set)')
    parser.add_argument('--validation-path',type=str, default=None, # Added default
                        help='Path to validation h5 files (reads from .env SINGLECOIL_VAL_PATH if set)')
    parser.add_argument('--timesteps', default=5, type=int,  help='Number of recurrent timesteps in model')
    parser.add_argument('--acceleration_factor',type=str, required=True, help='acceleration factors (e.g., 4x,5x)')
    parser.add_argument('--dataset_type',type=str, required=True, help='dataset types (e.g. mrbrain_t1,cardiac)')
    parser.add_argument('--usmask_path',type=str, default=None, # Added default
                        help='Path to undersampling masks directory (reads from .env USMASK_PATH if set)')
    parser.add_argument('--mask_type',type=str, required=True, help='mask type (e.g. cartesian, gaussian)')

    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Ensure CUDA seeds are set if using GPU
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed) # For multi-GPU
        # Potentially set deterministic options for reproducibility
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    print(args)
    main(args)