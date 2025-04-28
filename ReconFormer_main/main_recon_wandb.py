#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ReconFormer Training with Weights & Biases Logging
"""
import warnings
warnings.simplefilter("ignore", UserWarning)

import os
import pathlib
import torch
import wandb
from torch.utils.data import DataLoader

from utils.options import args_parser
from models.recon_Update import train_recon
from models.evaluation import evaluate_recon
from data.mri_data import SliceData, DataTransform
from data.subsample import create_mask_for_mask_type
from models.Recurrent_Transformer import ReconFormer
from tensorboardX import SummaryWriter

import dotenv
dotenv.load_dotenv()


def _create_dataset(data_path, data_transform, partition, sequence, bs, shuffle, sample_rate=None):
    sample_rate = sample_rate or 1.0
    dataset = SliceData(
        root=data_path / partition,
        transform=data_transform,
        sample_rate=sample_rate,
        challenge=args.challenge,
        sequence=sequence
    )
    return DataLoader(dataset, batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=8)


def print_options(opt, parser):
    message = '\n----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        default = parser.get_default(k)
        comment = '' if v == default else f"\t[default: {default}]"
        message += f'{k:>15}: {v:<20}{comment}\n'
    message += '----------------- End -------------------'
    print(message)
    os.makedirs(opt.save_dir, exist_ok=True)
    with open(os.path.join(opt.save_dir, f"{opt.phase}_opt.txt"), 'wt') as f:
        f.write(message)


def main():
    os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
    torch.backends.cudnn.benchmark = True

    # parse args
    args, parser = args_parser()

    # device selection
    n_gpus = torch.cuda.device_count()
    if torch.cuda.is_available() and args.gpu and args.gpu[0] >= 0:
        device_id = args.gpu[0] if args.gpu[0] < n_gpus else 0
        args.device = torch.device(f"cuda:{device_id}")
    else:
        args.device = torch.device("cpu")
    print(f"Using device: {args.device}")

    # resolution mapping
    resolution_map = {'F': 320}
    args.resolution = resolution_map.get(getattr(args, 'train_dataset', None), 320)

    # create summary writer
    writer = SummaryWriter(log_dir=str(pathlib.Path(args.save_dir) / 'tensorboard'))

    # print and save options
    print_options(args, parser)

    # initialize Weights & Biases
    wandb_project = os.getenv('WANDB_PROJECT_NAME', 'cs7643-fastmri')
    run_name = f"Train_{args.model}_{args.train_dataset}_{args.sequence}"
    run = wandb.init(
        project=wandb_project,
        name=run_name,
        config=vars(args),
        tags=['training', args.model],
        job_type='train'
    )
    print(f"W&B run initialized: {run.name} (ID: {run.id})")

    # data transforms and loaders
    path_dict = {'F': pathlib.Path(args.F_path)}
    mask = create_mask_for_mask_type(args.mask_type, args.center_fractions, args.accelerations)
    train_transform = DataTransform(args.resolution, args.challenge, mask, use_seed=False)
    val_transform   = DataTransform(args.resolution, args.challenge, mask, use_seed=True)

    dataset_train = _create_dataset(path_dict[args.train_dataset]/args.sequence,
                                    train_transform, 'train', args.sequence,
                                    args.bs, True, sample_rate=args.sample_rate)
    dataset_val   = _create_dataset(path_dict[args.test_dataset]/args.sequence,
                                    val_transform,   'val',   args.sequence,
                                    args.bs, False, sample_rate=1.0)

    # model instantiation
    net = ReconFormer(in_channels=2,
                      out_channels=2,
                      num_ch=(96,48,24),
                      num_iter=args.num_iter,
                      down_scales=args.down_scales,
                      img_size=args.resolution,
                      num_heads=args.num_heads,
                      depths=args.depths,
                      window_sizes=args.window_sizes,
                      mlp_ratio=args.mlp_ratio,
                      resi_connection=args.resi_connection,
                      use_checkpoint=args.use_checkpoint
    ).to(args.device)
    wandb.watch(net, log='all', log_graph=True)

    # optimizer and scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=args.lr_step_size,
                                                gamma=args.lr_gamma)

    # optionally resume
    start_epoch = 0
    if args.continues and args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        key = 'state_dict' if 'state_dict' in ckpt else None
        state = ckpt[key] if key else ckpt
        net.load_state_dict({k.replace('module.',''):v for k,v in state.items()})
        print(f"Resumed from checkpoint: {args.checkpoint}")
        start_epoch = int(pathlib.Path(args.checkpoint).stem.split('_')[0]) + 1

    # training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch}/{args.epochs - 1}")
        loss_train = train_recon(net, dataset_train, optimizer, epoch, args, writer)
        scheduler.step()

        # log training loss
        print(f"Train Loss: {loss_train:.4f}")
        wandb.log({'Train/Loss': loss_train, 'epoch': epoch})

        # validation
        metrics = evaluate_recon(net, dataset_val, args, writer, epoch)
        # assume evaluate_recon returns a dict of metrics
        if isinstance(metrics, dict):
            log_dict = {f'Val/{k}': v for k,v in metrics.items()}
            log_dict['epoch'] = epoch
            wandb.log(log_dict)
            print(f"Val metrics: {metrics}")

        # save checkpoint every epoch
        ckpt_path = pathlib.Path(args.save_dir) / f"{epoch}_net.pth"
        if isinstance(net, torch.nn.DataParallel):
            torch.save(net.module.state_dict(), ckpt_path)
        else:
            torch.save(net.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    writer.close()
    wandb.finish()


if __name__ == '__main__':
    main()
