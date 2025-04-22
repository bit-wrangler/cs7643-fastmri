#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import warnings
warnings.simplefilter("ignore", UserWarning)

import torch
import os
from utils.options import args_parser
from models.recon_Update import train_recon
from models.evaluation import evaluate_recon
from data.mri_data import SliceData, DataTransform
from data.subsample import create_mask_for_mask_type
from models.Recurrent_Transformer import ReconFormer
from tensorboardX import SummaryWriter
import pathlib
from torch.utils.data import DataLoader

def main():
    

    os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # parse args
    args, parser = args_parser()
    path_dict = {'F': pathlib.Path(args.F_path)}
    resolution_dict = {'F': 320}
    rate_dict = {'F': 1.0}
    # args.device = torch.device('cuda:{}'.format(args.gpu[0]) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} CUDA device(s), requested: {args.gpu}")
    if torch.cuda.is_available() and isinstance(args.gpu, (list,tuple)) and args.gpu:
        # keep only valid IDs
        valid = [i for i in args.gpu if 0 <= i < n_gpus]
        if not valid:
            raise RuntimeError(f"No valid GPUs in {args.gpu}, only 0–{n_gpus-1} available.")
        device_id = valid[0]
        args.device = torch.device(f"cuda:{device_id}")
        print(f"Using CUDA device {device_id}")
    else:
        args.device = torch.device("cpu")
        print("Using CPU")

        print('cuda-check', torch.cuda.is_available(), torch.cuda.device_count())
    print('cuda-check', torch.cuda.get_device_name(0))
    # Try a tiny tensor
    _ = torch.tensor([1], device='cuda')
    # or even
    # _ = torch.zeros(1, device=args.device)


    args.resolution = resolution_dict[args.train_dataset]
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    writer = SummaryWriter(log_dir=args.save_dir/ 'summary')
    print_options(args, parser)
    def save_networks(net, epoch, local=False, local_no = None):
        """Save all the networks to the disk.        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        if local:
            save_filename = '%s_C%s_net.pth' % (epoch,local_no)
        else:
            save_filename = '%s_net.pth' % (epoch)
        save_path = os.path.join(args.save_dir, save_filename)
        if len(args.gpu) > 1 and torch.cuda.is_available():
            torch.save(net.module.cpu().state_dict(), save_path)
            net.to(args.device)
        else:
            torch.save(net.cpu().state_dict(), save_path)
            net.to(args.device)

    # data loader
    def _create_dataset(data_path,data_transform, data_partition, sequence, bs, shuffle, sample_rate=None, display=False):
        sample_rate = sample_rate or args.sample_rate
        dataset = SliceData(
            root=data_path / data_partition,
            transform=data_transform,
            sample_rate=sample_rate,
            challenge=args.challenge,
            sequence=sequence
        )
        if display:
            dataset = [dataset[i] for i in range(100,108)]
        return DataLoader(dataset, batch_size=bs, shuffle=shuffle, pin_memory=True,num_workers=8)
    

    def _create_dataset_2(data_path,data_transform, data_partition, sequence, bs, shuffle, sample_rate=None, display=False):
        sample_rate = sample_rate or args.sample_rate
        print(data_path, 'data_split', data_partition)
        dataset = SliceData(
            root=data_path / data_partition,
            transform=data_transform,
            sample_rate=sample_rate,
            challenge=args.challenge,
            sequence=sequence
        )
        if display:
            dataset = [dataset[i] for i in range(100,108)]
        return DataLoader(dataset, batch_size=bs, shuffle=shuffle, pin_memory=True,num_workers=8)

    # load dataset and split users
    if args.challenge == 'singlecoil':
        mask = create_mask_for_mask_type(args.mask_type, args.center_fractions,
                                         args.accelerations)
        train_data_transform = DataTransform(args.resolution, args.challenge, mask, use_seed=False)
        val_data_transform = DataTransform(args.resolution, args.challenge, mask, use_seed=True)

        if args.phase == 'train':
            dataset_train = _create_dataset(path_dict[args.train_dataset]/args.sequence,train_data_transform, 'train', args.sequence, args.bs, True, rate_dict[args.train_dataset])
            dataset_val = _create_dataset(path_dict[args.test_dataset]/args.sequence,val_data_transform, 'val', args.sequence, 8, False, 1.0)
        
            print("finished dataset loading")
            print(dataset_train.dataset.examples[0])
            print('loaded both')
            print(dataset_val.dataset.examples[0])
            # dataset_train = _create_dataset(path_dict[args.train_dataset]/args.sequence,train_data_transform, 'train', args.sequence, args.bs, True, rate_dict[args.train_dataset])
            # dataset_val = _create_dataset(path_dict[args.test_dataset]/args.sequence,val_data_transform, 'val', args.sequence, 8, False, 1.0)
    else:
        exit('Error: unrecognized challenge')

    # build model

    if args.model == 'ReconFormer':
        print('begininng training', args.device)
        net = ReconFormer(in_channels=4, out_channels=2, num_ch=(96, 48, 24),num_iter=5,
        down_scales=(2,1,1.5), img_size=args.resolution, num_heads=(6,6,6), depths=(2,1,1),
        window_sizes=(8,8,8), mlp_ratio=2., resi_connection ='1conv',
        use_checkpoint=(False, False, True, True, False, False)
        ).to(args.device)
        print('finished training')
    else:
        exit('Error: unrecognized model')
    print('before print networks')
    print_networks(net)

    # print('before parllel tyr')
    # if len(args.gpu) > 1:
    #     net = torch.nn.DataParallel(net, args.gpu)
    n_cuda = torch.cuda.device_count()
    print(f'CUDA available? {torch.cuda.is_available()}, device count: {n_cuda}')
    print('Requested GPUs:', args.gpu)

    if n_cuda > 1:
        # filter out any bad indices
        device_ids = [i for i in args.gpu if i < n_cuda]
        if not device_ids:
            raise RuntimeError(f'None of the requested GPUs {args.gpu} are valid (max index {n_cuda-1}).')
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        print(f'Using DataParallel on devices: {device_ids}')
    else:
        print('Single‐GPU or no GPUs found, skipping DataParallel')
    
    print('after parllel try')
    # training
    if args.phase == 'train':
        start_epoch = -1
        print('start training')
        if args.continues:
            if len(args.gpu) > 1:
                net.module.load_state_dict(torch.load(args.checkpoint))
            else:
                net.load_state_dict(torch.load(args.checkpoint))
            print('Load checkpoint :', args.checkpoint)
            start_epoch = int(args.checkpoint.split('/')[-1].split('_')[0])
        print('optimize')
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

        for iter in range(start_epoch+1,args.epochs):
            print('start training epoch %d' % (iter))
            loss_avg = train_recon(net, dataset_train, optimizer, iter, args, writer)
            scheduler.step(iter)
            torch.cuda.empty_cache()
            # print loss
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            print('saving the model at the end of epoch %d' % (iter))
            save_networks(net, iter)

            print('Evaluation ...')
            evaluate_recon(net, dataset_val, args, writer, iter)
            torch.cuda.empty_cache()
        writer.close()


def print_networks(net):
    """Print the total number of parameters in the network and (if verbose) network architecture

    Parameters:
        verbose (bool) -- if verbose: print the network architecture
    """
    print('---------- Networks initialized -------------')
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
    print('-----------------------------------------------')

def print_options(opt,parser):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        file_name = os.path.join(opt.save_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
if __name__ == '__main__':
    main()

