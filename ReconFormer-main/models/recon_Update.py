#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import time
from torch.nn import functional as F
from data import transforms

# def train_recon(net, data_loader, optimizer, epoch, args, writer):
#     net.train()
#     # train and update
#     epoch_loss = []
#     batch_loss = []
#     iter_data_time = time.time()
#     for batch_idx, batch in enumerate(data_loader):
#         input, target, mean, std, norm, fname, slice, max, mask, masked_kspace = batch
#         output = net(input.to(args.device), masked_kspace.to(args.device), mask.to(args.device))
#         target = target.to(args.device)

#         output = transforms.complex_abs(output)
#         target = transforms.complex_abs(target)

#         loss = F.l1_loss(output, target)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if args.verbose and batch_idx % 10 == 0:
#             print('Update Epoch: {}  [{}/{} ({:.0f}%)]\tLoss: {:.12f}'.format(
#                 epoch, batch_idx * len(input), len(data_loader.dataset),
#                        100. * batch_idx / len(data_loader), loss.detach().item()))
#             t_comp = (time.time() - iter_data_time)
#             iter_data_time = time.time()
#             print('itr time: ',t_comp)
#             print('lr: ',optimizer.param_groups[0]['lr'])
#         batch_loss.append(loss.detach().item())
#         epoch_loss.append(sum(batch_loss)/len(batch_loss))
#         writer.add_scalar('TrainLoss/L1/'+ args.train_dataset, sum(epoch_loss) / len(epoch_loss), epoch)

#     return sum(epoch_loss) / len(epoch_loss)
import torch
def train_recon(net, data_loader, optimizer, epoch, args, writer):
    net.train()
    epoch_losses = []

    for batch_idx, batch in enumerate(data_loader):
        # unpack
        input_c, target_c, mean, std, norm, fname, slice_idx, maxval, mask, masked_kspace_c = batch

        # 1) make sure the tensors are 4-D: [B, 1, H, W]
        #    (fastMRI DataTransform usually returns [B, H, W], dtype=complex64)
        if input_c.dim() == 3:
            input_c = input_c.unsqueeze(1)
        if masked_kspace_c.dim() == 3:
            masked_kspace_c = masked_kspace_c.unsqueeze(1)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        # 2) split into real + imag, cat on channel axis → [B, 2, H, W]
        x  = torch.cat([input_c.real,           input_c.imag],           dim=1).to(args.device)
        k0 = torch.cat([masked_kspace_c.real, masked_kspace_c.imag], dim=1).to(args.device)
        mask = mask.to(args.device)  # now [B,1,H,W] float

        # debug sanity check (optional)
        # print("x:", x.dtype, x.shape, "k0:", k0.dtype, k0.shape, "mask:", mask.dtype, mask.shape)

        # 3) forward
        output_c = net(x, k0, mask)

        # 4) compute loss on magnitude
        #    transforms.complex_abs expects a complex input,
        #    and your target_c is still complex64
        output_mag = transforms.complex_abs(output_c)
        target_mag = transforms.complex_abs(target_c.to(args.device))

        loss = F.l1_loss(output_mag, target_mag)

        # 5) backward + step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 6) logging
        epoch_losses.append(loss.item())
        if batch_idx % 10 == 0 and args.verbose:
            print(f"Epoch {epoch}  [{batch_idx}/{len(data_loader)}]  Loss: {loss.item():.6f}")
        writer.add_scalar(f"TrainLoss/L1/{args.train_dataset}",
                          sum(epoch_losses) / len(epoch_losses),
                          epoch)

    return sum(epoch_losses) / len(epoch_losses)
