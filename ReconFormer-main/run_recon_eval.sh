#!/bin/env bash
# ReconFormer Evaluation
python main_recon_test_wandb.py --phase test --gpu 0 --model ReconFormer --bs 1 --challenge singlecoil --F_path /home/jschopme/dl/cs7643-fastmri/ReconFormer-main/data/ --test_dataset F --sequence PD --accelerations 8 --center-fractions 0.04 --checkpoint checkpoints/checkpoints/F_X8_checkpoint.pth --verbose