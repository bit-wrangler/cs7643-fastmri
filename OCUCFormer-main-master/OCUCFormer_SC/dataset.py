# OCUCFormer-main-master/OCUCFormer_SC/dataset.py

import pathlib
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch
import os
import warnings # Import warnings
import logging # Import logging
import sys # Import sys for exit
import torch.nn.functional as F # Import F for padding

# --- Import necessary fastmri transforms ---
try:
    # Try importing from the standard location
    import fastmri.data.transforms as T
    import fastmri
except ImportError as e:
    print(f"Error importing fastmri library: {e}")
    print("Please ensure the fastmri library is installed correctly (`pip install fastmri`).")
    # Exit if fastmri is essential and not found
    sys.exit(1)

# Define the standard width for intermediate k-space processing (matching mask generation)
KSPACE_CROP_WIDTH = 368
# Define the final image size for model input/output/target comparison
FINAL_IMG_SIZE = 320

# --- Helper Function for K-space Crop or Pad ---
def crop_or_pad_kspace(kspace: torch.Tensor, target_shape: tuple):
    """
    Crop or pad k-space tensor from shape [2, H, W] to [2, target_h, target_w].
    """
    target_h, target_w = target_shape
    c, h, w = kspace.shape

    # --- Crop ---
    crop_h = max(0, h - target_h)
    crop_w = max(0, w - target_w)
    h_start = crop_h // 2
    w_start = crop_w // 2
    cropped = kspace[:, h_start:h_start + min(h, target_h), w_start:w_start + min(w, target_w)]

    # --- Pad ---
    pad_h = max(0, target_h - cropped.shape[1])
    pad_w = max(0, target_w - cropped.shape[2])
    if pad_h or pad_w:
        # F.pad expects (left, right, top, bottom)
        pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        padded = F.pad(cropped, pad, mode='constant', value=0)
        if padded.shape[1:] != (target_h, target_w):
            raise RuntimeError(f"K-space pad failed: {padded.shape} vs {(2,target_h,target_w)}")
        return padded
    else:
        if cropped.shape[1:] != (target_h, target_w):
            raise RuntimeError(f"K-space crop failed: {cropped.shape} vs {(2,target_h,target_w)}")
        return cropped


def crop_or_pad_image(img: torch.Tensor, target_shape: tuple):
    """
    Crop or pad image tensor from shape [H, W] to [target_h, target_w].
    """
    target_h, target_w = target_shape
    h, w = img.shape

    # --- Crop ---
    crop_h = max(0, h - target_h)
    crop_w = max(0, w - target_w)
    h_start = crop_h // 2
    w_start = crop_w // 2
    cropped = img[h_start:h_start + min(h, target_h), w_start:w_start + min(w, target_w)]

    # --- Pad ---
    pad_h = max(0, target_h - cropped.shape[0])
    pad_w = max(0, target_w - cropped.shape[1])
    if pad_h or pad_w:
        pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        padded = F.pad(cropped, pad, mode='constant', value=0)
        if padded.shape != (target_h, target_w):
            raise RuntimeError(f"Image pad failed: {padded.shape} vs {(target_h,target_w)}")
        return padded
    else:
        if cropped.shape != (target_h, target_w):
            raise RuntimeError(f"Image crop failed: {cropped.shape} vs {(target_h,target_w)}")
        return cropped


# --- Dataset Classes ---

class SliceData(Dataset):
    """
    Training dataset: returns (input_image, masked_kspace, target_image, mask_1d).
    - input_image: [1, 320, 320]
    - masked_kspace: [2, H, W] (H based on target height, W=368)
    - target_image: [1, 320, 320]
    - mask_1d: [368]
    """
    def __init__(self, root: pathlib.Path, acc_factors, dtypes, mtypes, split: str, mask_path: pathlib.Path):
        self.examples = []
        self.mask_root = mask_path
        self.acc = acc_factors[0]
        self.dtype = dtypes[0]
        self.mtype = mtypes[0]

        root = pathlib.Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"Data dir not found: {root}")

        for f in sorted(root.iterdir()):
            if f.suffix in ['.h5', '.hdf5']:
                try:
                    with h5py.File(f, 'r') as hf:
                        ns = hf['kspace'].shape[0]
                        self.examples += [(f, i) for i in range(ns)]
                except Exception:
                    logging.warning(f"Skipping bad file: {f}")
        if not self.examples:
            raise ValueError(f"No examples found in {root}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        f, sl = self.examples[idx]
        try:
            with h5py.File(f, 'r') as hf:
                ksp = hf['kspace'][sl]              # shape (H, W, 2)
                tgt = hf['reconstruction_esc'][sl]  # shape (H, W)

            # to_tensor -> [2, H, W]
            ksp_t = T.to_tensor(ksp)
            # crop/pad k-space to [2, H', 368]
            target_h = tgt.shape[0]
            ksp2 = crop_or_pad_kspace(ksp_t, (target_h, KSPACE_CROP_WIDTH))

            # load 1D mask
            mask_file = self.mask_root / self.dtype / self.mtype / f'mask_{self.acc}.npy'
            m = np.load(mask_file)
            if m.shape[0] != KSPACE_CROP_WIDTH:
                raise ValueError(f"Mask length {m.shape[0]} != {KSPACE_CROP_WIDTH}")
            m_t = torch.from_numpy(m).float().to(ksp2.device)

            # apply mask: broadcast over [2, H', W]
            ksp_m = ksp2 * m_t[None, None, :]

            # iFFT -> complex_abs -> [H', W]
            im = fastmri.ifft2c(ksp_m)
            im = fastmri.complex_abs(im)
            im1 = crop_or_pad_image(im, (FINAL_IMG_SIZE, FINAL_IMG_SIZE))

            # target
            tgt_t = torch.from_numpy(tgt).float().to(im1.device)
            tgt1 = crop_or_pad_image(tgt_t, (FINAL_IMG_SIZE, FINAL_IMG_SIZE))

            return (
                im1.unsqueeze(0),      # [1,320,320]
                ksp_m.float(),         # [2,H',368]
                tgt1.unsqueeze(0),     # [1,320,320]
                m_t                     # [368]
            )
        except Exception:
            logging.error(f"Error idx {idx}, file {f}, slice {sl}", exc_info=True)
            raise


class SliceDataDev(Dataset):
    """
    Validation/Dev dataset. Loads raw kspace, crops/pads kspace, applies mask,
    computes input image via iFFT, crops/pads input, loads target, crops/pads target.
    Also returns filename and slice number.
    """
    def __init__(self, root: pathlib.Path, acc_factor: str, dataset_type: str, mask_type: str, mask_path: pathlib.Path):
        self.root = root
        self.mask_path = mask_path
        self.examples = []
        self.acc_factor = acc_factor
        self.mask_type = mask_type
        self.dataset_type = dataset_type

        if not self.root.is_dir():
             raise FileNotFoundError(f"Validation data directory not found: {self.root}")

        print(f"Looking for validation HDF5 files in: {self.root}")
        file_count = 0
        example_count = 0
        try:
            files = list(self.root.iterdir())
            for fname in sorted(files):
                 if fname.is_file() and fname.suffix in ['.h5', '.hdf5']:
                    file_count += 1
                    try:
                        with h5py.File(fname, 'r') as hf:
                            if 'kspace' in hf and 'reconstruction_esc' in hf and hf['kspace'].ndim >= 3:
                                num_slices = hf['kspace'].shape[0]
                                self.examples += [(fname, slice_idx) for slice_idx in range(num_slices)]
                                example_count += num_slices
                            else:
                                logging.warning(f"Skipping validation file {fname}, missing required keys or unexpected shape.")
                    except Exception as e:
                         logging.warning(f"Could not read metadata from {fname}: {e}")
        except Exception as e:
             print(f"Error listing files in {self.root}: {e}")
        print(f"Found {file_count} validation HDF5 files and added {example_count} examples.")


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice_num = self.examples[i]

        try:
            with h5py.File(fname, 'r') as data:
                kspace_np = data['kspace'][slice_num]
                target_np = data['reconstruction_esc'][slice_num]

                kspace_torch_orig = T.to_tensor(kspace_np)

                kspace_crop_height = target_np.shape[0]
                kspace_torch = crop_or_pad_kspace(kspace_torch_orig, (kspace_crop_height, KSPACE_CROP_WIDTH))

                mask_file = self.mask_path / self.dataset_type / self.mask_type / f'mask_{self.acc_factor}.npy'
                if not mask_file.is_file(): raise FileNotFoundError(f"Mask file not found: {mask_file}")
                mask_np = np.load(mask_file)
                if mask_np.shape[0] != KSPACE_CROP_WIDTH: raise ValueError(f"Mask width {mask_np.shape[0]} != KSPACE_CROP_WIDTH {KSPACE_CROP_WIDTH}")
                mask_torch = torch.from_numpy(mask_np).float()

                h, w, _ = kspace_torch.shape
                mask_expanded = mask_torch.reshape(1, w, 1).expand(h, w, 1)
                masked_kspace_torch = kspace_torch * mask_expanded

                input_img_torch = fastmri.ifft2c(masked_kspace_torch)
                input_img_abs = fastmri.complex_abs(input_img_torch)
                input_img_final = crop_or_pad_image(input_img_abs, (FINAL_IMG_SIZE, FINAL_IMG_SIZE))

                target_torch_orig = torch.from_numpy(target_np).float()
                target_final = crop_or_pad_image(target_torch_orig, (FINAL_IMG_SIZE, FINAL_IMG_SIZE))

                return_mask_final = mask_torch.float()

            # Return: final_input_image, processed_masked_kspace, final_target_image, 1D_mask, fname_str, slice_int
            return input_img_final.float(), masked_kspace_torch.float(), target_final.float(), return_mask_final, str(fname.name), slice_num

        except Exception as e:
             logging.error(f"Error loading validation data for index {i}, file {fname}, slice {slice_num}: {e}", exc_info=True)
             raise e


class SliceDisplayDataDev(Dataset):
    """
    Dataset for visualization. Loads raw kspace, crops/pads kspace, applies mask,
    computes input image via iFFT, crops/pads input, loads target, crops/pads target.
    Returns items needed by train_wand.py's visualize function.
    """
    def __init__(self, root: pathlib.Path, dataset_type: str, mask_type: str, acc_factor: str, mask_path: pathlib.Path):
        self.examples = []
        self.acc_factor = acc_factor
        self.dataset_type = dataset_type
        self.mask_type = mask_type
        self.mask_path = mask_path

        data_files_dir = root
        if not data_files_dir.is_dir():
             print(f"Warning: Display data directory not found: {data_files_dir}")
             return # Allow empty display dataset

        self.mask_load_path = self.mask_path / self.dataset_type / self.mask_type / f'mask_{self.acc_factor}.npy'
        if not self.mask_load_path.is_file():
            raise FileNotFoundError(f"Display mask file not found: {self.mask_load_path}")

        print(f"Looking for display HDF5 files in: {data_files_dir}")
        file_count = 0
        example_count = 0
        try:
            files = list(data_files_dir.iterdir())
            for fname in sorted(files):
                if fname.is_file() and fname.suffix in ['.h5', '.hdf5']:
                    file_count += 1
                    try:
                        with h5py.File(fname, 'r') as hf:
                            if 'kspace' in hf and 'reconstruction_esc' in hf and hf['kspace'].ndim >= 3:
                                num_slices = hf['kspace'].shape[0]
                                self.examples += [(fname, slice_idx) for slice_idx in range(num_slices)]
                                example_count += num_slices
                            else:
                                logging.warning(f"Skipping display file {fname}, missing required keys or unexpected shape.")
                    except Exception as e:
                        logging.warning(f"Could not read metadata from display file {fname}: {e}")
        except Exception as e:
             print(f"Error listing files in display directory {data_files_dir}: {e}")
        print(f"Found {file_count} display HDF5 files and added {example_count} display examples.")


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice_num = self.examples[i]

        try:
            with h5py.File(fname, 'r') as data:
                kspace_np = data['kspace'][slice_num]
                target_np = data['reconstruction_esc'][slice_num]

                kspace_torch_orig = T.to_tensor(kspace_np)

                kspace_crop_height = target_np.shape[0]
                kspace_torch = crop_or_pad_kspace(kspace_torch_orig, (kspace_crop_height, KSPACE_CROP_WIDTH))

                mask_np = np.load(self.mask_load_path)
                if mask_np.shape[0] != KSPACE_CROP_WIDTH: raise ValueError(f"Mask width {mask_np.shape[0]} != KSPACE_CROP_WIDTH {KSPACE_CROP_WIDTH}")
                mask_torch = torch.from_numpy(mask_np).float()

                h, w, _ = kspace_torch.shape
                mask_expanded = mask_torch.reshape(1, w, 1).expand(h, w, 1)
                masked_kspace_torch = kspace_torch * mask_expanded

                input_img_torch = fastmri.ifft2c(masked_kspace_torch)
                input_img_abs = fastmri.complex_abs(input_img_torch)
                input_img_final = crop_or_pad_image(input_img_abs, (FINAL_IMG_SIZE, FINAL_IMG_SIZE))

                target_torch_orig = torch.from_numpy(target_np).float()
                target_final = crop_or_pad_image(target_torch_orig, (FINAL_IMG_SIZE, FINAL_IMG_SIZE))

                return_mask_final = mask_torch.float()

            # Return items needed by train_wand.py visualize function: input, kspace, target, mask
            return input_img_final.float(), masked_kspace_torch.float(), target_final.float(), return_mask_final

        except Exception as e:
             logging.error(f"Error loading display data for index {i}, file {fname}, slice {slice_num}: {e}", exc_info=True)
             raise e