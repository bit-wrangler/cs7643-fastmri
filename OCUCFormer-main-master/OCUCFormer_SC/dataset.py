import pathlib
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch
# from skimage import feature # Unused import
import os
# Remove dependency on utils.npComplexToTorch, use fastmri transforms
# from utils import npComplexToTorch
import warnings # Import warnings

# --- Import necessary fastmri transforms ---
try:
    # Try importing from the standard location
    import fastmri.data.transforms as T
except ImportError:
    # Fallback or error if fastmri not installed properly
    print("Error: Could not import fastmri.data.transforms.")
    print("Please ensure the fastmri library is installed correctly (`pip install fastmri`).")
    raise

class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    Loads raw kspace, applies mask, computes input image via iFFT.
    """
    def __init__(self, root: pathlib.Path, acc_factors: list, dataset_types: list, mask_types: list, train_or_valid: str, mask_path: pathlib.Path):
        self.examples = []
        self.mask_path = mask_path
        # Store first elements for consistency in key checking and mask loading path structure
        self.acc_factor = acc_factors[0]
        self.dataset_type = dataset_types[0]
        self.mask_type = mask_types[0]

        # --- Simplified Path Logic ---
        data_files_dir = root
        if not data_files_dir.is_dir():
            raise FileNotFoundError(f"Data directory specified ({train_or_valid}) not found: {data_files_dir}")

        print(f"Looking for {train_or_valid} HDF5 files in: {data_files_dir}")
        file_count = 0
        example_count = 0

        # Iterate through all files in the root directory
        for fname in sorted(list(data_files_dir.iterdir())):
            if fname.is_file() and fname.suffix in ['.h5', '.hdf5']:
                file_count += 1
                try:
                    with h5py.File(fname, 'r') as hf:
                        # Check if essential RAW data keys exist
                        if 'kspace' in hf and 'reconstruction_esc' in hf and hf['kspace'].ndim >= 3:
                            # Get num_slices from kspace now
                            num_slices = hf['kspace'].shape[0]
                            # Add examples for each slice
                            # Store necessary info: file path, slice index
                            # We'll use self.acc_factor etc. in getitem
                            self.examples += [(fname, slice_idx) for slice_idx in range(num_slices)]
                            example_count += num_slices
                        else:
                            print(f"Warning: Skipping file {fname}, missing required keys ('kspace', 'reconstruction_esc') or unexpected 'kspace' shape.")
                except Exception as e:
                    print(f"Warning: Could not process file {fname}: {e}")

        print(f"Found {file_count} HDF5 files and added {example_count} examples for {train_or_valid}.")
        if example_count == 0:
             print(f"ERROR: No valid examples found in {data_files_dir}. Check HDF5 files and keys.")
        # -----------------------------

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Retrieve filename and slice index
        fname, slice_num = self.examples[i]

        try:
            with h5py.File(fname, 'r') as data:
                # 1. Load raw kspace and target
                kspace_np = data['kspace'][slice_num] # Shape: (H, W) complex
                target_np = data['reconstruction_esc'][slice_num] # Shape: (H, W) float/real

                # 2. Convert kspace to PyTorch tensor (H, W, 2)
                kspace_torch = T.to_tensor(kspace_np) # Shape: (H, W, 2)

                # 3. Load the mask
                mask_file = self.mask_path / self.dataset_type / self.mask_type / f'mask_{self.acc_factor}.npy'
                if not mask_file.is_file():
                     raise FileNotFoundError(f"Mask file not found: {mask_file}")
                mask_np = np.load(mask_file) # Shape: (W,)
                mask_torch = torch.from_numpy(mask_np).float() # Ensure float tensor

                # 4. Apply mask to kspace
                # Expand mask to match kspace shape (H, W, 1) for broadcasting
                # Mask needs to be applied to both real and imaginary (last dim size 2)
                h, w, _ = kspace_torch.shape
                mask_expanded = mask_torch.reshape(1, w, 1).expand(h, w, 1)
                # Apply the mask (multiply k-space columns)
                masked_kspace_torch = kspace_torch * mask_expanded

                # 5. Compute undersampled image via iFFT
                # Input to ifft2 should be complex (H, W, 2)
                input_img_torch = T.ifft2(masked_kspace_torch) # Output: (H, W, 2)

                # We usually need the magnitude image as input to the model
                input_img_abs = T.complex_abs(input_img_torch) # Output: (H, W)

                # 6. Prepare target tensor
                target_torch = torch.from_numpy(target_np).float() # Ensure float

                # 7. Prepare mask tensor for return (model might need it)
                # Return the expanded mask used for calculations? Or the 1D mask?
                # The train_epoch uses mask directly with input/kspace, let's return the expanded one used
                # Note: train_epoch passes this mask to the model, ensure model expects (H,W,1) or similar
                # Let's return the 1D mask that was loaded, as the model might expect that format
                # Or let's return the expanded mask without the complex dim for now
                # return_mask = mask_expanded.squeeze(-1) # Shape (H, W)
                # Check train_epoch: model(input, input_kspace, mask) - mask needs to be compatible
                # If model does DC, it needs kspace and mask. Let's return the 1D mask (W,)
                # and the masked_kspace (H, W, 2). Input image is also needed.
                return_mask_torch = mask_torch # Return the original 1D mask (W,)

            # Ensure correct data types for model (often float32)
            input_img_final = input_img_abs.float()
            masked_kspace_final = masked_kspace_torch.float() # Model might expect float real/imag pairs
            target_final = target_torch.float()
            return_mask_final = return_mask_torch.float()

            # Return: undersampled_image, masked_kspace, target_image, 1D_mask
            return input_img_final, masked_kspace_final, target_final, return_mask_final

        except Exception as e:
             print(f"Error loading data for index {i}, file {fname}, slice {slice_num}: {e}")
             raise e


# --- SliceDataDev ---
# Needs similar modification to load raw kspace and apply mask on the fly
class SliceDataDev(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices for Dev/Validation.
    Loads raw kspace, applies mask, computes input image via iFFT.
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
                            # Check for raw data keys
                            if 'kspace' in hf and 'reconstruction_esc' in hf and hf['kspace'].ndim >= 3:
                                num_slices = hf['kspace'].shape[0]
                                self.examples += [(fname, slice_idx) for slice_idx in range(num_slices)]
                                example_count += num_slices
                            else:
                                print(f"Warning: Skipping validation file {fname}, missing required keys ('kspace', 'reconstruction_esc') or unexpected 'kspace' shape.")
                    except Exception as e:
                         print(f"Warning: Could not read metadata from {fname}: {e}")
        except Exception as e:
             print(f"Error listing files in {self.root}: {e}")
        print(f"Found {file_count} validation HDF5 files and added {example_count} examples.")


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice_num = self.examples[i]

        try:
            with h5py.File(fname, 'r') as data:
                 # 1. Load raw kspace and target
                kspace_np = data['kspace'][slice_num]
                target_np = data['reconstruction_esc'][slice_num]

                # 2. Convert kspace to PyTorch tensor (H, W, 2)
                kspace_torch = T.to_tensor(kspace_np)

                # 3. Load the mask
                mask_file = self.mask_path / self.dataset_type / self.mask_type / f'mask_{self.acc_factor}.npy'
                if not mask_file.is_file():
                     raise FileNotFoundError(f"Mask file not found: {mask_file}")
                mask_np = np.load(mask_file)
                mask_torch = torch.from_numpy(mask_np).float()

                # 4. Apply mask to kspace
                h, w, _ = kspace_torch.shape
                mask_expanded = mask_torch.reshape(1, w, 1).expand(h, w, 1)
                masked_kspace_torch = kspace_torch * mask_expanded

                # 5. Compute undersampled image via iFFT
                input_img_torch = T.ifft2(masked_kspace_torch)
                input_img_abs = T.complex_abs(input_img_torch)

                # 6. Prepare target tensor
                target_torch = torch.from_numpy(target_np).float()

                # 7. Prepare mask tensor for return (1D version)
                return_mask_torch = mask_torch

            input_img_final = input_img_abs.float()
            masked_kspace_final = masked_kspace_torch.float()
            target_final = target_torch.float()
            return_mask_final = return_mask_torch.float()

            # Return items needed by valid.py/evaluate.py (check those scripts)
            # valid.py likely needs: input_img, masked_kspace, target, mask, fname, slice_num
            # Let's return input_img_abs, masked_kspace, target, 1D mask, fname, slice_num
            return input_img_final, masked_kspace_final, target_final, return_mask_final, str(fname.name), slice_num

        except Exception as e:
             print(f"Error loading validation data for index {i}, file {fname}, slice {slice_num}: {e}")
             raise e


# --- SliceDisplayDataDev ---
# Needs similar modification if it's used (e.g., by train.py visualize)
class SliceDisplayDataDev(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices for display.
    Loads raw kspace, applies mask, computes input image via iFFT.
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
             return

        # Construct mask path using pathlib - check existence
        self.mask_load_path = self.mask_path / self.dataset_type / self.mask_type / f'mask_{self.acc_factor}.npy'
        if not self.mask_load_path.is_file():
            # Raise error here as display won't work without mask
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
                             # Check for raw data keys
                            if 'kspace' in hf and 'reconstruction_esc' in hf and hf['kspace'].ndim >= 3:
                                num_slices = hf['kspace'].shape[0]
                                self.examples += [(fname, slice_idx) for slice_idx in range(num_slices)]
                                example_count += num_slices
                            else:
                                print(f"Warning: Skipping display file {fname}, missing required keys ('kspace', 'reconstruction_esc') or unexpected 'kspace' shape.")
                    except Exception as e:
                        print(f"Warning: Could not read metadata from display file {fname}: {e}")
        except Exception as e:
             print(f"Error listing files in display directory {data_files_dir}: {e}")
        print(f"Found {file_count} display HDF5 files and added {example_count} display examples.")


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice_num = self.examples[i]

        try:
            with h5py.File(fname, 'r') as data:
                # 1. Load raw kspace and target
                kspace_np = data['kspace'][slice_num]
                target_np = data['reconstruction_esc'][slice_num]

                # 2. Convert kspace to PyTorch tensor (H, W, 2)
                kspace_torch = T.to_tensor(kspace_np)

                # 3. Load the mask
                mask_np = np.load(self.mask_load_path)
                mask_torch = torch.from_numpy(mask_np).float()

                # 4. Apply mask to kspace
                h, w, _ = kspace_torch.shape
                mask_expanded = mask_torch.reshape(1, w, 1).expand(h, w, 1)
                masked_kspace_torch = kspace_torch * mask_expanded

                # 5. Compute undersampled image via iFFT
                input_img_torch = T.ifft2(masked_kspace_torch)
                input_img_abs = T.complex_abs(input_img_torch)

                # 6. Prepare target tensor
                target_torch = torch.from_numpy(target_np).float()

                # 7. Prepare mask tensor for return (1D version)
                return_mask_torch = mask_torch

            input_img_final = input_img_abs.float()
            masked_kspace_final = masked_kspace_torch.float() # Pass masked kspace
            target_final = target_torch.float()
            return_mask_final = return_mask_torch.float() # Pass 1D mask

            # Return items needed by train.py visualize function: input, input_kspace, target, mask
            # Note: visualize expects 4 items, let's match train_epoch's return format for now
            # Return: undersampled_image, masked_kspace, target_image, 1D_mask
            return input_img_final, masked_kspace_final, target_final, return_mask_final

        except Exception as e:
             print(f"Error loading display data for index {i}, file {fname}, slice {slice_num}: {e}")
             raise e