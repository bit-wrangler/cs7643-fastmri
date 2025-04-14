import h5py
import dotenv
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
import fastmri
import fastmri.data.transforms as T
import torch
# from models.singlecoil_kspace_pixelwise_masked_transformer_denoiser import SingleCoilKspacePixelwiseMaskedTransformerDenoiser
from models.singlecoil_kspace_columnwise_masked_transformer_denoiser import SingleCoilKspaceColumnwiseMaskedTransformerDenoiser
from fastmri.data.subsample import RandomMaskFunc


def show_coils(data, slice_nums, cmap=None):
    fig = plt.figure(figsize=(10, 4))
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.imshow(data[num], cmap=cmap)
    plt.show()

dotenv.load_dotenv()

SINGLECOIL_TRAIN_PATH = os.environ.get('SINGLECOIL_TRAIN_PATH')
SINGLECOIL_VAL_PATH = os.environ.get('SINGLECOIL_VAL_PATH')

print(f'SINGLECOIL_TRAIN_PATH: {SINGLECOIL_TRAIN_PATH}')
print(f'SINGLECOIL_VAL_PATH: {SINGLECOIL_VAL_PATH}')

train_files = glob.glob(os.path.join(SINGLECOIL_TRAIN_PATH, '*.h5'))

file_name = train_files[0]
hf = h5py.File(file_name)

print('Keys:', list(hf.keys()))
print('Attrs:', dict(hf.attrs))

volume_kspace = hf['kspace'][()]
print(volume_kspace.dtype)
print(volume_kspace.shape)

reconstruction_esc = hf['reconstruction_esc'][()]
print(reconstruction_esc.dtype)
print(reconstruction_esc.shape)

reconstruction_rss = hf['reconstruction_rss'][()]
print(reconstruction_rss.dtype)
print(reconstruction_rss.shape)

# plt.imshow(np.abs(reconstruction_esc[5]), cmap='gray')
# plt.show()

volume = T.to_tensor(volume_kspace)
print(volume.shape)

model = SingleCoilKspaceColumnwiseMaskedTransformerDenoiser()

mask_func = RandomMaskFunc(center_fractions=[0.04], accelerations=[8])
seed=42

mask, _ = mask_func((1,) * len(volume.shape[:-3]) + tuple(volume.shape[-3:]), None,seed)
mask = mask.type(torch.bool)
print(mask.shape)

volume_input = volume.permute(0, 3, 1, 2)

output = model(volume_input, mask).detach()

output_permuted = output.permute(0, 2, 3, 1)

show_coils(np.log(np.abs(volume_kspace) + 1e-9), [0, 5, 10])

output_complex = T.tensor_to_complex_np(output_permuted.contiguous())
show_coils(np.log(np.abs(output_complex) + 1e-9), [0, 5, 10])

output_image = fastmri.ifft2c(output_permuted)
output_image_abs = fastmri.complex_abs(output_image)

show_coils(output_image_abs, [0, 5, 10], cmap='gray')