import numpy as np
import pathlib
import argparse
from fastmri.data import subsample

def generate_cartesian_masks(output_dir, accelerations, center_fractions, shape):
    """
    Generates and saves Cartesian undersampling masks.

    Args:
        output_dir (pathlib.Path): Directory to save the generated masks.
        accelerations (list[int]): List of acceleration factors (e.g., [4, 5]).
        center_fractions (list[float]): List of center fractions corresponding
                                         to the accelerations (e.g., [0.08, 0.04]).
        shape (tuple[int, int]): The shape of the k-space (e.g., (640, 368)).
                                  The mask will be generated based on the last dimension (width).
    """
    if len(accelerations) != len(center_fractions):
        raise ValueError("Length of accelerations and center_fractions must match.")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving masks to: {output_dir.resolve()}")

    # The fastMRI mask functions expect shape like (1, 1, height, width, 1) or similar
    # We mainly care about the width dimension for 1D Cartesian masks.
    mask_shape = (1, shape[0], shape[1], 1) # Example shape format (B, H, W, C=1 for mask)

    for acc, cf in zip(accelerations, center_fractions):
        # Corrected usage: Access class via the imported module
        mask_func = subsample.EquiSpacedMaskFunc(center_fractions=[cf], accelerations=[acc])
        # seed=0 ensures reproducibility for this specific mask type
        # offset=None lets the function determine offset, often 0 for equispaced
        mask_torch, _ = mask_func(mask_shape, seed=0, offset=None)

        # Squeeze unnecessary dimensions and convert to numpy
        # Mask shape from func is likely (1, H, W, 1) -> squeeze -> (H, W) -> numpy
        mask_np_squeezed = mask_torch.squeeze().numpy() # Shape should be (H, W)

        # For 1D Cartesian mask, all rows are the same. Take the profile from the first row.
        # Check if it's already 1D after squeeze (might happen if H=1 in mask_shape somehow)
        if mask_np_squeezed.ndim == 1:
             mask_1d_profile = mask_np_squeezed
        elif mask_np_squeezed.ndim == 2:
             mask_1d_profile = mask_np_squeezed[0, :] # Take the profile along the width dimension from the first row
        else:
             raise ValueError(f"Unexpected mask dimension after squeeze: {mask_np_squeezed.ndim}")

        # Ensure the final mask has the correct shape (e.g., (368,) for knee width)
        if mask_1d_profile.shape[0] != shape[1]:
             print(f"Warning: Generated mask width {mask_1d_profile.shape[0]} doesn't match expected shape width {shape[1]}. Adjusting.")
             # This might indicate an issue with shape interpretation, but let's try to save anyway

        # Construct the filename
        filename = output_dir / f"mask_{acc}x.npy"
        # Use the 1D profile directly
        print(f"Saving mask for {acc}x acceleration (shape: {mask_1d_profile.shape}) to {filename}")
        np.save(filename, mask_1d_profile.astype(np.float32)) # Save as float32 usually

    print("Mask generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Cartesian undersampling masks for fastMRI.")
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save the generated mask .npy files.'
    )
    parser.add_argument(
        '--accelerations',
        type=int,
        nargs='+',
        default=[4, 5], # Default accelerations often used
        help='List of acceleration factors (e.g., 4 5 8).'
    )
    parser.add_argument(
        '--center-fractions',
        type=float,
        nargs='+',
        default=[0.08, 0.04], # Default center fractions for 4x, 5x
        help='List of center fractions corresponding to accelerations (e.g., 0.08 0.04).'
    )
    parser.add_argument(
        '--kspace-width',
        type=int,
        default=368, # Common width for fastMRI knee data phase-encoding
        help='Width of the k-space (phase-encoding dimension).'
    )
    parser.add_argument(
        '--kspace-height',
        type=int,
        default=640, # Common height for fastMRI knee data frequency-encoding
        help='Height of the k-space (frequency-encoding dimension).'
    )

    args = parser.parse_args()

    kspace_shape = (args.kspace_height, args.kspace_width)
    output_path = pathlib.Path(args.output_dir)

    generate_cartesian_masks(output_path, args.accelerations, args.center_fractions, kspace_shape)

    print(f"\nReminder: Update the USMASK_PATH in your .env file to point to:")
    print(f"{output_path.resolve()}")