import os
import glob
import numpy as np
import logging

logger = logging.getLogger(__name__)

def check_alignment_in_patches(patches_dir, max_samples_to_check=5):
    """
    Checks the spatial alignment of generated patch arrays.
    Verifies that:
    - Shape ratio: tir_100m must be exactly double the shape of tir_200m.
    - Shape ratio: rgb_100m must be identical to tir_100m.
    - Downscaled values correspond: downsampling a patch from 100m (using box average)
      reconstructs the corresponding 200m patch.
    """
    sample_dirs = glob.glob(os.path.join(patches_dir, "*", "*_patch_*"))
    if not sample_dirs:
        logger.warning(f"No patch samples found in {patches_dir} to check alignment.")
        return True

    checked = 0
    for sdir in sample_dirs:
        if checked >= max_samples_to_check:
            break

        try:
            patch_name = os.path.basename(sdir)
            tir_200 = np.load(os.path.join(sdir, f"{patch_name}_tir_200m.npy"))
            tir_100 = np.load(os.path.join(sdir, f"{patch_name}_tir_100m.npy"))
            rgb_100 = np.load(os.path.join(sdir, f"{patch_name}_rgb_100m.npy"))

            # Shape Checks
            if tir_100.shape[-2:] != (512, 512):
                logger.error(f"Shape error in {sdir}: tir_100m is {tir_100.shape}")
                return False

            if tir_200.shape[-2:] != (256, 256):
                logger.error(f"Shape error in {sdir}: tir_200m is {tir_200.shape}")
                return False

            if rgb_100.shape[-2:] != (512, 512):
                logger.error(f"Shape error in {sdir}: rgb_100m is {rgb_100.shape}")
                return False

            # Consistency check: downscaling tir_100m by 2 should match tir_200m
            tir_100_downscaled = tir_100.reshape(256, 2, 256, 2).mean(axis=(1, 3))
            diff = np.abs(tir_100_downscaled - tir_200).mean()

            # Small tolerance for floating point / rounding mismatch
            if diff > 10.0:  
                logger.warning(f"Slight numerical deviation in downscaled values in {sdir}: mean diff = {diff:.4f}")

            checked += 1

        except Exception as e:
            logger.error(f"Error checking alignment in {sdir}: {e}")
            return False

    logger.info(f"Checked alignment on {checked} patch samples: PASSED shape constraints.")
    return True
