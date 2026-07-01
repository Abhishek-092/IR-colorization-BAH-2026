import os
import logging
from utils.file_utils import find_file

logger = logging.getLogger(__name__)

def check_baseline_patches(patches_dir, product_id):
    """
    Verifies if patch generation has completed correctly for a product.
    Checks that the expected .npy and .png outputs exist in the output patches folder.
    """
    product_patches_path = os.path.join(patches_dir, product_id)
    if not os.path.isdir(product_patches_path):
        logger.error(f"Patches folder for {product_id} not found in {patches_dir}")
        return False

    samples = [d for d in os.listdir(product_patches_path) if os.path.isdir(os.path.join(product_patches_path, d))]
    if not samples:
        logger.error(f"No patch samples found in {product_patches_path}")
        return False

    # Check validity of one sample
    test_sample = os.path.join(product_patches_path, samples[0])
    required_files = [
        "rgb_100m_512.npy",
        "rgb_100m_512.png",
        "tir_100m_512.npy",
        "tir_100m_512.png",
        "tir_200m.npy",
        "tir_200m.png"
    ]
    for rfile in required_files:
        if not os.path.isfile(os.path.join(test_sample, rfile)):
            logger.error(f"Missing required patch file {rfile} in sample {samples[0]}")
            return False

    logger.info(f"Verified {len(samples)} baseline patches successfully for {product_id}")
    return True
