import os
import sys

# Insert root directory into sys.path to allow running python commands without setting PYTHONPATH
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import glob
import logging
import numpy as np
import rasterio
from rasterio.enums import Resampling

logger = logging.getLogger(__name__)

def process_product(product_dir, output_dir):
    """
    Processes a single product directory: downscales bands using rasterio box-averaging,
    and slices into aligned patches.
    """
    product_name = os.path.basename(product_dir.rstrip('/\\'))
    logger.info(f"Processing product: {product_name}")
    
    # Find band files
    b2_files = glob.glob(os.path.join(product_dir, "*_B2.TIF"))
    b3_files = glob.glob(os.path.join(product_dir, "*_B3.TIF"))
    b4_files = glob.glob(os.path.join(product_dir, "*_B4.TIF"))
    b10_files = glob.glob(os.path.join(product_dir, "*_B10.TIF"))
    
    if not (b2_files and b3_files and b4_files and b10_files):
        logger.error(f"Missing required bands in product directory {product_dir}")
        return
        
    # Read metadata from one of the bands to get shapes
    with rasterio.open(b2_files[0]) as src:
        H, W = src.shape

    # Calculate target shapes as multiples of patch size (512 at 100m, 256 at 200m)
    # The downscale factor from 30m to 100m is 3.33, so target shape H_100 is approx H / 3.33
    target_H_100 = (int(round(H / 3.33)) // 512) * 512
    target_W_100 = (int(round(W / 3.33)) // 512) * 512
    
    # Fallback to at least 512 if image is too small
    target_H_100 = max(512, target_H_100)
    target_W_100 = max(512, target_W_100)
    
    target_H_200 = target_H_100 // 2
    target_W_200 = target_W_100 // 2
    
    # Read and resample bands
    with rasterio.open(b4_files[0]) as src:
        r_100m = src.read(1, out_shape=(target_H_100, target_W_100), resampling=Resampling.average).astype(np.float32)
    with rasterio.open(b3_files[0]) as src:
        g_100m = src.read(1, out_shape=(target_H_100, target_W_100), resampling=Resampling.average).astype(np.float32)
    with rasterio.open(b2_files[0]) as src:
        b_100m = src.read(1, out_shape=(target_H_100, target_W_100), resampling=Resampling.average).astype(np.float32)
    with rasterio.open(b10_files[0]) as src:
        tir_100m = src.read(1, out_shape=(target_H_100, target_W_100), resampling=Resampling.average).astype(np.float32)
        tir_200m = src.read(1, out_shape=(target_H_200, target_W_200), resampling=Resampling.average).astype(np.float32)
        
    rgb_100m = np.stack([r_100m, g_100m, b_100m], axis=0) # (3, H, W)
    prefix = product_name.split('_')[0]
    
    patch_size_100 = 512
    patch_size_200 = 256
    
    y_steps = target_H_100 // patch_size_100
    x_steps = target_W_100 // patch_size_100
    
    count = 0
    for y in range(y_steps):
        for x in range(x_steps):
            # Crop 100m patches
            y_start_100 = y * patch_size_100
            x_start_100 = x * patch_size_100
            
            rgb_patch = rgb_100m[:, y_start_100:y_start_100 + patch_size_100, x_start_100:x_start_100 + patch_size_100]
            tir_100_patch = tir_100m[y_start_100:y_start_100 + patch_size_100, x_start_100:x_start_100 + patch_size_100]
            
            # Crop 200m patches
            y_start_200 = y * patch_size_200
            x_start_200 = x * patch_size_200
            tir_200_patch = tir_200m[y_start_200:y_start_200 + patch_size_200, x_start_200:x_start_200 + patch_size_200]
            
            # Save
            sample_dir = os.path.join(output_dir, prefix, f"sample_{count:03d}")
            os.makedirs(sample_dir, exist_ok=True)
            
            np.save(os.path.join(sample_dir, "rgb_100m_512.npy"), rgb_patch)
            np.save(os.path.join(sample_dir, "tir_100m_512.npy"), tir_100_patch)
            np.save(os.path.join(sample_dir, "tir_200m.npy"), tir_200_patch)
            count += 1
                
    logger.info(f"Generated {count} aligned patches for {prefix}")

def prepare_all_datasets(input_dir="input", output_dir="output/patches", force=False):
    """
    Finds and processes all products in the input folder. Skips if patches already exist unless force=True.
    """
    if os.path.exists(output_dir) and not force:
        existing_npy = glob.glob(os.path.join(output_dir, "**", "*.npy"), recursive=True)
        if len(existing_npy) > 0:
            logger.info("Aligned dataset patches already exist. Skipping patch generation to avoid repeated generation. Use --force to regenerate.")
            return

    product_dirs = [d for d in glob.glob(os.path.join(input_dir, "*")) if os.path.isdir(d)]
    
    if not product_dirs:
        logger.error(f"No products found in {input_dir}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    for product_dir in product_dirs:
        process_product(product_dir, output_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SUTRAM Dataset Preparation")
    parser.add_argument("--force", action="store_true", help="Force dataset patch generation even if patches already exist")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    prepare_all_datasets(force=args.force)
