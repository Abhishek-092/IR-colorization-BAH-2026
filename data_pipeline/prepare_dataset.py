import os
import sys
import shutil

# Insert root directory into sys.path to allow running python commands without setting PYTHONPATH
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import glob
import logging
import numpy as np
import rasterio
import csv
from rasterio.enums import Resampling

logger = logging.getLogger(__name__)

def process_product(product_dir, output_dir, manifest_writer):
    """
    Processes a single product directory: downscales bands using rasterio box-averaging,
    slices into aligned patches using a sliding window with 50% overlap,
    rejects patches with > 10% nodata, and logs metadata to the manifest.
    """
    product_name = os.path.basename(product_dir.rstrip('/\\'))
    logger.info(f"Processing product: {product_name}")
    
    # Find band files
    b2_files = glob.glob(os.path.join(product_dir, "*_B2.TIF")) + glob.glob(os.path.join(product_dir, "*_B2.tif"))
    b3_files = glob.glob(os.path.join(product_dir, "*_B3.TIF")) + glob.glob(os.path.join(product_dir, "*_B3.tif"))
    b4_files = glob.glob(os.path.join(product_dir, "*_B4.TIF")) + glob.glob(os.path.join(product_dir, "*_B4.tif"))
    b10_files = glob.glob(os.path.join(product_dir, "*_B10.TIF")) + glob.glob(os.path.join(product_dir, "*_B10.tif"))
    
    if not (b2_files and b3_files and b4_files and b10_files):
        logger.error(f"Missing required bands in product directory {product_dir}")
        return
        
    # Read metadata from one of the bands to get shapes
    with rasterio.open(b2_files[0]) as src:
        H, W = src.shape

    # Calculate downscaled dimensions at 100m (downscale factor ~ 3.33)
    target_H_100 = int(round(H / 3.33))
    target_W_100 = int(round(W / 3.33))
    
    # Enforce minimum patch dimensions
    target_H_100 = max(512, target_H_100)
    target_W_100 = max(512, target_W_100)
    
    target_H_200 = target_H_100 // 2
    target_W_200 = target_W_100 // 2
    
    # Read and resample bands (preserving radiometric precision as float32)
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
    prefix = product_name
    
    # Clear out any stale patches to prevent split contamination
    target_dir = os.path.join(output_dir, prefix)
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    
    patch_size_100 = 512
    patch_size_200 = 256
    
    # Slide with 50% overlap
    stride_100 = 256
    
    # Generate y_starts
    y_starts = []
    y = 0
    while y + patch_size_100 <= target_H_100:
        y_starts.append(y)
        y += stride_100
    if y_starts and y_starts[-1] + patch_size_100 < target_H_100:
        y_starts.append(target_H_100 - patch_size_100)
    elif not y_starts:
        y_starts.append(0)
        
    # Generate x_starts
    x_starts = []
    x = 0
    while x + patch_size_100 <= target_W_100:
        x_starts.append(x)
        x += stride_100
    if x_starts and x_starts[-1] + patch_size_100 < target_W_100:
        x_starts.append(target_W_100 - patch_size_100)
    elif not x_starts:
        x_starts.append(0)
        
    count = 0
    for y_start_100 in y_starts:
        for x_start_100 in x_starts:
            # Crop 100m patches
            rgb_patch = rgb_100m[:, y_start_100:y_start_100 + patch_size_100, x_start_100:x_start_100 + patch_size_100]
            tir_100_patch = tir_100m[y_start_100:y_start_100 + patch_size_100, x_start_100:x_start_100 + patch_size_100]
            
            # Crop 200m patches
            y_start_200 = y_start_100 // 2
            x_start_200 = x_start_100 // 2
            tir_200_patch = tir_200m[y_start_200:y_start_200 + patch_size_200, x_start_200:x_start_200 + patch_size_200]
            
            # Nodata check (pixels exactly equal to 0)
            nodata_count = np.sum(tir_100_patch == 0)
            nodata_fraction = float(nodata_count / tir_100_patch.size)
            
            # Filter threshold: reject patches with > 10% nodata
            if nodata_fraction > 0.10:
                continue
                
            # Save NPY files
            sample_dir = os.path.join(target_dir, f"sample_{count:03d}")
            os.makedirs(sample_dir, exist_ok=True)
            
            np.save(os.path.join(sample_dir, "rgb_100m_512.npy"), rgb_patch)
            np.save(os.path.join(sample_dir, "tir_100m_512.npy"), tir_100_patch)
            np.save(os.path.join(sample_dir, "tir_200m.npy"), tir_200_patch)
            
            # Write to manifest
            parts = prefix.split("_")
            satellite = "Landsat-8" if parts[0] == "LC08" else "Landsat-9"
            path = parts[2][:3]
            row = parts[2][3:]
            
            manifest_writer.writerow([
                f"{prefix}_sample_{count:03d}",
                prefix,
                satellite,
                path,
                row,
                x_start_100,
                y_start_100,
                f"{H}x{W}",
                f"{nodata_fraction:.6f}"
            ])
            count += 1
            
    logger.info(f"Generated {count} valid patches for {prefix} (nodata filtered)")

def is_product_dirty(product_dir, output_dir):
    """
    Checks if the patches for a product are missing or if the raw TIF files are newer.
    """
    product_name = os.path.basename(product_dir.rstrip('/\\'))
    target_dir = os.path.join(output_dir, product_name)
    
    if not os.path.exists(target_dir):
        return True
        
    npy_files = glob.glob(os.path.join(target_dir, "**", "*.npy"), recursive=True)
    if not npy_files:
        return True
        
    min_npy_mtime = min(os.path.getmtime(f) for f in npy_files)
    
    tif_files = glob.glob(os.path.join(product_dir, "*.TIF")) + glob.glob(os.path.join(product_dir, "*.tif"))
    if not tif_files:
        return False
        
    max_tif_mtime = max(os.path.getmtime(f) for f in tif_files)
    return max_tif_mtime > min_npy_mtime

def prepare_all_datasets(input_dir="input", output_dir="output/patches", force=False):
    """
    Finds and processes all products in the input folder. Auto-detects updates or new directories.
    Generates a unified manifest.csv at the end.
    """
    product_dirs = [d for d in glob.glob(os.path.join(input_dir, "*")) if os.path.isdir(d)]
    
    if not product_dirs:
        logger.error(f"No products found in {input_dir}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.csv")
    
    # We will rewrite the manifest
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patch_id", "product_id", "satellite", "path", "row", "x_offset", "y_offset", "source_shape", "nodata_fraction"])
        
        for product_dir in product_dirs:
            product_name = os.path.basename(product_dir.rstrip('/\\'))
            # Force regeneration is handled by force=True
            process_product(product_dir, output_dir, writer)

    logger.info(f"Dataset preparation completed. Manifest saved to {manifest_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SUTRAM Dataset Preparation")
    parser.add_argument("--force", action="store_true", help="Force dataset patch generation even if patches already exist")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    prepare_all_datasets(force=args.force)
