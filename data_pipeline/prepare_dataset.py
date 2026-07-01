import os
import glob
import logging
import numpy as np
import rasterio
from rasterio.enums import Resampling

logger = logging.getLogger(__name__)

def downscale_band(src_data, scale_factor):
    """
    Downscales a single band using block-averaging (coarse graining) to maintain physical radiance.
    """
    H, W = src_data.shape
    new_h = int(round(H / scale_factor))
    new_w = int(round(W / scale_factor))
    
    # We use numpy to resize/average blocks
    h_pad = (scale_factor - (H % scale_factor)) % scale_factor
    w_pad = (scale_factor - (W % scale_factor)) % scale_factor
    
    if h_pad > 0 or w_pad > 0:
        src_data = np.pad(src_data, ((0, int(h_pad)), (0, int(w_pad))), mode='edge')
        
    H, W = src_data.shape
    block_h = int(round(scale_factor))
    block_w = int(round(scale_factor))
    
    # Reshape and compute mean
    reshaped = src_data[:new_h * block_h, :new_w * block_w].reshape(new_h, block_h, new_w, block_w)
    downscaled = reshaped.mean(axis=(1, 3))
    return downscaled

def process_product(product_dir, output_dir):
    """
    Processes a single product directory: reads bands, downscales, and slices into patches.
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
        
    # Load bands
    with rasterio.open(b2_files[0]) as src: b2 = src.read(1).astype(np.float32)
    with rasterio.open(b3_files[0]) as src: b3 = src.read(1).astype(np.float32)
    with rasterio.open(b4_files[0]) as src: b4 = src.read(1).astype(np.float32)
    with rasterio.open(b10_files[0]) as src: b10 = src.read(1).astype(np.float32)
    
    # Derive product prefix (e.g., LC09)
    prefix = product_name.split('_')[0]
    
    # Downscale RGB (30m to 100m -> scale 3.33)
    r_100m = downscale_band(b4, 3.33)
    g_100m = downscale_band(b3, 3.33)
    b_100m = downscale_band(b2, 3.33)
    rgb_100m = np.stack([r_100m, g_100m, b_100m], axis=0) # (3, H, W)
    
    # Downscale TIR (B10: 30m to 100m -> scale 3.33, 30m to 200m -> scale 6.67)
    tir_100m = downscale_band(b10, 3.33)
    tir_200m = downscale_band(b10, 6.67)
    
    H_100, W_100 = tir_100m.shape
    H_200, W_200 = tir_200m.shape
    
    # Slice into patches (512x512 at 100m scale, corresponding to 256x256 at 200m scale)
    patch_size_100 = 512
    patch_size_200 = 256
    
    y_steps = H_100 // patch_size_100
    x_steps = W_100 // patch_size_100
    
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
            
            # Ensure shape match
            if rgb_patch.shape == (3, patch_size_100, patch_size_100) and \
               tir_100_patch.shape == (patch_size_100, patch_size_100) and \
               tir_200_patch.shape == (patch_size_200, patch_size_200):
               
                sample_dir = os.path.join(output_dir, prefix, f"sample_{count:03d}")
                os.makedirs(sample_dir, exist_ok=True)
                
                np.save(os.path.join(sample_dir, "rgb_100m_512.npy"), rgb_patch)
                np.save(os.path.join(sample_dir, "tir_100m_512.npy"), tir_100_patch)
                np.save(os.path.join(sample_dir, "tir_200m.npy"), tir_200_patch)
                count += 1
                
    logger.info(f"Generated {count} patches for {prefix}")

def prepare_all_datasets(input_dir="input", output_dir="output/patches"):
    """
    Finds and processes all products in the input folder.
    """
    product_dirs = [d for d in glob.glob(os.path.join(input_dir, "*")) if os.path.isdir(d)]
    
    if not product_dirs:
        logger.error(f"No products found in {input_dir}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    for product_dir in product_dirs:
        process_product(product_dir, output_dir)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prepare_all_datasets()
