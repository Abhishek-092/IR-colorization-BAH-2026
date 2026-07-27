import os
import sys
import shutil
import glob
import logging
import csv
import numpy as np
import rasterio
from rasterio.enums import Resampling
from omegaconf import OmegaConf

# Insert root directory into sys.path to allow running python commands without setting PYTHONPATH
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from data_pipeline.pipeline_state import PipelineState

logger = logging.getLogger(__name__)

def process_product(product_dir, output_dir, manifest_writer, splits_map, force=False, state=None):
    """
    Processes a single product directory: downscales bands using rasterio box-averaging,
    slices into aligned patches using a sliding window with 50% overlap,
    rejects patches with > 10% nodata, and logs comprehensive metadata to the manifest.
    Uses PipelineState to skip regeneration if input data and pipeline state are unchanged.
    """
    product_name = os.path.basename(product_dir.rstrip('/\\'))
    
    if state is None:
        state = PipelineState()

    target_dir = os.path.join(output_dir, product_name)

    # Skip check if not force and dataset state is valid
    if not force and state.is_product_dataset_valid(product_dir, output_dir):
        logger.info(f"Skipping product '{product_name}': Prepared dataset patches already exist and are up to date.")
        patch_dirs = sorted(glob.glob(os.path.join(target_dir, f"{product_name}_patch_*")))
        for pdir in patch_dirs:
            p_name = os.path.basename(pdir)
            manifest_writer.writerow([
                p_name,
                product_name,
                "Landsat-8" if product_name.startswith("LC08") else "Landsat-9",
                product_name.split("_")[2][:3] if len(product_name.split("_")) >= 3 else "",
                product_name.split("_")[2][3:] if len(product_name.split("_")) >= 3 else "",
                product_name.split("_")[3] if len(product_name.split("_")) >= 4 else "",
                "", "", "", "", "", splits_map.get(product_name, "unknown"), "0.000000", "", ""
            ])
        return

    logger.info(f"Processing product: {product_name}")
    
    # Find band files (robust naming search)
    b2_files = glob.glob(os.path.join(product_dir, "*_B2.TIF")) + glob.glob(os.path.join(product_dir, "*_B2.tif")) + glob.glob(os.path.join(product_dir, "*_SR_B2.TIF")) + glob.glob(os.path.join(product_dir, "*_SR_B2.tif"))
    b3_files = glob.glob(os.path.join(product_dir, "*_B3.TIF")) + glob.glob(os.path.join(product_dir, "*_B3.tif")) + glob.glob(os.path.join(product_dir, "*_SR_B3.TIF")) + glob.glob(os.path.join(product_dir, "*_SR_B3.tif"))
    b4_files = glob.glob(os.path.join(product_dir, "*_B4.TIF")) + glob.glob(os.path.join(product_dir, "*_B4.tif")) + glob.glob(os.path.join(product_dir, "*_SR_B4.TIF")) + glob.glob(os.path.join(product_dir, "*_SR_B4.tif"))
    b10_files = glob.glob(os.path.join(product_dir, "*_B10.TIF")) + glob.glob(os.path.join(product_dir, "*_B10.tif")) + glob.glob(os.path.join(product_dir, "*_ST_B10.TIF")) + glob.glob(os.path.join(product_dir, "*_ST_B10.tif"))
    
    # Remove duplicates
    b2_files = list(set(b2_files))
    b3_files = list(set(b3_files))
    b4_files = list(set(b4_files))
    b10_files = list(set(b10_files))
    
    if not (b2_files and b3_files and b4_files and b10_files):
        logger.error(f"Missing required bands in product directory {product_dir}")
        return
        
    # Read metadata from one of the bands to get shapes
    with rasterio.open(b2_files[0]) as src:
        H, W = src.shape

    # Calculate downscaled dimensions at 100m (downscale factor exactly 10/3)
    target_H_100 = (int(round(H / (10.0 / 3.0))) // 2) * 2
    target_W_100 = (int(round(W / (10.0 / 3.0))) // 2) * 2
    
    # Enforce minimum patch dimensions
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
        
    rgb_100m = np.stack([r_100m, g_100m, b_100m], axis=0)
    prefix = product_name
    
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    
    patch_size_100 = 512
    patch_size_200 = 256
    stride_100 = 256
    
    y_starts = []
    y = 0
    while y + patch_size_100 <= target_H_100:
        y_starts.append(y)
        y += stride_100
    if y_starts and y_starts[-1] + patch_size_100 < target_H_100:
        y_starts.append(target_H_100 - patch_size_100)
    elif not y_starts:
        y_starts.append(0)
        
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
            rgb_patch = rgb_100m[:, y_start_100:y_start_100 + patch_size_100, x_start_100:x_start_100 + patch_size_100]
            tir_100_patch = tir_100m[y_start_100:y_start_100 + patch_size_100, x_start_100:x_start_100 + patch_size_100]
            
            y_start_200 = y_start_100 // 2
            x_start_200 = x_start_100 // 2
            tir_200_patch = tir_200m[y_start_200:y_start_200 + patch_size_200, x_start_200:x_start_200 + patch_size_200]
            
            nodata_mask = (tir_100_patch == 0) | (rgb_patch[0] == 0) | (rgb_patch[1] == 0) | (rgb_patch[2] == 0) | (tir_100_patch < -100) | (rgb_patch[0] < -100) | (rgb_patch[1] < -100) | (rgb_patch[2] < -100)
            nodata_fraction = float(np.sum(nodata_mask) / tir_100_patch.size)
            
            if nodata_fraction > 0.10:
                continue
                
            patch_name = f"{prefix}_patch_{count:03d}"
            sample_dir = os.path.join(target_dir, patch_name)
            os.makedirs(sample_dir, exist_ok=True)
            
            np.save(os.path.join(sample_dir, f"{patch_name}_rgb_100m.npy"), rgb_patch)
            np.save(os.path.join(sample_dir, f"{patch_name}_tir_100m.npy"), tir_100_patch)
            np.save(os.path.join(sample_dir, f"{patch_name}_tir_200m.npy"), tir_200_patch)
            
            parts = prefix.split("_")
            satellite = "Landsat-8" if parts[0] == "LC08" else "Landsat-9"
            path = parts[2][:3] if len(parts) >= 3 else ""
            row = parts[2][3:] if len(parts) >= 3 else ""
            acquisition_date = parts[3] if len(parts) >= 4 else ""
            
            scale_ratio = 10.0 / 3.0
            source_coords = f"x={x_start_100*scale_ratio:.1f},y={y_start_100*scale_ratio:.1f},w={512*scale_ratio:.1f},h={512*scale_ratio:.1f}"
            coords_100m = f"x={x_start_100},y={y_start_100},w=512,h=512"
            coords_200m = f"x={x_start_200},y={y_start_200},w=256,h=256"
            
            manifest_writer.writerow([
                patch_name,
                prefix,
                satellite,
                path,
                row,
                acquisition_date,
                os.path.basename(b2_files[0]),
                os.path.basename(b3_files[0]),
                os.path.basename(b4_files[0]),
                os.path.basename(b10_files[0]),
                source_coords,
                splits_map.get(prefix, "unknown"),
                f"{nodata_fraction:.6f}",
                coords_100m,
                coords_200m
            ])
            count += 1
            
    state.record_product_dataset(product_dir, count)
    logger.info(f"Generated {count} valid patches for {prefix} (nodata filtered)")

def prepare_all_datasets(input_dir="input", output_dir="output/patches", force=False):
    product_dirs = [d for d in glob.glob(os.path.join(input_dir, "*")) if os.path.isdir(d)]
    
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.csv")
    state = PipelineState()

    # Automatically purge patch folders & state records for any products deleted from input/
    valid_pids = [os.path.basename(d) for d in product_dirs]
    state.purge_orphaned_products(valid_pids, output_dir)
    
    if not product_dirs:
        logger.error(f"No products found in {input_dir}")
        return
    
    cfg_path = os.path.join(root_dir, "configs", "data.yaml")
    splits_map = {}
    if os.path.exists(cfg_path):
        cfg = OmegaConf.load(cfg_path)
        for sname in ["train", "val", "test"]:
            if hasattr(cfg, 'splits') and sname in cfg.splits:
                for pid in cfg.splits[sname]:
                    splits_map[pid] = sname
                    
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "patch_id", "product_id", "satellite", "path", "row", "acquisition_date",
            "source_B2", "source_B3", "source_B4", "source_B10",
            "source_pixel_window_coordinates", "split", "nodata_fraction",
            "coordinates_100m", "coordinates_200m"
        ])
        
        for product_dir in product_dirs:
            process_product(product_dir, output_dir, writer, splits_map, force=force, state=state)

    logger.info(f"Dataset preparation completed. Manifest saved to {manifest_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SUTRAM Dataset Preparation")
    parser.add_argument("--force", action="store_true", help="Force dataset patch generation even if patches already exist")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    prepare_all_datasets(force=args.force)
