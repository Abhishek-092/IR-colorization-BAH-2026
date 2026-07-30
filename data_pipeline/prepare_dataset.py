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

from sutram.calibration.autocalibrate import (
    fit_thermal_calibration, apply_thermal_calibration, calibrate_rgb_to_display,
)

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
        
    rgb_100m_raw = np.stack([r_100m, g_100m, b_100m], axis=0) # (3, H, W)

    # --- Per-scene calibration (single source of truth: sutram.calibration) ---
    # Nodata masks must come from the RAW values (L2SP fill = 0) BEFORE
    # calibration, because calibration maps 0 to a valid temperature/colour.
    nodata_100 = (tir_100m == 0) | np.any(rgb_100m_raw == 0, axis=0)
    # Fit the thermal mapping ONCE on the full-scene 100 m band and apply the
    # same mapping to both resolutions, so every patch of this scene — and the
    # SR input/target pair — share one consistent DN -> Kelvin transform.
    cal = fit_thermal_calibration(tir_100m)
    tir_100m = apply_thermal_calibration(tir_100m, cal)   # Kelvin
    tir_200m = apply_thermal_calibration(tir_200m, cal)   # Kelvin
    # Optical -> display RGB 0-255 (auto-detects browse / synthetic / L2 SR).
    rgb_100m = calibrate_rgb_to_display(rgb_100m_raw)
    # Namespace patches by the *full* product name. Using only the sensor prefix
    # (e.g. "LC09") makes every product of the same sensor write into the same
    # folder, and because `count` resets to 0 per product, later products silently
    # overwrite earlier ones' patches. The full name keeps every product distinct
    # and lets the config split by product for a genuine train/val holdout.
    prefix = product_name

    patch_size_100 = 512
    patch_size_200 = 256
    # Overlapping extraction (stride < patch) yields several times more training
    # patches from the same scenes, and quality gates drop the ones that would
    # teach the model the wrong thing (nodata borders, cloud-covered tiles).
    stride_100 = 256
    stride_200 = stride_100 // 2

    rgb_max = float(rgb_100m.max()) if rgb_100m.max() > 0 else 1.0

    def _reject(rgb_patch, nodata_patch):
        """Return (reject: bool, reason: str) for a candidate patch."""
        # 1. Nodata (from the RAW fill mask). Too much fill -> border tile.
        if nodata_patch.mean() > 0.05:
            return True, "nodata"
        # 2. Cloud: bright in every optical band (scale-agnostic vs the scene max).
        #    Clouds are radiometrically uninformative for colour/thermal learning.
        bright_all = np.all(rgb_patch > 0.80 * rgb_max, axis=0)
        if bright_all.mean() > 0.50:
            return True, "cloud"
        return False, ""

    count = 0
    rej = {"nodata": 0, "cloud": 0}
    max_y = target_H_100 - patch_size_100
    max_x = target_W_100 - patch_size_100
    ys = list(range(0, max_y + 1, stride_100)) or [0]
    xs = list(range(0, max_x + 1, stride_100)) or [0]

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            rgb_patch = rgb_100m[:, y:y + patch_size_100, x:x + patch_size_100]
            tir_100_patch = tir_100m[y:y + patch_size_100, x:x + patch_size_100]
            nodata_patch = nodata_100[y:y + patch_size_100, x:x + patch_size_100]
            y2, x2 = i * stride_200, j * stride_200
            tir_200_patch = tir_200m[y2:y2 + patch_size_200, x2:x2 + patch_size_200]

            # Shape guards (edge of scene)
            if (rgb_patch.shape[-2:] != (patch_size_100, patch_size_100)
                    or tir_100_patch.shape != (patch_size_100, patch_size_100)
                    or tir_200_patch.shape != (patch_size_200, patch_size_200)):
                continue

            reject, reason = _reject(rgb_patch, nodata_patch)
            if reject:
                rej[reason] += 1
                continue

            sample_dir = os.path.join(output_dir, prefix, f"sample_{count:03d}")
            os.makedirs(sample_dir, exist_ok=True)
            np.save(os.path.join(sample_dir, "rgb_100m_512.npy"), rgb_patch)
            np.save(os.path.join(sample_dir, "tir_100m_512.npy"), tir_100_patch)
            np.save(os.path.join(sample_dir, "tir_200m.npy"), tir_200_patch)
            count += 1

    logger.info(f"Generated {count} patches for {prefix} "
                f"(rejected: {rej['nodata']} nodata, {rej['cloud']} cloud)")

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
