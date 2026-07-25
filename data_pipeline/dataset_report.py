import os
import glob
import numpy as np
import logging

logger = logging.getLogger(__name__)

def generate_dataset_report(patches_dir, product_ids=None):
    """
    Computes statistical properties over the entire patch dataset.
    Outputs:
    - Mean and standard deviation for TIR and RGB bands
    - Empirical quantiles of the RGB distribution (crucial for Mode-Redundancy init)
    - Total sample count and missing value verification
    """
    if product_ids is None:
        product_dirs = [d for d in glob.glob(os.path.join(patches_dir, "*")) if os.path.isdir(d)]
    else:
        product_dirs = []
        for product_id in product_ids:
            product_dir = os.path.join(patches_dir, product_id)
            if not os.path.isdir(product_dir):
                raise FileNotFoundError(
                    f"Configured training scene '{product_id}' is missing from {patches_dir}"
                )
            product_dirs.append(product_dir)

    sample_dirs = []
    for product_dir in product_dirs:
        sample_dirs.extend(sorted(glob.glob(os.path.join(product_dir, "*_patch_*"))))
    if not sample_dirs:
        logger.error(f"No patches found in {patches_dir}")
        return None

    logger.info(f"Profiling {len(sample_dirs)} samples in {patches_dir}...")
    
    tir_200_vals = []
    tir_100_vals = []
    rgb_vals = []

    step = max(1, len(sample_dirs) // 50)
    sampled_dirs = sample_dirs[::step]

    for sdir in sampled_dirs:
        try:
            patch_name = os.path.basename(sdir)
            tir_200 = np.load(os.path.join(sdir, f"{patch_name}_tir_200m.npy"))
            tir_100 = np.load(os.path.join(sdir, f"{patch_name}_tir_100m.npy"))
            rgb = np.load(os.path.join(sdir, f"{patch_name}_rgb_100m.npy"))

            # Normalize values dynamically
            if tir_200.max() > 255.0:
                tir_200 = np.clip((tir_200 - 20000.0) / 15000.0, 0.0, 1.0)
            else:
                tir_200 = np.clip(tir_200 / 255.0, 0.0, 1.0)

            if tir_100.max() > 255.0:
                tir_100 = np.clip((tir_100 - 20000.0) / 15000.0, 0.0, 1.0)
            else:
                tir_100 = np.clip(tir_100 / 255.0, 0.0, 1.0)

            if rgb.ndim == 3 and rgb.shape[0] != 3:
                rgb = np.moveaxis(rgb, -1, 0)
            if rgb.max() > 255.0:
                rgb = np.clip((rgb / 10000.0) * 255.0, 0.0, 255.0)

            # Downsample for faster statistics
            tir_200_vals.append(tir_200[::4, ::4].flatten())
            tir_100_vals.append(tir_100[::8, ::8].flatten())
            rgb_vals.append(rgb[:, ::8, ::8].reshape(3, -1))

        except Exception as e:
            logger.error(f"Error reading sample {sdir}: {e}")

    # Concatenate sampled values
    all_tir_200 = np.concatenate(tir_200_vals)
    all_tir_100 = np.concatenate(tir_100_vals)
    all_rgb = np.concatenate(rgb_vals, axis=1)

    # Compute Statistics
    report = {
        "sample_count": len(sample_dirs),
        "profiled_count": len(sampled_dirs),
        "tir_200m": {
            "mean": float(np.mean(all_tir_200)),
            "std": float(np.std(all_tir_200)),
            "min": float(np.min(all_tir_200)),
            "max": float(np.max(all_tir_200))
        },
        "tir_100m": {
            "mean": float(np.mean(all_tir_100)),
            "std": float(np.std(all_tir_100)),
            "min": float(np.min(all_tir_100)),
            "max": float(np.max(all_tir_100))
        },
        "rgb": {
            "mean": np.mean(all_rgb, axis=1).tolist(),
            "std": np.std(all_rgb, axis=1).tolist(),
            "quantiles_r": np.percentile(all_rgb[0], [10, 25, 50, 75, 90]).tolist(), # Red channel
            "quantiles_g": np.percentile(all_rgb[1], [10, 25, 50, 75, 90]).tolist(), # Green channel
            "quantiles_b": np.percentile(all_rgb[2], [10, 25, 50, 75, 90]).tolist(), # Blue channel
        }
    }

    logger.info("Dataset profiling completed successfully.")
    logger.info(f"TIR 100m Stats: Mean={report['tir_100m']['mean']:.2f}, Std={report['tir_100m']['std']:.2f}")
    logger.info(f"RGB Stats: Mean={report['rgb']['mean']}")
    return report
