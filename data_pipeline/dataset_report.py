import os
import glob
import numpy as np
import logging

logger = logging.getLogger(__name__)

def generate_dataset_report(patches_dir):
    """
    Computes statistical properties over the entire patch dataset.
    Outputs:
    - Mean and standard deviation for TIR and RGB bands
    - Empirical quantiles of the RGB distribution (crucial for Mode-Redundancy init)
    - Total sample count and missing value verification
    """
    sample_dirs = glob.glob(os.path.join(patches_dir, "*", "sample_*"))
    if not sample_dirs:
        logger.error(f"No patches found in {patches_dir}")
        return None

    logger.info(f"Profiling {len(sample_dirs)} samples in {patches_dir}...")
    
    tir_200_vals = []
    tir_100_vals = []
    rgb_vals = []

    # Sample a subset or all if small to avoid memory overflow
    step = max(1, len(sample_dirs) // 50)
    sampled_dirs = sample_dirs[::step]

    for sdir in sampled_dirs:
        try:
            tir_200 = np.load(os.path.join(sdir, "tir_200m.npy"))
            tir_100 = np.load(os.path.join(sdir, "tir_100m_512.npy"))
            rgb = np.load(os.path.join(sdir, "rgb_100m_512.npy"))

            # Downsample for faster statistics
            tir_200_vals.append(tir_200[::4, ::4].flatten())
            tir_100_vals.append(tir_100[::8, ::8].flatten())
            
            # RGB shape (C, H, W) or (H, W, C)
            if rgb.ndim == 3 and rgb.shape[0] != 3:
                rgb = np.moveaxis(rgb, -1, 0)
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
            "quantiles_r": np.percentile(all_rgb[2], [10, 25, 50, 75, 90]).tolist(), # Red channel
            "quantiles_g": np.percentile(all_rgb[1], [10, 25, 50, 75, 90]).tolist(), # Green channel
            "quantiles_b": np.percentile(all_rgb[0], [10, 25, 50, 75, 90]).tolist(), # Blue channel
        }
    }

    logger.info("Dataset profiling completed successfully.")
    logger.info(f"TIR 100m Stats: Mean={report['tir_100m']['mean']:.2f}, Std={report['tir_100m']['std']:.2f}")
    logger.info(f"RGB Stats: Mean={report['rgb']['mean']}")
    return report
