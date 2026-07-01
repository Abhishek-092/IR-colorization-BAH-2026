import os
import glob
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Lazy import rasterio to avoid import failures before installation completes
def get_rasterio():
    import rasterio
    return rasterio

def validate_input_product(product_dir):
    """
    Validates a raw Landsat-9 product directory before ingestion.
    Checks:
    - Required bands presence: B2, B3, B4, B10
    - GeoTIFF validity and readable shapes
    - Coordinate Reference System (CRS) consistency
    - Data corruption (extreme values, NaNs)
    """
    if not os.path.isdir(product_dir):
        logger.error(f"Product directory does not exist: {product_dir}")
        return False, {"error": "Directory not found"}

    # Find band files
    files = glob.glob(os.path.join(product_dir, "*"))
    band_paths = {}
    for suffix in ["_B2", "_B3", "_B4", "_B10"]:
        matching = [f for f in files if suffix in os.path.basename(f) and f.lower().endswith(('.tif', '.tiff'))]
        if not matching:
            logger.error(f"Missing required band with suffix: {suffix}")
            return False, {"error": f"Missing required band {suffix}"}
        band_paths[suffix] = matching[0]

    rasterio = get_rasterio()
    crs_set = set()
    shapes = {}
    corruption_detected = False
    details = {}

    try:
        for suffix, path in band_paths.items():
            with rasterio.open(path) as src:
                # Read metadata
                crs = src.crs.to_string() if src.crs else "None"
                crs_set.add(crs)
                shapes[suffix] = src.shape
                
                # Check a sample/profile for NaNs/infs
                band_data = src.read(1)
                nan_count = np.isnan(band_data).sum()
                inf_count = np.isinf(band_data).sum()
                
                if nan_count > 0 or inf_count > 0:
                    logger.warning(f"Corrupted pixels detected in {suffix}: NaNs={nan_count}, Infs={inf_count}")
                    corruption_detected = True

                details[suffix] = {
                    "shape": src.shape,
                    "crs": crs,
                    "dtype": str(src.dtypes[0]),
                    "nans": int(nan_count),
                    "infs": int(inf_count)
                }

    except Exception as e:
        logger.error(f"Failed to read GeoTIFF: {e}")
        return False, {"error": f"Failed to read GeoTIFF: {str(e)}"}

    # Validation checks
    # 1. CRS Consistency
    if len(crs_set) > 1:
        logger.error(f"Inconsistent CRS coordinates found: {crs_set}")
        return False, {"error": "CRS mismatch", "details": details}

    # 2. Shape compatibility: B2, B3, B4 (RGB) must be identical shape
    rgb_shapes = {shapes["_B2"], shapes["_B3"], shapes["_B4"]}
    if len(rgb_shapes) > 1:
        logger.error(f"Optical RGB bands shape mismatch: {shapes}")
        return False, {"error": "RGB shape mismatch", "details": details}

    if corruption_detected:
        logger.warning("Ingestion completed with warnings: corruption detected")
        return True, {"status": "warning", "message": "Corrupted values (NaNs/Infs) found in bands", "details": details}

    logger.info(f"Successfully validated product {os.path.basename(product_dir)}")
    return True, {"status": "success", "details": details}
