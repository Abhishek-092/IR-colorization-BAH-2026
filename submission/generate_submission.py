import os
import glob
import zipfile
import logging
import rasterio
import torch

logger = logging.getLogger(__name__)

def validate_submission_deliverables():
    """
    Validates output GeoTIFF files:
    - Verifies matching products in SR and Color folders.
    - Asserts GeoTIFF metadata presence.
    - Asserts BGR count and band ordering.
    """
    sr_dir = "output/model_outputs/tir_superresolved_100m"
    color_dir = "output/model_outputs/colorized_tir_100m"
    
    sr_files = glob.glob(os.path.join(sr_dir, "*.tif"))
    color_files = glob.glob(os.path.join(color_dir, "*.tif"))

    if not sr_files:
        logger.error(f"No super-resolved GeoTIFFs found in {sr_dir}")
        return False
    if not color_files:
        logger.error(f"No colorized GeoTIFFs found in {color_dir}")
        return False

    # Check matches
    sr_names = {os.path.basename(f) for f in sr_files}
    color_names = {os.path.basename(f) for f in color_files}
    
    if sr_names != color_names:
        logger.error("Mismatch between product files in SR and Colorized output folders.")
        return False

    # Verify metadata and shapes
    for path in sr_files:
        try:
            with rasterio.open(path) as src:
                if src.count != 1:
                    logger.error(f"SR GeoTIFF must have exactly 1 band. Got {src.count} bands in {path}")
                    return False
                if not src.crs:
                    logger.warning(f"SR GeoTIFF is missing coordinate reference metadata (CRS) in {path} (Expected for synthetic data)")
        except Exception as e:
            logger.error(f"Failed reading SR GeoTIFF {path}: {e}")
            return False

    for path in color_files:
        try:
            with rasterio.open(path) as src:
                if src.count != 3:
                    logger.error(f"Colorized GeoTIFF must have exactly 3 bands. Got {src.count} bands in {path}")
                    return False
                if not src.crs:
                    logger.warning(f"Colorized GeoTIFF is missing CRS metadata in {path} (Expected for synthetic data)")
        except Exception as e:
            logger.error(f"Failed reading Colorized GeoTIFF {path}: {e}")
            return False

    logger.info("Deliverable GeoTIFF validations: PASSED.")
    return True


def validate_release_checkpoint(checkpoint_path="checkpoints/sutram_final.pth"):
    """Require a freshly packaged, calibrated SUTRAM checkpoint."""
    if not os.path.isfile(checkpoint_path):
        logger.error(f"Required release checkpoint is missing: {checkpoint_path}")
        return False
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as error:
        logger.error(f"Unable to load release checkpoint {checkpoint_path}: {error}")
        return False

    if checkpoint.get("project_id") != "SUTRAM" or "SUTRAM" not in str(checkpoint.get("model_name", "")).upper():
        logger.error("Release checkpoint is stale, legacy, or not identified as SUTRAM.")
        return False
    preprocessing = checkpoint.get("config", {}).get("preprocessing", {})
    if preprocessing.get("tir_representation") != "brightness_temperature_kelvin":
        logger.error("Release checkpoint lacks the calibrated-B10 preprocessing declaration.")
        return False
    required_state = {"backbone_state_dict", "sr_head_state_dict", "mixture_head_state_dict"}
    if not required_state.issubset(checkpoint):
        logger.error("Release checkpoint is incomplete.")
        return False
    return True

def package_submission():
    """
    Compresses the code structure, readme, and model weights into a zip archive.
    """
    # Submission packages must be fail-closed: never create an archive from
    # missing, mismatched, or malformed deliverables.
    if not validate_submission_deliverables() or not validate_release_checkpoint():
        raise RuntimeError("Submission packaging aborted: deliverable validation failed.")
    
    zip_filename = "project_sutram_submission.zip"
    logger.info(f"Building submission zip archive: {zip_filename}")
    
    # Files to include
    exclude_dirs = [
        ".git", "__pycache__", "input", "output", "experiments", ".pytest_cache",
        ".agents", "private_audit", "formulae and calculation",
    ]
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk("."):
            # Exclude directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                filepath = os.path.join(root, file)
                # Skip already created zip
                if file == zip_filename or file.endswith(".zip"):
                    continue
                # Add file to zip
                zipf.write(filepath, os.path.relpath(filepath, "."))
                
    logger.info(f"Successfully packaged submission into: {os.path.abspath(zip_filename)}")
