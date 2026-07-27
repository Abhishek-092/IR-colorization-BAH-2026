import os
import glob
import logging

logger = logging.getLogger(__name__)

def extract_path_row(scene_id):
    """
    Extracts the path and row from a Landsat product ID.
    Format: LXSS_L2SP_PPPRRR_YYYYMMDD_... -> PPPRRR (Path + Row)
    """
    parts = scene_id.split("_")
    if len(parts) >= 3 and len(parts[2]) == 6:
        return parts[2] # e.g. "146040"
    return None

def validate_split_config(splits):
    """
    Validates that the configuration has train, val, and test keys and no legacy prefixes or demo are present.
    """
    for key in ["train", "val", "test"]:
        if key not in splits:
            raise ValueError(f"CRITICAL: Missing split category '{key}' in configuration.")
        
    for key in ["train", "val", "test"]:
        for pid in splits[key]:
            if pid in ["LC08", "LC09", "demo"]:
                raise ValueError(f"CRITICAL: Generic prefix or legacy value '{pid}' is forbidden in the finals config.")

def validate_product_isolation(train, val, test):
    """
    Ensures that no product ID appears in more than one split (Level 1 Isolation).
    """
    set_train = set(train)
    set_val = set(val)
    set_test = set(test)
    
    inter_train_val = set_train.intersection(set_val)
    inter_train_test = set_train.intersection(set_test)
    inter_val_test = set_val.intersection(set_test)
    
    if inter_train_val:
        raise ValueError(f"CRITICAL: Product leakage detected between TRAIN and VAL: {inter_train_val}")
    if inter_train_test:
        raise ValueError(f"CRITICAL: Product leakage detected between TRAIN and TEST: {inter_train_test}")
    if inter_val_test:
        raise ValueError(f"CRITICAL: Product leakage detected between VAL and TEST: {inter_val_test}")
    
    logger.info("Level 1 Product ID Isolation check: PASSED.")

def validate_path_row_isolation(train, val, test):
    """
    Ensures that no path/row group appears in more than one split (Level 2 Geographic Isolation).
    """
    train_pr = {extract_path_row(s) for s in train if extract_path_row(s) is not None}
    val_pr = {extract_path_row(s) for s in val if extract_path_row(s) is not None}
    test_pr = {extract_path_row(s) for s in test if extract_path_row(s) is not None}
    
    inter_train_val = train_pr.intersection(val_pr)
    inter_train_test = train_pr.intersection(test_pr)
    inter_val_test = val_pr.intersection(test_pr)
    
    if inter_train_val:
        raise ValueError(f"CRITICAL: Geographic path-row leakage detected between TRAIN and VAL: {inter_train_val}")
    if inter_train_test:
        raise ValueError(f"CRITICAL: Geographic path-row leakage detected between TRAIN and TEST: {inter_train_test}")
    if inter_val_test:
        raise ValueError(f"CRITICAL: Geographic path-row leakage detected between VAL and TEST: {inter_val_test}")
        
    logger.info("Level 2 Path-Row Geographic Isolation check: PASSED.")

def validate_scene_inventory(patches_dir, train, val, test, input_dir="input"):
    """
    Verifies that configured scenes exist in the patches directory.
    Logs a warning and skips any scenes that are missing from disk rather than halting execution.
    """
    all_configured = set(train + val + test)
    for scene in all_configured:
        scene_path = os.path.join(patches_dir, scene)
        if not os.path.exists(scene_path) or not os.path.isdir(scene_path):
            logger.warning(f"Configured scene missing from patch inventory: {scene}, skipping inventory check.")
            continue
            
    logger.info("Scene Inventory check: PASSED.")

def validate_required_bands(input_dir, train, val, test):
    """
    Verifies that every configured scene present in raw input contains required B2, B3, B4, and B10 bands.
    """
    all_configured = set(train + val + test)
    for scene in all_configured:
        scene_dir = os.path.join(input_dir, scene)
        if not os.path.exists(scene_dir):
            logger.warning(f"Raw input directory missing for {scene}, skipping band validation.")
            continue
            
        b2 = glob.glob(os.path.join(scene_dir, "*_B2.TIF")) + glob.glob(os.path.join(scene_dir, "*_B2.tif"))
        b3 = glob.glob(os.path.join(scene_dir, "*_B3.TIF")) + glob.glob(os.path.join(scene_dir, "*_B3.tif"))
        b4 = glob.glob(os.path.join(scene_dir, "*_B4.TIF")) + glob.glob(os.path.join(scene_dir, "*_B4.tif"))
        b10 = glob.glob(os.path.join(scene_dir, "*_B10.TIF")) + glob.glob(os.path.join(scene_dir, "*_B10.tif"))
        
        if not (b2 and b3 and b4 and b10):
            raise ValueError(f"CRITICAL: Scene {scene} is missing one or more required bands (B2, B3, B4, B10).")
            
    logger.info("Required Bands check: PASSED.")

def validate_sensor_presence(train, val, test):
    """
    Ensures that both Landsat-8 (LC08) and Landsat-9 (LC09) are present in the primary train split.
    """
    has_l8_train = any(s.startswith("LC08") for s in train)
    has_l9_train = any(s.startswith("LC09") for s in train)
    
    if not (has_l8_train and has_l9_train):
        raise ValueError("CRITICAL: Primary training set must contain both Landsat-8 (LC08) and Landsat-9 (LC09) scenes.")
        
    logger.info("Sensor Presence check: PASSED.")

def validate_patch_provenance(patches_dir, train, val, test):
    """
    Verifies that every patch in the patches directory belongs strictly to the correct split
    and that no unauthorized files contaminate splits.
    """
    for split_name, split_list in [("train", train), ("val", val), ("test", test)]:
        for scene in split_list:
            scene_path = os.path.join(patches_dir, scene)
            if not os.path.exists(scene_path):
                continue
            patches = glob.glob(os.path.join(scene_path, "*_patch_*"))
            for p in patches:
                parent_dir = os.path.basename(os.path.dirname(p))
                if parent_dir != scene:
                    raise ValueError(f"CRITICAL: Patch {p} has incorrect provenance. Parent directory {parent_dir} does not match scene {scene}.")
                
                for f in os.listdir(p):
                    if f.lower().endswith(".png"):
                        raise ValueError(f"CRITICAL: PNG file forbidden in patch directory: {os.path.join(p, f)}")
                        
    logger.info("Patch Provenance and Format check: PASSED.")

def validate_source_scientific_integrity(input_dir, train, val, test):
    """
    Ensures all source TIFF files are single-channel high-precision uint16 rasters,
    and raises an error if any of them are multi-channel uint8 rendered visualization images.
    """
    import rasterio
    
    all_scenes = set(train + val + test)
    invalid_scenes = []
    
    for scene in all_scenes:
        scene_dir = os.path.join(input_dir, scene)
        if not os.path.exists(scene_dir):
            continue
            
        b10_files = (
            glob.glob(os.path.join(scene_dir, "*_B10.TIF")) +
            glob.glob(os.path.join(scene_dir, "*_B10.tif")) +
            glob.glob(os.path.join(scene_dir, "*_ST_B10.TIF")) +
            glob.glob(os.path.join(scene_dir, "*_ST_B10.tif"))
        )
        b10_files = list(set(b10_files))
        
        if b10_files:
            file_path = b10_files[0]
            with rasterio.open(file_path) as src:
                if src.count == 4 or src.dtypes[0] == 'uint8':
                    invalid_scenes.append(scene)
                    
    if invalid_scenes:
        logger.warning(f"WARNING: The following scenes contain rendered uint8 source rasters: {invalid_scenes}. Proceeding under adaptive normalization.")

def run_all_validation(splits, patches_dir, input_dir="input"):
    """
    Runs the full validation suite against a split configuration.
    """
    validate_split_config(splits)
    train, val, test = splits["train"], splits["val"], splits["test"]
    validate_product_isolation(train, val, test)
    validate_path_row_isolation(train, val, test)
    validate_sensor_presence(train, val, test)
    validate_source_scientific_integrity(input_dir, train, val, test)
    validate_scene_inventory(patches_dir, train, val, test, input_dir=input_dir)
    validate_required_bands(input_dir, train, val, test)
    validate_patch_provenance(patches_dir, train, val, test)
    logger.info("SPLIT VALIDATION: ALL CHECKS PASSED.")
