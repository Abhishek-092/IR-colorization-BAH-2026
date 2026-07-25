import os
import glob
import pytest
from omegaconf import OmegaConf
from data_pipeline.split_validator import extract_path_row, run_all_validation
from data_pipeline.dataset_loader import PatchDataset

def load_data_config():
    return OmegaConf.load("configs/data.yaml")

def test_finals_splits_validity():
    """Verify that splits configuration satisfies all the strict requirements."""
    cfg = load_data_config()
    splits = cfg.splits
    
    # 1. Train/val/test categories must exist
    assert "train" in splits
    assert "val" in splits
    assert "test" in splits
    
    train = splits.train
    val = splits.val
    test = splits.test
    
    # 2. Check no generic prefixes or demo
    for name, split_list in [("train", train), ("val", val), ("test", test)]:
        for item in split_list:
            assert item not in ["LC08", "LC09", "demo"], f"Generic item '{item}' found in {name} split."
            assert len(item) > 10, f"Item '{item}' in {name} split does not look like an explicit product ID."

    # 3. Disjoint product IDs (Level 1)
    set_train = set(train)
    set_val = set(val)
    set_test = set(test)
    assert len(set_train.intersection(set_val)) == 0, "Product overlap between train and val!"
    assert len(set_train.intersection(set_test)) == 0, "Product overlap between train and test!"
    assert len(set_val.intersection(set_test)) == 0, "Product overlap between val and test!"

    # 4. Disjoint path/row groups (Level 2)
    train_pr = {extract_path_row(s) for s in train if extract_path_row(s) is not None}
    val_pr = {extract_path_row(s) for s in val if extract_path_row(s) is not None}
    test_pr = {extract_path_row(s) for s in test if extract_path_row(s) is not None}
    
    assert len(train_pr.intersection(val_pr)) == 0, "Geographic path-row overlap between train and val!"
    assert len(train_pr.intersection(test_pr)) == 0, "Geographic path-row overlap between train and test!"
    assert len(val_pr.intersection(test_pr)) == 0, "Geographic path-row overlap between val and test!"

    # 5. Primary training contains both sensors
    assert any(s.startswith("LC08") for s in train), "No Landsat-8 in train!"
    assert any(s.startswith("LC09") for s in train), "No Landsat-9 in train!"

def test_scene_existence_and_bands():
    """Verify that all configured scenes exist on disk and contain required bands."""
    cfg = load_data_config()
    splits = cfg.splits
    patches_dir = cfg.patches_dir
    input_dir = cfg.input_dir
    
    all_configured = set(splits.train + splits.val + splits.test)
    
    for scene in all_configured:
        # Check scene exists in output/patches
        scene_patch_path = os.path.join(patches_dir, scene)
        assert os.path.exists(scene_patch_path), f"Scene patch directory does not exist: {scene_patch_path}"
        assert os.path.isdir(scene_patch_path)
        
        # Check scene exists in input/
        scene_input_path = os.path.join(input_dir, scene)
        if os.path.exists(scene_input_path):
            # Check bands
            b2 = glob.glob(os.path.join(scene_input_path, "*_B2.TIF")) + glob.glob(os.path.join(scene_input_path, "*_B2.tif"))
            b3 = glob.glob(os.path.join(scene_input_path, "*_B3.TIF")) + glob.glob(os.path.join(scene_input_path, "*_B3.tif"))
            b4 = glob.glob(os.path.join(scene_input_path, "*_B4.TIF")) + glob.glob(os.path.join(scene_input_path, "*_B4.tif"))
            b10 = glob.glob(os.path.join(scene_input_path, "*_B10.TIF")) + glob.glob(os.path.join(scene_input_path, "*_B10.tif"))
            
            assert b2, f"Missing B2 in {scene}"
            assert b3, f"Missing B3 in {scene}"
            assert b4, f"Missing B4 in {scene}"
            assert b10, f"Missing B10 in {scene}"

def test_patch_provenance_and_loader_boundaries():
    """Verify that PatchDataset loader isolates split boundaries exactly and traces provenance."""
    cfg = load_data_config()
    patches_dir = cfg.patches_dir
    
    train_dataset = PatchDataset(patches_dir=patches_dir, product_ids=cfg.splits.train)
    val_dataset = PatchDataset(patches_dir=patches_dir, product_ids=cfg.splits.val)
    test_dataset = PatchDataset(patches_dir=patches_dir, product_ids=cfg.splits.test)
    
    train_scenes = set()
    for s in train_dataset.samples:
        parent_dir = os.path.basename(os.path.dirname(s))
        assert parent_dir in cfg.splits.train, f"Train patch {s} loaded from unconfigured scene '{parent_dir}'."
        train_scenes.add(parent_dir)
        
    val_scenes = set()
    for s in val_dataset.samples:
        parent_dir = os.path.basename(os.path.dirname(s))
        assert parent_dir in cfg.splits.val, f"Val patch {s} loaded from unconfigured scene '{parent_dir}'."
        val_scenes.add(parent_dir)
        
    test_scenes = set()
    for s in test_dataset.samples:
        parent_dir = os.path.basename(os.path.dirname(s))
        assert parent_dir in cfg.splits.test, f"Test patch {s} loaded from unconfigured scene '{parent_dir}'."
        test_scenes.add(parent_dir)
        
    # Cross-loader isolation
    assert len(train_scenes.intersection(val_scenes)) == 0, "Loader boundary breach! Train loader contains val scenes."
    assert len(train_scenes.intersection(test_scenes)) == 0, "Loader boundary breach! Train loader contains test scenes."
    assert len(val_scenes.intersection(test_scenes)) == 0, "Loader boundary breach! Val loader contains test scenes."

def test_split_validator_helper():
    """Verify that split_validator.py behaves correctly and catches duplicate directories."""
    cfg = load_data_config()
    splits_dict = OmegaConf.to_container(cfg.splits, resolve=True)
    patches_dir = cfg.patches_dir
    
    # Standard check should pass
    run_all_validation(splits_dict, patches_dir, input_dir=cfg.input_dir)
    
    # Corrupt check by leaking a scene should raise ValueError
    corrupt_splits = {
        "train": list(splits_dict["train"]),
        "val": list(splits_dict["val"]) + [splits_dict["train"][0]],
        "test": list(splits_dict["test"])
    }
    with pytest.raises(ValueError, match="Product leakage detected"):
        run_all_validation(corrupt_splits, patches_dir, input_dir=cfg.input_dir)
