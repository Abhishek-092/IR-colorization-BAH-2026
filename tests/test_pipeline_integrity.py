import os
import shutil
import tempfile
import numpy as np
import torch
import pytest
from data_pipeline.pipeline_state import (
    PipelineState,
    get_dataset_pipeline_signature,
    get_stage1_pipeline_signature,
    get_stage2_pipeline_signature,
    get_product_input_signature
)
from data_pipeline.dataset_loader import PatchDataset

def test_pipeline_state_hashing_and_validation():
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = os.path.join(tmpdir, "pipeline_state.json")
        state = PipelineState(state_path=state_file)

        # Mock product dir
        prod_dir = os.path.join(tmpdir, "input", "LC09_TEST_PRODUCT")
        os.makedirs(prod_dir, exist_ok=True)

        for b in ["_B2.TIF", "_B3.TIF", "_B4.TIF", "_B10.TIF"]:
            fpath = os.path.join(prod_dir, f"LC09_TEST_PRODUCT{b}")
            with open(fpath, "w") as f:
                f.write("mock_data")

        patches_dir = os.path.join(tmpdir, "patches")
        target_dir = os.path.join(patches_dir, "LC09_TEST_PRODUCT")
        patch_0 = os.path.join(target_dir, "LC09_TEST_PRODUCT_patch_000")
        os.makedirs(patch_0, exist_ok=True)

        np.save(os.path.join(patch_0, "LC09_TEST_PRODUCT_patch_000_rgb_100m.npy"), np.zeros((3, 512, 512), dtype=np.float32))
        np.save(os.path.join(patch_0, "LC09_TEST_PRODUCT_patch_000_tir_100m.npy"), np.zeros((512, 512), dtype=np.float32))
        np.save(os.path.join(patch_0, "LC09_TEST_PRODUCT_patch_000_tir_200m.npy"), np.zeros((256, 256), dtype=np.float32))

        # Initially invalid (not recorded)
        assert not state.is_product_dataset_valid(prod_dir, patches_dir)

        # Record dataset
        state.record_product_dataset(prod_dir, patch_count=1)

        # Now valid
        assert state.is_product_dataset_valid(prod_dir, patches_dir)

        # Modify input file mtime -> should become invalid
        os.utime(os.path.join(prod_dir, "LC09_TEST_PRODUCT_B2.TIF"), (2000000000, 2000000000))
        assert not state.is_product_dataset_valid(prod_dir, patches_dir)

def test_dataset_loader_numerical_integrity():
    with tempfile.TemporaryDirectory() as tmpdir:
        patches_dir = os.path.join(tmpdir, "patches")
        target_dir = os.path.join(patches_dir, "LC09_TEST_PRODUCT")
        patch_0 = os.path.join(target_dir, "LC09_TEST_PRODUCT_patch_000")
        os.makedirs(patch_0, exist_ok=True)

        rgb_raw = np.random.randint(0, 10000, (3, 512, 512)).astype(np.float32)
        tir_100_raw = np.random.randint(20000, 35000, (512, 512)).astype(np.float32)
        tir_200_raw = np.random.randint(20000, 35000, (256, 256)).astype(np.float32)

        np.save(os.path.join(patch_0, "LC09_TEST_PRODUCT_patch_000_rgb_100m.npy"), rgb_raw)
        np.save(os.path.join(patch_0, "LC09_TEST_PRODUCT_patch_000_tir_100m.npy"), tir_100_raw)
        np.save(os.path.join(patch_0, "LC09_TEST_PRODUCT_patch_000_tir_200m.npy"), tir_200_raw)

        dataset = PatchDataset(patches_dir=patches_dir, augment=False)
        assert len(dataset) == 1

        sample = dataset[0]
        assert "tir_200m" in sample
        assert "tir_100m_512" in sample
        assert "rgb_100m_512" in sample

        assert sample["tir_200m"].shape == (1, 256, 256)
        assert sample["tir_100m_512"].shape == (1, 512, 512)
        assert sample["rgb_100m_512"].shape == (3, 512, 512)

        # Check normalization bounds
        assert sample["tir_200m"].min() >= 0.0 and sample["tir_200m"].max() <= 1.0
        assert sample["tir_100m_512"].min() >= 0.0 and sample["tir_100m_512"].max() <= 1.0
        assert sample["rgb_100m_512"].min() >= 0.0 and sample["rgb_100m_512"].max() <= 1.0
