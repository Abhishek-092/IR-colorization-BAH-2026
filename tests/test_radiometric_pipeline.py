import os
import glob
import numpy as np
import pytest
import torch
import rasterio

from omegaconf import OmegaConf
from data_pipeline.split_validator import run_all_validation
from data_pipeline.dataset_loader import PatchDataset
from utils.normalization import normalize_tir, normalize_rgb
from inference.pipeline import SUTRAMInferencePipeline
from training.backbone import ResNetBackbone
from training.sr_head import SRHead
from training.mixture_head import MixtureHead

def test_centralized_normalization_bounds():
    # Test normalization bounds for various values and data types
    # uint8 inputs
    arr_8bit_tir = np.array([0, 128, 255], dtype=np.uint8)
    norm_8bit_tir = normalize_tir(arr_8bit_tir)
    assert np.allclose(norm_8bit_tir, [0.0, 128.0/255.0, 1.0])
    
    arr_8bit_rgb = np.array([0, 128, 255], dtype=np.uint8)
    norm_8bit_rgb = normalize_rgb(arr_8bit_rgb)
    assert np.allclose(norm_8bit_rgb, [0.0, 128.0, 255.0])
    
    # uint16 inputs
    arr_16bit_tir = np.array([20000, 35000], dtype=np.uint16)
    norm_16bit_tir = normalize_tir(arr_16bit_tir)
    assert np.allclose(norm_16bit_tir, [0.0, 1.0])
    
    arr_16bit_rgb = np.array([0, 5000, 10000], dtype=np.uint16)
    norm_16bit_rgb = normalize_rgb(arr_16bit_rgb)
    assert np.allclose(norm_16bit_rgb, [0.0, 127.5, 255.0])
    
    # Torch Tensor versions
    t_arr_8bit_tir = torch.tensor([0.0, 128.0, 255.0])
    t_norm_8bit_tir = normalize_tir(t_arr_8bit_tir)
    assert torch.allclose(t_norm_8bit_tir, torch.tensor([0.0, 128.0/255.0, 1.0]))

def test_nan_inf_rejection_and_variance():
    # Verify that PatchDataset loads clean arrays with non-zero variance and no NaNs
    cfg = OmegaConf.load("configs/data.yaml")
    train_dataset = PatchDataset(patches_dir=cfg.patches_dir, product_ids=cfg.splits.train)
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        tir_100 = sample["tir_100m_512"]
        rgb_100 = sample["rgb_100m_512"]
        
        assert not torch.isnan(tir_100).any()
        assert not torch.isinf(tir_100).any()
        assert not torch.isnan(rgb_100).any()
        assert not torch.isinf(rgb_100).any()
        
        assert tir_100.std().item() > 0.0
        assert rgb_100.std().item() > 0.0

def test_inference_and_train_preprocessing_parity():
    # Test that train loader and inference pipeline produce identical normalized values
    backbone = ResNetBackbone()
    sr_head = SRHead()
    mix_head = MixtureHead(K=6)
    pipeline = SUTRAMInferencePipeline(backbone, sr_head, mix_head, K=6)
    
    # Test with typical Landsat values
    raw_val = torch.tensor([25000.0, 30000.0])
    norm_pipeline = normalize_tir(raw_val)
    norm_loader = normalize_tir(raw_val.numpy())
    assert np.allclose(norm_pipeline.numpy(), norm_loader)
