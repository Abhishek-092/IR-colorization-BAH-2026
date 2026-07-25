import pytest
import torch
from training.backbone import ResNetBackbone
from training.sr_head import SRHead
from training.mixture_head import MixtureHead
from inference.pipeline import SUTRAMInferencePipeline

def test_inference_pipeline_forward_shapes():
    """
    Validates that the combined inference pipeline outputs match expected sizes:
    - Input: (B, 1, 256, 256) at 200m
    - Output SR TIR: (B, 1, 512, 512) at 100m
    - Output Colorized: (B, 3, 512, 512) at 100m
    - Variance/Entropy maps: (B, 512, 512)
    """
    B, K = 2, 6
    backbone = ResNetBackbone()
    sr_head = SRHead()
    mix_head = MixtureHead(K=K)
    
    pipeline = SUTRAMInferencePipeline(backbone, sr_head, mix_head, K=K)
    pipeline.eval()
    
    lr_tir = torch.randn(B, 1, 256, 256)
    
    with torch.no_grad():
        sr_tir, decode_outs = pipeline(lr_tir)
        
    assert sr_tir.shape == (B, 1, 512, 512)
    assert decode_outs["dominant_color"].shape == (B, 3, 512, 512)
    assert decode_outs["secondary_color"].shape == (B, 3, 512, 512)
    assert decode_outs["secondary_weight"].shape == (B, 512, 512)
    assert decode_outs["within_mode_variance"].shape == (B, 512, 512)
    assert decode_outs["between_mode_variance"].shape == (B, 512, 512)
    assert decode_outs["entropy"].shape == (B, 512, 512)
    
    print("Inference pipeline shape assertions: PASSED.")

def test_inference_pipeline_with_custom_scale_mode():
    """
    Validates that the pipeline can run using custom scale_mode parameter,
    bypassing dynamic scale detection.
    """
    B, K = 1, 6
    backbone = ResNetBackbone()
    sr_head = SRHead()
    mix_head = MixtureHead(K=K)
    pipeline = SUTRAMInferencePipeline(backbone, sr_head, mix_head, K=K)
    pipeline.eval()
    
    lr_tir = torch.ones(B, 1, 256, 256) * 25000.0 # Raw uint16 scale
    
    with torch.no_grad():
        sr_tir, decode_outs = pipeline(lr_tir, scale_mode="16bit")
        
    assert sr_tir.shape == (B, 1, 512, 512)
    assert decode_outs["dominant_color"].shape == (B, 3, 512, 512)
    print("Inference with custom scale_mode: PASSED.")

def test_vectorized_report_indexing_equivalence():
    """
    Ensures that vectorized np.take_along_axis logic in report.py
    is mathematically equivalent to the nested pixel loops.
    """
    import numpy as np
    K, C, H, W = 6, 3, 32, 32
    
    # Mock output parameters
    pi = np.random.dirichlet(np.ones(K), size=(H, W)).transpose(2, 0, 1) # (K, H, W)
    means_np = np.random.randn(K, C, H, W)
    scales_np = np.random.rand(K, C, H, W)
    
    k_star = np.argmax(pi, axis=0) # (H, W)
    
    # 1. Loop implementation
    dom_mean_loop = np.zeros((C, H, W))
    dom_scale_loop = np.zeros((C, H, W))
    for h in range(H):
        for w in range(W):
            k = k_star[h, w]
            dom_mean_loop[:, h, w] = means_np[k, :, h, w]
            dom_scale_loop[:, h, w] = scales_np[k, :, h, w]
            
    # 2. Vectorized implementation
    k_star_expanded = np.expand_dims(k_star, axis=(0, 1)).repeat(C, axis=1)
    dom_mean_vec = np.take_along_axis(means_np, k_star_expanded, axis=0).squeeze(0)
    dom_scale_vec = np.take_along_axis(scales_np, k_star_expanded, axis=0).squeeze(0)
    
    assert np.allclose(dom_mean_loop, dom_mean_vec)
    assert np.allclose(dom_scale_loop, dom_scale_vec)
    print("Vectorized report indexing equivalence check: PASSED.")

