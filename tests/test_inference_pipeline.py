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
